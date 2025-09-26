#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera+Laser-based muzzle calibration + tracking
- 레이저 점을 "총구 위치"로 사용하여 카메라-기계 시점(오프셋) 보정
- 목표(빨간색 물체)를 찾고 레이저→목표 오프셋을 서보 명령으로 보냄
"""

from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time, math, serial

# ---------------- Config ----------------
HFOV_DEG = 66.0
VFOV_DEG = 41.0
FRAME_W, FRAME_H = 1280, 720
TARGET_FPS = 30
MIRROR = True

# area/circularity thresholds
AREA_FRAC_MIN = 0.0005
AREA_FRAC_MAX = 0.20
CIRC_MIN = 0.80

# morphology
OPEN_K = 5
CLOSE_K = 7
OPEN_ITERS = 1
CLOSE_ITERS = 2

# tracking/motion
SMOOTH_ALPHA = 0.4
MIN_MOVE_DEG = 1.0
DEADZONE_PX  = 8
RANGE_CM     = 200.0
H_CM_PER_DEG = 1.32   # 사용자가 설정했던 값 (cm / degree)
V_CM_PER_DEG = 1.32

# UART
UART_PAN  = '/dev/ttyAMA2'
UART_TILT = '/dev/ttyAMA3'
BAUDRATE  = 115200
SER_TIMEOUT = 0.1

# servo limits & centers
SERVO_MIN = 0
SERVO_MAX = 180
SERVO_CENTER_PAN  = 90
SERVO_CENTER_TILT = 90

SIGN_PAN  = -1
SIGN_TILT = -1
PAN_MIN,  PAN_MAX  = 0, 180
TILT_MIN, TILT_MAX = 75, 180

# ★ 서보 기준 정조준 보정값 (하드웨어별로 조정)
PAN_OFFSET  = 119
TILT_OFFSET = 98

# muzzle fallback (if no laser detected)
MUZZLE_PX = FRAME_W // 2
MUZZLE_PY = FRAME_H - 100

# scoring weights
W_CIRC  = 0.40
W_COLOR = 0.60
WC_H = 0.15
WC_S = 0.65
WC_V = 0.20

# target (red) color band (HSV)
H_LO, H_HI = 0.0, 15.0   # 적색 계열 (예시)
S_LO, S_HI = 100.0, 255.0
V_LO, V_HI = 120.0, 255.0

H_SIGMA_OUT = 6.0
S_SIGMA_OUT = 15.0
V_SIGMA_OUT = 15.0

# ---------- Utility / Vision helpers ----------
def make_mask_from_bgr(frame_bgr, use_clahe=True):
    """기존 타깃(빨강 물체)을 위한 마스크"""
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    if use_clahe:
        h, s, v = cv.split(hsv)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv.merge([h, s, v])
    def or_ranges(hsv_img, ranges):
        m = None
        for lo, hi in ranges:
            cur = cv.inRange(hsv_img, np.array(lo), np.array(hi))
            m = cur if m is None else cv.bitwise_or(m, cur)
        return m
    red_bright = [((0, int(S_LO), int(V_LO)), (15, 255, 255)), ((165, int(S_LO), int(V_LO)), (180, 255, 255))]
    m_red = cv.bitwise_or(or_ranges(hsv, red_bright), or_ranges(hsv, red_bright))
    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    mask = cv.morphologyEx(m_red, cv.MORPH_OPEN,  k_open,  iterations=OPEN_ITERS)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_close, iterations=CLOSE_ITERS)
    return mask

def find_laser_point(frame_bgr):
    """
    레이저 점 검출: 보통 매우 밝고 포화된 빨간 점.
    반환: (x, y) 중심 픽셀 좌표 또는 None
    """
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # 조건: V 매우 큼, S 큼, H는 빨강대역
    # 임계값은 환경에 따라 조정 필요
    lower1 = np.array((0, 150, 200), dtype=np.uint8)
    upper1 = np.array((10, 255, 255), dtype=np.uint8)
    lower2 = np.array((170, 150, 200), dtype=np.uint8)
    upper2 = np.array((180, 255, 255), dtype=np.uint8)
    m1 = cv.inRange(hsv, lower1, upper1)
    m2 = cv.inRange(hsv, lower2, upper2)
    mask = cv.bitwise_or(m1, m2)
    # 노이즈 제거
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, k_open, iterations=1)
    # 작은 블롭 중 가장 밝고 작은 영역(레이저 점)을 선택
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # 가장 면적이 작은(점에 가까운) 컨투어 선택하되, 너무 작으면 무시
    best = None
    for c in cnts:
        a = cv.contourArea(c)
        if a < 1:  # 잡음
            continue
        if best is None or a < best[0]:
            best = (a, c)
    if best is None:
        return None
    _, c = best
    M = cv.moments(c)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def circularity_from_contour(cnt):
    area = cv.contourArea(cnt)
    perim = cv.arcLength(cnt, True)
    if perim <= 0:
        return 0.0, area
    circ = 4.0 * math.pi * area / (perim * perim)
    return circ, area

def mean_hsv_in_hull(frame_bgr, hull):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    poly = hull.reshape(-1, 2)
    cv.fillConvexPoly(mask, poly, 255)
    m = mask > 0
    if not np.any(m): return None
    h, s, v = cv.split(hsv)
    Hm = float(np.mean(h[m]))
    Sm = float(np.mean(s[m]))
    Vm = float(np.mean(v[m]))
    return (Hm, Sm, Vm)

def band_score(val, lo, hi, sigma_out):
    if lo <= val <= hi: return 1.0
    if val < lo: d = lo - val
    else: d = val - hi
    return math.exp(-0.5 * (d / max(1e-6, sigma_out))**2)

def color_score_band(Hm, Sm, Vm):
    scH = band_score(Hm, H_LO, H_HI, H_SIGMA_OUT)
    scS = band_score(Sm, S_LO, S_HI, S_SIGMA_OUT)
    scV = band_score(Vm, V_LO, V_HI, V_SIGMA_OUT)
    num = WC_H*scH + WC_S*scS + WC_V*scV
    den = WC_H + WC_S + WC_V
    return max(0.0, min(1.0, num/den)), (scH, scS, scV)

def pick_best_target(mask, frame_area_min, frame_area_max, circ_min, frame_bgr=None):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        hull = cv.convexHull(c)
        circ, area = circularity_from_contour(hull)
        if area < frame_area_min or area > frame_area_max: continue
        if circ < circ_min: continue
        M = cv.moments(hull)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        hsv_mean = None; col_sc = 0.0; scH=scS=scV=0.0
        if frame_bgr is not None:
            hsv_mean = mean_hsv_in_hull(frame_bgr, hull)
            if hsv_mean is not None:
                Hm, Sm, Vm = hsv_mean
                col_sc, (scH, scS, scV) = color_score_band(Hm, Sm, Vm)
        total_sc = (W_CIRC * circ) + (W_COLOR * col_sc)
        cand = {
            "score": total_sc, "score_breakdown": {"circ": circ, "color": col_sc, "H": scH, "S": scS, "V": scV},
            "circularity": circ, "area": int(area), "center": (cx, cy), "hull": hull,
        }
        if (best is None) or (cand["score"] > best["score"]):
            best = cand
    return best

# ---------- Geometry / conversion ----------
def px_to_deg_with_muzzle(target_px, target_py, muzzle_px, muzzle_py, width, height):
    dx_px = float(target_px - muzzle_px)
    dy_px = float(target_py - muzzle_py)
    deg_x = (dx_px / float(width))  * HFOV_DEG
    deg_y = (dy_px / float(height)) * VFOV_DEG
    return deg_x, deg_y

def angle_to_cm(deg_x, deg_y, R_cm):
    # 간단선형 근사: deg -> cm (사용자가 정의한 H_CM_PER_DEG 사용)
    dx_cm = deg_x * H_CM_PER_DEG
    dy_cm = deg_y * V_CM_PER_DEG
    return dx_cm, dy_cm

def cm_to_motor_deg(dx_cm, dy_cm):
    need_deg_x = dx_cm / max(1e-6, H_CM_PER_DEG)
    need_deg_y = dy_cm / max(1e-6, V_CM_PER_DEG)
    def quantize(v):
        mag = abs(v)
        if mag < MIN_MOVE_DEG:
            return 0
        return int(round(v))
    return quantize(need_deg_x), quantize(need_deg_y)

def delta_to_servo_angles(cmd_deg_x, cmd_deg_y):
    pan  = SERVO_CENTER_PAN  + SIGN_PAN  * cmd_deg_x
    tilt = SERVO_CENTER_TILT + SIGN_TILT * cmd_deg_y
    pan  = max(PAN_MIN,  min(PAN_MAX,  int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    return pan, tilt

# ---------- UART helper ----------
class MotorUART:
    def __init__(self, port_pan, port_tilt, baud, timeout):
        try:
            self.ser_pan  = serial.Serial(port_pan,  baud, timeout=timeout)
        except Exception as e:
            print("WARN: cannot open pan serial:", e)
            self.ser_pan = None
        try:
            self.ser_tilt = serial.Serial(port_tilt, baud, timeout=timeout)
        except Exception as e:
            print("WARN: cannot open tilt serial:", e)
            self.ser_tilt = None
        time.sleep(0.2)
        self.last_pan_angle  = None
        self.last_tilt_angle = None

    def send_servo(self, pan_angle, tilt_angle):
        pan_angle  = max(PAN_MIN,  min(PAN_MAX,  int(pan_angle)))
        tilt_angle = max(TILT_MIN, min(TILT_MAX, int(tilt_angle)))
        # 간단히 항상 두 포트에 쓰기 (하드웨어 프로토콜에 맞춰 수정 가능)
        if self.ser_pan:
            try:
                self.ser_pan.write(f"{pan_angle}\n".encode('ascii'))
                self.last_pan_angle = pan_angle
            except Exception as e:
                print("pan write err:", e)
        if self.ser_tilt:
            try:
                self.ser_tilt.write(f"{tilt_angle}\n".encode('ascii'))
                self.last_tilt_angle = tilt_angle
            except Exception as e:
                print("tilt write err:", e)

    def close(self):
        for s in (self.ser_pan, self.ser_tilt):
            try:
                if s: s.close()
            except:
                pass

# ---------- Main ----------
def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    motors = MotorUART(UART_PAN, UART_TILT, BAUDRATE, SER_TIMEOUT)

    cx0, cy0 = FRAME_W // 2, FRAME_H // 2
    ema_cx, ema_cy = None, None

    frame_area = FRAME_W * FRAME_H
    AREA_MIN = int(AREA_FRAC_MIN * frame_area)
    AREA_MAX = int(AREA_FRAC_MAX * frame_area)

    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            if MIRROR:
                frame = cv.flip(frame, 1)

            mask = make_mask_from_bgr(frame, use_clahe=True)
            target = pick_best_target(mask, AREA_MIN, AREA_MAX, CIRC_MIN, frame_bgr=frame)
            laser_pt = find_laser_point(frame)   # 레이저 좌표(카메라 기준)

            # muzzle 기준: 레이저가 있으면 그걸 사용, 없으면 고정 MUZZLE
            if laser_pt is not None:
                muzzle_x, muzzle_y = laser_pt
            else:
                muzzle_x, muzzle_y = MUZZLE_PX, MUZZLE_PY

            vis = frame.copy()
            # 시각화: 카메라 중앙
            cv.drawMarker(vis, (cx0, cy0), (255, 255, 255), markerType=cv.MARKER_CROSS, markerSize=24, thickness=2)
            cv.circle(vis, (cx0, cy0), 3, (0, 0, 255), -1)

            # 시각화: 레이저(총구) 위치
            cv.drawMarker(vis, (int(muzzle_x), int(muzzle_y)), (0,255,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=24, thickness=2)
            if laser_pt is not None:
                cv.putText(vis, "LASER", (int(muzzle_x)+6, int(muzzle_y)-6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            if target is not None:
                (bx, by, bw, bh) = cv.boundingRect(target["hull"])
                (cx, cy) = target["center"]
                area = target["area"]
                circ = target["circularity"]
                hull = target["hull"]

                # EMA smoothing
                if ema_cx is None:
                    ema_cx, ema_cy = cx, cy
                else:
                    ema_cx = int(SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * ema_cx)
                    ema_cy = int(SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * ema_cy)

                dx_px = ema_cx - cx0
                dy_px = ema_cy - cy0
                if abs(dx_px) < DEADZONE_PX:
                    dx_px = 0
                if abs(dy_px) < DEADZONE_PX:
                    dy_px = 0

                # 여기서 muzzle(=laser) 기준으로 각도 계산
                deg_x, deg_y = px_to_deg_with_muzzle(ema_cx, ema_cy, muzzle_x, muzzle_y, FRAME_W, FRAME_H)
                dx_cm, dy_cm = angle_to_cm(deg_x, deg_y, RANGE_CM)
                cmd_deg_x, cmd_deg_y = cm_to_motor_deg(dx_cm, dy_cm)

                # --- 서보 기준 정조준 보정 적용 ---
                servo_pan, servo_tilt = delta_to_servo_angles(cmd_deg_x, cmd_deg_y)
                servo_pan  = PAN_OFFSET  + (servo_pan  - SERVO_CENTER_PAN)
                servo_tilt = TILT_OFFSET + (servo_tilt - SERVO_CENTER_TILT)
                servo_pan  = max(PAN_MIN,  min(PAN_MAX,  int(servo_pan)))
                servo_tilt = max(TILT_MIN, min(TILT_MAX, int(servo_tilt)))

                # 전송
                motors.send_servo(servo_pan, servo_tilt)

                # --- 시각화 ---
                cv.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv.circle(vis, (ema_cx, ema_cy), 5, (0, 0, 255), -1)
                cv.drawContours(vis, [hull], -1, (255, 0, 255), 2)

                y0 = 28; dy = 28
                cv.putText(vis, f"area:{area}  circ:{circ:.3f}  score:{target['score']:.3f}",
                           (bx, max(0, by - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(vis, f"offset_px:   ({dx_px:+d}, {dy_px:+d})", (10, y0), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)
                cv.putText(vis, f"offset_deg:  ({deg_x:+.2f}, {deg_y:+.2f})", (10, y0+dy), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)
                cv.putText(vis, f"offset_cm*:  ({dx_cm:+.1f}, {dy_cm:+.1f})  @R={RANGE_CM:.0f}cm", (10, y0+2*dy), cv.FONT_HERSHEY_SIMPLEX, 0.7, (50,220,50), 2)
                cv.putText(vis, f"motor_cmd:   panΔ={cmd_deg_x:+d}°, tiltΔ={cmd_deg_y:+d}°", (10, y0+3*dy), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv.putText(vis, f"servo_out:   pan={servo_pan:3d}°, tilt={servo_tilt:3d}° (0~180)", (10, y0+4*dy), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv.circle(vis, (cx, cy), 6, (0,0,255), -1)
            else:
                ema_cx = ema_cy = None
                cv.putText(vis, "No target", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # FPS
            now = time.time()
            dt  = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            cv.putText(vis, f"FPS: {fps:.1f}", (10, FRAME_H - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv.imshow("frame", vis)
            cv.imshow("mask", mask)
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv.destroyAllWindows()
        try:
            picam2.stop()
        except:
            pass
        motors.close()

if __name__ == "__main__":
    main()
