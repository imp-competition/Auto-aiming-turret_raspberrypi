#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
타겟 중앙 -> 총구 정조준 보정 (offsets.csv 사용)
- offsets.csv가 있으면 거리별 보간(또는 평균)으로 muzzle offset을 계산
- ToF 센서가 있으면 get_distance_cm()을 연결하면 거리별 보정 사용
"""

from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time, math, serial, csv, os

# ---------- Config ----------
HFOV_DEG = 66.0
VFOV_DEG = 41.0
FRAME_W, FRAME_H = 1280, 720
MIRROR = True

AREA_FRAC_MIN = 0.0005
AREA_FRAC_MAX = 0.30
CIRC_MIN = 0.80

SMOOTH_ALPHA = 0.4
DEADZONE_PX  = 8
RANGE_CM     = 200.0
H_CM_PER_DEG = 1.32
V_CM_PER_DEG = 1.32

# UART / servo
UART_PAN  = '/dev/ttyAMA2'
UART_TILT = '/dev/ttyAMA3'
BAUDRATE  = 115200
SER_TIMEOUT = 0.1

SERVO_CENTER_PAN  = 90
SERVO_CENTER_TILT = 90
SIGN_PAN  = -1
SIGN_TILT = -1
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 75, 180

PAN_OFFSET  = 119   # hardware 정조준 상수 (필요하면 조정)
TILT_OFFSET = 98

# offsets CSV path (생성 툴에서 만든 파일)
OFF_CSV = "offsets.csv"

# If no CSV / no distance known, use these fallback offsets (pixels)
DEFAULT_DX = 0
DEFAULT_DY = 0

# ---------- Vision helpers ----------
def make_mask_from_bgr(frame_bgr, use_clahe=True):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    if use_clahe:
        h, s, v = cv.split(hsv)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        v = clahe.apply(v)
        hsv = cv.merge([h, s, v])
    # red ranges (tweak if needed)
    red1 = cv.inRange(hsv, np.array((0, 80, 80)), np.array((15, 255, 255)))
    red2 = cv.inRange(hsv, np.array((165, 80, 80)), np.array((180, 255, 255)))
    m_red = cv.bitwise_or(red1, red2)
    k_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    mask = cv.morphologyEx(m_red, cv.MORPH_OPEN, k_open, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_close, iterations=2)
    return mask

def circularity_from_contour(cnt):
    area = cv.contourArea(cnt)
    perim = cv.arcLength(cnt, True)
    if perim <= 0:
        return 0.0, area
    circ = 4.0 * math.pi * area / (perim * perim)
    return circ, area

def pick_best_target(mask, frame_area_min, frame_area_max, circ_min, frame_bgr=None):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        hull = cv.convexHull(c)
        circ, area = circularity_from_contour(hull)
        if area < frame_area_min or area > frame_area_max: continue
        if circ < circ_min: continue
        M = cv.moments(hull)
        if M.get("m00",0) == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        cand = {"score": circ, "circularity": circ, "area": int(area), "center": (cx, cy), "hull": hull}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best

def px_to_deg_with_muzzle(target_px, target_py, muzzle_px, muzzle_py, width, height):
    dx_px = float(target_px - muzzle_px)
    dy_px = float(target_py - muzzle_py)
    deg_x = (dx_px / float(width))  * HFOV_DEG
    deg_y = (dy_px / float(height)) * VFOV_DEG
    return deg_x, deg_y

def angle_to_cm(deg_x, deg_y, R_cm):
    dx_cm = deg_x * H_CM_PER_DEG
    dy_cm = deg_y * V_CM_PER_DEG
    return dx_cm, dy_cm

def cm_to_motor_deg(dx_cm, dy_cm):
    need_deg_x = dx_cm / max(1e-6, H_CM_PER_DEG)
    need_deg_y = dy_cm / max(1e-6, V_CM_PER_DEG)
    def quantize(v):
        mag = abs(v)
        if mag < 1.0: return 0
        return int(round(v))
    return quantize(need_deg_x), quantize(need_deg_y)

def delta_to_servo_angles(cmd_deg_x, cmd_deg_y):
    pan  = SERVO_CENTER_PAN  + SIGN_PAN  * cmd_deg_x
    tilt = SERVO_CENTER_TILT + SIGN_TILT * cmd_deg_y
    pan  = max(PAN_MIN,  min(PAN_MAX,  int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    return pan, tilt

# ---------- Offsets loader / interpolator ----------
def load_offsets(csv_path):
    """
    CSV expected columns: timestamp,distance_cm,muzzle_x,muzzle_y,target_x,target_y,dx,dy
    Returns sorted list of (distance_cm_or_None, dx, dy)
    """
    out = []
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, newline='') as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        for row in rdr:
            if len(row) < 8: continue
            _, dist_s, _, _, _, _, dx_s, dy_s = row[:8]
            try:
                dist = float(dist_s) if dist_s.strip() != "" else None
                dx = float(dx_s); dy = float(dy_s)
                out.append((dist, dx, dy))
            except:
                continue
    # sort by distance (None goes last)
    out.sort(key=lambda x: (float('inf') if x[0] is None else x[0]))
    return out

def get_offset_for_distance(offsets_list, distance_cm=None):
    """
    offsets_list: list of (dist_or_None, dx, dy), sorted by dist
    If distance_cm provided and there are at least two distanceed entries -> linear interp.
    Else return average of available offsets (or default)
    """
    if not offsets_list:
        return DEFAULT_DX, DEFAULT_DY

    # collect distance-known entries
    dist_entries = [(d,dx,dy) for (d,dx,dy) in offsets_list if d is not None]
    if distance_cm is not None and len(dist_entries) >= 2:
        # find bounding interval
        for i in range(len(dist_entries)-1):
            d0, dx0, dy0 = dist_entries[i]
            d1, dx1, dy1 = dist_entries[i+1]
            if d0 <= distance_cm <= d1:
                t = (distance_cm - d0) / max(1e-6, (d1 - d0))
                dx = dx0 + t*(dx1 - dx0)
                dy = dy0 + t*(dy1 - dy0)
                return dx, dy
        # if outside range, use nearest
        if distance_cm < dist_entries[0][0]:
            return dist_entries[0][1], dist_entries[0][2]
        else:
            return dist_entries[-1][1], dist_entries[-1][2]

    # else: use average of all dx,dy (including None-dist ones)
    dxs = [dx for (_,dx,_) in offsets_list]
    dys = [dy for (_,_,dy) in offsets_list]
    return float(sum(dxs))/len(dxs), float(sum(dys))/len(dys)

# ---------- UART helper ----------
class MotorUART:
    def __init__(self, port_pan, port_tilt, baud, timeout):
        try:
            self.ser_pan  = serial.Serial(port_pan,  baud, timeout=timeout)
        except Exception as e:
            print("WARN: cannot open pan serial:", e); self.ser_pan = None
        try:
            self.ser_tilt = serial.Serial(port_tilt, baud, timeout=timeout)
        except Exception as e:
            print("WARN: cannot open tilt serial:", e); self.ser_tilt = None
        time.sleep(0.2)
    def send_servo(self, pan_angle, tilt_angle):
        pan_angle = max(PAN_MIN, min(PAN_MAX, int(pan_angle)))
        tilt_angle = max(TILT_MIN, min(TILT_MAX, int(tilt_angle)))
        if self.ser_pan:
            try: self.ser_pan.write(f"{pan_angle}\n".encode('ascii'))
            except Exception as e: print("pan write err:", e)
        if self.ser_tilt:
            try: self.ser_tilt.write(f"{tilt_angle}\n".encode('ascii'))
            except Exception as e: print("tilt write err:", e)
    def close(self):
        for s in (self.ser_pan, self.ser_tilt):
            try:
                if s: s.close()
            except: pass

# ---------- Optional: get distance (ToF) ----------
def get_distance_cm():
    """
    만약 ToF 센서(VL53L7CX 등)를 연결해 거리를 읽을 수 있다면
    여기에서 읽어서 return (float cm). 현재 기본은 None (거리 미사용).
    """
    # TODO: 실제 ToF 읽기 코드를 넣을 것
    return None

# ---------- Main ----------
def main():
    offsets = load_offsets(OFF_CSV)
    print("Loaded offsets:", offsets)
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)})
    picam2.configure(config); picam2.start()

    motors = MotorUART(UART_PAN, UART_TILT, BAUDRATE, SER_TIMEOUT)

    frame_area = FRAME_W * FRAME_H
    AREA_MIN = int(AREA_FRAC_MIN * frame_area)
    AREA_MAX = int(AREA_FRAC_MAX * frame_area)

    cx0, cy0 = FRAME_W//2, FRAME_H//2
    ema_cx = ema_cy = None
    prev_t = time.time(); fps = 0.0

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            if MIRROR: frame = cv.flip(frame, 1)

            mask = make_mask_from_bgr(frame, use_clahe=True)
            target = pick_best_target(mask, AREA_MIN, AREA_MAX, CIRC_MIN, frame_bgr=frame)

            # 거리(있으면) 읽기
            dist = get_distance_cm()  # None이면 거리미사용
            # 현재 보정 dx,dy
            dx_offset, dy_offset = get_offset_for_distance(offsets, dist)

            vis = frame.copy()
            # draw center
            cv.drawMarker(vis, (cx0, cy0), (255,255,255), cv.MARKER_CROSS, 24, 2)

            if target is not None:
                (cx, cy) = target["center"]
                hull = target["hull"]
                if ema_cx is None:
                    ema_cx, ema_cy = cx, cy
                else:
                    ema_cx = int(SMOOTH_ALPHA * cx + (1-SMOOTH_ALPHA)*ema_cx)
                    ema_cy = int(SMOOTH_ALPHA * cy + (1-SMOOTH_ALPHA)*ema_cy)

                # muzzle predicted from offset: muzzle = target + (dx,dy)
                muzzle_x = int(round(ema_cx + dx_offset))
                muzzle_y = int(round(ema_cy + dy_offset))

                # visualize muzzle predicted
                cv.drawMarker(vis, (muzzle_x, muzzle_y), (0,255,255), cv.MARKER_TILTED_CROSS, 24, 2)
                cv.putText(vis, f"OFF(dx,dy)=({dx_offset:+.1f},{dy_offset:+.1f})", (10,30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                # compute angle from muzzle->target (we want to move turret so muzzle goes to target)
                deg_x, deg_y = px_to_deg_with_muzzle(ema_cx, ema_cy, muzzle_x, muzzle_y, FRAME_W, FRAME_H)
                dx_cm, dy_cm = angle_to_cm(deg_x, deg_y, RANGE_CM)
                cmd_deg_x, cmd_deg_y = cm_to_motor_deg(dx_cm, dy_cm)

                # servo angle output
                servo_pan, servo_tilt = delta_to_servo_angles(cmd_deg_x, cmd_deg_y)
                servo_pan  = PAN_OFFSET  + (servo_pan  - SERVO_CENTER_PAN)
                servo_tilt = TILT_OFFSET + (servo_tilt - SERVO_CENTER_TILT)
                servo_pan  = max(PAN_MIN,  min(PAN_MAX,  int(servo_pan)))
                servo_tilt = max(TILT_MIN, min(TILT_MAX, int(servo_tilt)))

                # send to motors
                motors.send_servo(servo_pan, servo_tilt)

                # draw target
                cv.circle(vis, (ema_cx, ema_cy), 5, (0,0,255), -1)
                cv.drawContours(vis, [hull], -1, (255,0,255), 2)
                cv.putText(vis, f"servo out pan={servo_pan},tilt={servo_tilt}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            else:
                ema_cx = ema_cy = None
                cv.putText(vis, "No target", (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # fps
            now = time.time(); dt = now - prev_t; prev_t = now
            if dt>0:
                fps = 0.9*fps + 0.1*(1.0/dt) if fps>0 else (1.0/dt)
            cv.putText(vis, f"FPS:{fps:.1f}", (10, FRAME_H-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv.imshow("frame", vis); cv.imshow("mask", mask)
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        cv.destroyAllWindows()
        try: picam2.stop()
        except: pass
        motors.close()

if __name__ == "__main__":
    main()
