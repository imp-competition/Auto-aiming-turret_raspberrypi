from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import math
import serial

# =========================================================
# 기본 설정 (원본 초기값 유지)
# =========================================================
HFOV_DEG = 66.0
VFOV_DEG = 41.0
FRAME_W, FRAME_H = 1280, 720
TARGET_FPS = 30
MIRROR = True

AREA_FRAC_MIN = 0.0003
AREA_FRAC_MAX = 0.30
CIRC_MIN = 0.80

# 모폴로지
OPEN_K = 5
CLOSE_K = 7
OPEN_ITERS = 1
CLOSE_ITERS = 2

# 추적/모터
SMOOTH_ALPHA = 0.4
MIN_MOVE_DEG = 1.0        # <= 원본 유지 (아래 로직에선 사용하지 않음)
DEADZONE_PX  = 8          # <= 원본 유지 (아래 로직에선 사용하지 않음)
RANGE_CM     = 200.0
H_CM_PER_DEG = 1.32
V_CM_PER_DEG = 1.32

# UART
UART_PAN  = '/dev/ttyAMA2'
UART_TILT = '/dev/ttyAMA3'
BAUDRATE  = 115200
SER_TIMEOUT = 0.1

# 서보
SERVO_MIN = 0
SERVO_MAX = 180
SERVO_CENTER_PAN  = 90
SERVO_CENTER_TILT = 90
SIGN_PAN  = +1
SIGN_TILT = -1

PAN_MIN,  PAN_MAX  = 0, 180
TILT_MIN, TILT_MAX = 75, 180

# =========================================================
# 점수 가중치 (원형 + 색)
# =========================================================
W_CIRC  = 0.40
W_COLOR = 0.60

WC_H = 0.15
WC_S = 0.65
WC_V = 0.20

H_LO, H_HI = 3.0, 11.0
S_LO, S_HI = 180.0, 199.0
V_LO, V_HI = 175.0, 199.0

H_SIGMA_OUT = 6.0
S_SIGMA_OUT = 15.0
V_SIGMA_OUT = 15.0

# =========================================================
# (추가) 런타임 튜닝 파라미터 (초기값은 원본 동작 훼손 없이 설정)
# =========================================================
KpX = 1.00   # 좌우 비례 이득
KpY = 1.00   # 상하 비례 이득
KvX = 0.30   # 좌우 속도 이득
KvY = 0.30   # 상하 속도 이득
MAX_STEP_DEG = 0        # 0 = 제한 없음 (9/0 키로 조절)
ALPHA_V = 0.6           # 속도 추정 EMA

# =========================================================
# 마스크 생성 (빨강 계열 후보)
# =========================================================
def make_mask_from_bgr(frame_bgr, use_clahe=True):
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

    red_bright = [((0,130,80),(15,255,255)), ((165,130,80),(180,255,255))]
    red_dark   = [((0, 90,40),(15,255,255)), ((165, 90,40),(180,255,255))]
    m_red = cv.bitwise_or(or_ranges(hsv, red_bright), or_ranges(hsv, red_dark))

    mask = m_red

    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN,  k_open,  iterations=OPEN_ITERS)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, k_close, iterations=CLOSE_ITERS)
    return mask

# =========================================================
# 기하/보조
# =========================================================
def circularity_from_contour(cnt):
    area = cv.contourArea(cnt)
    perim = cv.arcLength(cnt, True)
    if perim <= 0:
        return 0.0, area
    circ = 4.0 * math.pi * area / (perim * perim)
    return circ, area

def px_to_deg(dx_px, dy_px, width, height):
    deg_x = (dx_px / float(width))  * HFOV_DEG
    deg_y = (dy_px / float(height)) * VFOV_DEG
    return deg_x, deg_y

def angle_to_cm(deg_x, deg_y, range_cm):
    dx_cm = range_cm * math.tan(math.radians(deg_x))
    dy_cm = range_cm * math.tan(math.radians(deg_y))
    return dx_cm, dy_cm

def mean_hsv_in_hull(frame_bgr, hull):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    poly = hull.reshape(-1, 2)
    cv.fillConvexPoly(mask, poly, 255)
    m = mask > 0
    if not np.any(m):
        return None
    h, s, v = cv.split(hsv)
    Hm = float(np.mean(h[m]))
    Sm = float(np.mean(s[m]))
    Vm = float(np.mean(v[m]))
    return (Hm, Sm, Vm)

def band_score(val, lo, hi, sigma_out):
    if lo <= val <= hi:
        return 1.0
    d = (lo - val) if val < lo else (val - hi)
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
        if area < frame_area_min or area > frame_area_max:
            continue
        if circ < circ_min:
            continue

        x, y, w, h = cv.boundingRect(hull)
        M = cv.moments(hull)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        hsv_mean = None
        col_sc = 0.0
        scH = scS = scV = 0.0
        if frame_bgr is not None:
            hsv_mean = mean_hsv_in_hull(frame_bgr, hull)
            if hsv_mean is not None:
                Hm, Sm, Vm = hsv_mean
                col_sc, (scH, scS, scV) = color_score_band(Hm, Sm, Vm)

        total_sc = (W_CIRC * circ) + (W_COLOR * col_sc)

        cand = {
            "score": total_sc,
            "score_breakdown": {"circ": circ, "color": col_sc, "H": scH, "S": scS, "V": scV},
            "circularity": circ,
            "area": int(area),
            "bbox": (x, y, w, h),
            "center": (cx, cy),
            "hull": hull,
            "hsv_mean": hsv_mean,
        }

        if (best is None) or (cand["score"] > best["score"]) or \
           (abs(cand["score"] - best["score"]) < 1e-6 and cand["area"] > best["area"]):
            best = cand
    return best

def delta_to_servo_angles(cmd_deg_x, cmd_deg_y):
    pan  = SERVO_CENTER_PAN  + SIGN_PAN  * cmd_deg_x
    tilt = SERVO_CENTER_TILT + SIGN_TILT * cmd_deg_y
    pan  = max(PAN_MIN,  min(PAN_MAX,  int(round(pan))))
    tilt = max(TILT_MIN, min(TILT_MAX, int(round(tilt))))
    return pan, tilt

# =========================================================
# UART
# =========================================================
class MotorUART:
    def __init__(self, port_pan, port_tilt, baud, timeout):
        self.ser_pan  = serial.Serial(port_pan,  baud, timeout=timeout)
        self.ser_tilt = serial.Serial(port_tilt, baud, timeout=timeout)
        time.sleep(0.2)
        self.last_pan_angle  = None
        self.last_tilt_angle = None

    def send_servo(self, pan_angle, tilt_angle):
        pan_angle  = max(PAN_MIN,  min(PAN_MAX,  int(pan_angle)))
        tilt_angle = max(TILT_MIN, min(TILT_MAX, int(tilt_angle)))
        if pan_angle != self.last_pan_angle:
            self.ser_pan.write(f"{pan_angle}\n".encode('ascii'))
            self.last_pan_angle = pan_angle
        if tilt_angle != self.last_tilt_angle:
            self.ser_tilt.write(f"{tilt_angle}\n".encode('ascii'))
            self.last_tilt_angle = tilt_angle

    def close(self):
        for s in (self.ser_pan, self.ser_tilt):
            try: s.close()
            except: pass

# =========================================================
# UI 유틸
# =========================================================
def draw_text(vis, text, org, color=(0,255,255), scale=0.7):
    cv.putText(vis, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv.LINE_AA)
    cv.putText(vis, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv.LINE_AA)

# =========================================================
# 메인
# =========================================================
def main():
    global KpX, KpY, KvX, KvY, MAX_STEP_DEG

    motors = MotorUART(UART_PAN, UART_TILT, BAUDRATE, SER_TIMEOUT)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    cx0, cy0 = FRAME_W // 2, FRAME_H // 2
    ema_cx, ema_cy = None, None

    frame_area = FRAME_W * FRAME_H
    AREA_MIN = int(AREA_FRAC_MIN * frame_area)
    AREA_MAX = int(AREA_FRAC_MAX * frame_area)

    prev_t = time.time()
    fps = 0.0

    # 속도 추정 상태
    prev_deg_x = None
    prev_deg_y = None
    v_deg_x = 0.0
    v_deg_y = 0.0

    try:
        while True:
            # dt
            now = time.time()
            dt  = max(1e-3, now - prev_t)
            prev_t = now

            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            if MIRROR:
                frame = cv.flip(frame, 1)

            mask = make_mask_from_bgr(frame, use_clahe=True)
            target = pick_best_target(mask, AREA_MIN, AREA_MAX, CIRC_MIN, frame_bgr=frame)

            vis = frame.copy()
            cv.drawMarker(vis, (cx0, cy0), (255, 255, 255), markerType=cv.MARKER_CROSS, markerSize=24, thickness=2)
            cv.circle(vis, (cx0, cy0), 3, (0, 0, 255), -1)
            if MIRROR:
                draw_text(vis, "MIRROR: ON", (FRAME_W - 180, 30), color=(255,255,255), scale=0.7)

            if target is not None:
                (bx, by, bw, bh) = target["bbox"]
                (cx, cy) = target["center"]
                area = target["area"]
                circ = target["circularity"]
                hull = target["hull"]

                # 스무딩(EMA)
                if ema_cx is None:
                    ema_cx, ema_cy = cx, cy
                else:
                    ema_cx = int(SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * ema_cx)
                    ema_cy = int(SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * ema_cy)

                # 픽셀 오차 (데드존 적용 안 함)
                dx_px = ema_cx - cx0
                dy_px = ema_cy - cy0

                # 각도 오차
                deg_x, deg_y = px_to_deg(dx_px, dy_px, FRAME_W, FRAME_H)

                # 각속도 추정 + EMA
                if prev_deg_x is None:
                    prev_deg_x, prev_deg_y = deg_x, deg_y
                    est_vx = est_vy = 0.0
                else:
                    est_vx = (deg_x - prev_deg_x) / dt
                    est_vy = (deg_y - prev_deg_y) / dt
                    prev_deg_x, prev_deg_y = deg_x, deg_y
                v_deg_x = ALPHA_V * v_deg_x + (1 - ALPHA_V) * est_vx
                v_deg_y = ALPHA_V * v_deg_y + (1 - ALPHA_V) * est_vy

                # 각도 → 거리(cm) → 필요한 모터각(연속값)
                dx_cm, dy_cm = angle_to_cm(deg_x, deg_y, RANGE_CM)
                need_x_deg = dx_cm / H_CM_PER_DEG
                need_y_deg = dy_cm / V_CM_PER_DEG

                # 축별 위치+속도 결합
                cmd_x = KpX * need_x_deg + KvX * v_deg_x
                cmd_y = KpY * need_y_deg + KvY * v_deg_y

                # 프레임당 최대각 제한(기본 비활성: 0)
                if MAX_STEP_DEG > 0:
                    cmd_x = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, cmd_x))
                    cmd_y = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, cmd_y))

                # 절대 서보각 산출(정수화만 적용)
                servo_pan, servo_tilt = delta_to_servo_angles(cmd_x, cmd_y)

                motors.send_servo(servo_pan, servo_tilt)

                # 시각화
                cv.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv.circle(vis, (ema_cx, ema_cy), 5, (0, 0, 255), -1)
                cv.drawContours(vis, [hull], -1, (255, 0, 255), 2)

                y0 = 28
                dy = 26
                draw_text(vis, f"area:{area}  circ:{circ:.3f}",
                          (bx, max(0, by - 10)), color=(0,255,0), scale=0.6)
                draw_text(vis, f"offset_px:   ({dx_px:+d}, {dy_px:+d})", (10, y0), color=(50,220,50))
                draw_text(vis, f"offset_deg:  ({deg_x:+.2f}, {deg_y:+.2f})", (10, y0+dy), color=(50,220,50))
                draw_text(vis, f"cmd(raw):    pan={cmd_x:+.2f}°, tilt={cmd_y:+.2f}°",
                          (10, y0+2*dy), color=(0,255,255))
                draw_text(vis, f"servo_out:   pan={servo_pan:3d}°, tilt={servo_tilt:3d}°",
                          (10, y0+3*dy), color=(0,255,255))
                draw_text(vis, f"KpX:{KpX:.2f} KvX:{KvX:.2f} | KpY:{KpY:.2f} KvY:{KvY:.2f} | MAX:{MAX_STEP_DEG}°",
                          (FRAME_W - 650, FRAME_H - 30), color=(200,220,255), scale=0.6)

            else:
                ema_cx = ema_cy = None
                draw_text(vis, "No target", (10, 30), color=(0,0,255), scale=0.8)

            # FPS
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            draw_text(vis, f"FPS: {fps:.1f}", (10, FRAME_H - 10), color=(255,255,0), scale=0.8)

            cv.imshow("frame", vis)
            cv.imshow("mask", mask)

            # 키 입력
            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # 숫자키 튜닝
            if   key == ord('1'):  KpX = max(0.0, round(KpX - 0.05, 3))
            elif key == ord('2'):  KpX = round(KpX + 0.05, 3)
            elif key == ord('3'):  KpY = max(0.0, round(KpY - 0.05, 3))
            elif key == ord('4'):  KpY = round(KpY + 0.05, 3)
            elif key == ord('5'):  KvX = max(0.0, round(KvX - 0.05, 3))
            elif key == ord('6'):  KvX = round(KvX + 0.05, 3)
            elif key == ord('7'):  KvY = max(0.0, round(KvY - 0.05, 3))
            elif key == ord('8'):  KvY = round(KvY + 0.05, 3)
            elif key == ord('9'):  MAX_STEP_DEG = max(0, MAX_STEP_DEG - 1)   # 0이면 무제한
            elif key == ord('0'):  MAX_STEP_DEG = min(60, MAX_STEP_DEG + 1)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv.destroyAllWindows()
        picam2.stop()
        motors.close()

if __name__ == "__main__":
    main()
