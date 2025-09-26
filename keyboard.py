from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time
import math
import serial
import json, os

# ==============================
# 기본 설정
# ==============================
HFOV_DEG = 66.0
VFOV_DEG = 41.0
FRAME_W, FRAME_H = 1280, 720
TARGET_FPS = 30
MIRROR = True

AREA_FRAC_MIN = 0.0003
AREA_FRAC_MAX = 0.30
CIRC_MIN      = 0.80

# 모폴로지
OPEN_K = 5
CLOSE_K = 7
OPEN_ITERS = 1
CLOSE_ITERS = 2

# 추적/모터 기본
SMOOTH_ALPHA = 0.40     # 타깃 중심 EMA (스무딩; 0~1)
MIN_MOVE_DEG = 1.0      # 1도 미만은 구동 안함(헌팅 방지)
DEADZONE_PX  = 8        # 총구 기준 데드존(픽셀)
RANGE_CM     = 200.0
H_CM_PER_DEG = 1.32
V_CM_PER_DEG = 1.32

# ===== 실시간 튜닝 파라미터 =====
Kp = 1.00            # 위치 오차(각도 기반 명령) 비례 이득
Kv = 0.30            # 타깃 각속도(deg/s) 부스트 이득
MAX_STEP_DEG = 8     # 프레임당 최대 구동각 제한(°)
ALPHA_V = 0.6        # 속도 추정 EMA

STEP_K  = 0.05       # Kp/Kv 변경 스텝
STEP_DZ = 1          # DEADZONE_PX 변경 스텝(px)
STEP_MINMOVE = 1.0   # MIN_MOVE_DEG 변경 스텝(°)

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

# ==============================
# 총구 위치(영상 좌표) & 영점 보정
# ==============================
MUZZLE_PX = 640      # 예시: 화면 중앙
MUZZLE_PY = 600      # 예시: 중앙보다 아래(레이저 높이)

# 전기적 영점 초기값(하드웨어 기준)
PAN_OFFSET_INIT  = 119
TILT_OFFSET_INIT = 98

# 런타임 영점(키로 조절)
PAN_OFFSET  = PAN_OFFSET_INIT
TILT_OFFSET = TILT_OFFSET_INIT
ADJ_STEP    = 1      # 한 번 누를 때 오프셋 변화(°)

# 선택: 영점/튜닝 저장 파일
SETTINGS = "/home/pi/zero_offsets.json"  # 경로 조정 가능

# ==============================
# 타깃 스코어(원형+색)
# ==============================
W_CIRC, W_COLOR = 0.40, 0.60
WC_H, WC_S, WC_V = 0.15, 0.65, 0.20

H_LO, H_HI = 3.0, 11.0
S_LO, S_HI = 180.0, 199.0
V_LO, V_HI = 175.0, 199.0
H_SIGMA_OUT, S_SIGMA_OUT, V_SIGMA_OUT = 6.0, 15.0, 15.0

# ==============================
# 유틸
# ==============================
def draw_text(img, text, org, color=(255,255,255), scale=0.7, thick=2):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)

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
            cur = cv.inRange(hsv_img, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            m = cur if m is None else cv.bitwise_or(m, cur)
        return m

    # 조명 견고성 위해 밝음/어두움 두 구간
    red_bright = [((0,130,80),(15,255,255)), ((165,130,80),(180,255,255))]
    red_dark   = [((0, 90,40),(15,255,255)), ((165, 90,40),(180,255,255))]
    m_red = cv.bitwise_or(or_ranges(hsv, red_bright), or_ranges(hsv, red_dark))

    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (OPEN_K, OPEN_K))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    mask = cv.morphologyEx(m_red,  cv.MORPH_OPEN,  k_open,  iterations=OPEN_ITERS)
    mask = cv.morphologyEx(mask,   cv.MORPH_CLOSE, k_close, iterations=CLOSE_ITERS)
    return mask

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
    if not np.any(m):
        return None
    h, s, v = cv.split(hsv)
    return (float(np.mean(h[m])), float(np.mean(s[m])), float(np.mean(v[m])))

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

def pick_best_target(mask, area_min, area_max, circ_min, frame_bgr=None):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        hull = cv.convexHull(c)
        circ, area = circularity_from_contour(hull)
        if area < area_min or area > area_max:  continue
        if circ < circ_min:                      continue

        M = cv.moments(hull)
        if M["m00"] == 0:                        continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        col_sc = 0.0; scH = scS = scV = 0.0; hsv_mean = None
        if frame_bgr is not None:
            hsv_mean = mean_hsv_in_hull(frame_bgr, hull)
            if hsv_mean is not None:
                Hm, Sm, Vm = hsv_mean
                col_sc, (scH, scS, scV) = color_score_band(Hm, Sm, Vm)

        total_sc = (W_CIRC * circ) + (W_COLOR * col_sc)
        cand = {
            "score": total_sc,
            "score_breakdown": {"circ": circ, "color": col_sc, "H": scH, "S": scS, "V": scV},
            "circularity": circ, "area": int(area), "center": (cx, cy), "hull": hull, "hsv_mean": hsv_mean,
            "bbox": cv.boundingRect(hull),
        }
        if (best is None) or (cand["score"] > best["score"]) or \
           (abs(cand["score"] - best["score"]) < 1e-6 and cand["area"] > best["area"]):
            best = cand
    return best

# 픽셀→각도 (총구 픽셀 기준)
def px_to_deg_with_muzzle(target_px, target_py, muzzle_px, muzzle_py, width, height):
    dx_px = target_px - muzzle_px
    dy_px = target_py - muzzle_py
    # 데드존
    if abs(dx_px) < DEADZONE_PX: dx_px = 0
    if abs(dy_px) < DEADZONE_PX: dy_px = 0
    deg_x = (dx_px / float(width))  * HFOV_DEG
    deg_y = (dy_px / float(height)) * VFOV_DEG
    return deg_x, deg_y, dx_px, dy_px

def angle_to_cm(deg_x, deg_y, range_cm):
    dx_cm = range_cm * math.tan(math.radians(deg_x))
    dy_cm = range_cm * math.tan(math.radians(deg_y))
    return dx_cm, dy_cm

def cm_to_motor_deg(dx_cm, dy_cm):
    need_deg_x = dx_cm / H_CM_PER_DEG
    need_deg_y = dy_cm / V_CM_PER_DEG
    def quantize(v):
        s = 1 if v >= 0 else -1
        mag = abs(v)
        if mag < MIN_MOVE_DEG:
            return 0
        return int(round(s * mag))
    return quantize(need_deg_x), quantize(need_deg_y)

def delta_to_servo_angles(cmd_deg_x, cmd_deg_y):
    pan  = SERVO_CENTER_PAN  + SIGN_PAN  * cmd_deg_x
    tilt = SERVO_CENTER_TILT + SIGN_TILT * cmd_deg_y
    pan  = max(PAN_MIN,  min(PAN_MAX,  int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    return pan, tilt

# ==============================
# UART
# ==============================
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
            try:
                s.close()
            except:
                pass

# ==============================
# 설정 저장/로드 (선택)
# ==============================
def save_offsets():
    try:
        with open(SETTINGS, "w") as f:
            json.dump({
                "PAN_OFFSET": PAN_OFFSET,
                "TILT_OFFSET": TILT_OFFSET,
                "ADJ_STEP": ADJ_STEP,
                "Kp": Kp, "Kv": Kv,
                "MAX_STEP_DEG": MAX_STEP_DEG,
                "DEADZONE_PX": DEADZONE_PX,
                "MIN_MOVE_DEG": MIN_MOVE_DEG,
                "SMOOTH_ALPHA": SMOOTH_ALPHA
            }, f)
        print("[SAVE] Settings saved:", SETTINGS)
    except Exception as e:
        print("[SAVE] Failed:", e)

def load_offsets():
    global PAN_OFFSET, TILT_OFFSET, ADJ_STEP
    global Kp, Kv, MAX_STEP_DEG, DEADZONE_PX, MIN_MOVE_DEG, SMOOTH_ALPHA
    try:
        if os.path.exists(SETTINGS):
            d = json.load(open(SETTINGS))
            PAN_OFFSET  = int(d.get("PAN_OFFSET", PAN_OFFSET))
            TILT_OFFSET = int(d.get("TILT_OFFSET", TILT_OFFSET))
            ADJ_STEP    = int(d.get("ADJ_STEP", ADJ_STEP))
            Kp = float(d.get("Kp", Kp))
            Kv = float(d.get("Kv", Kv))
            MAX_STEP_DEG = int(d.get("MAX_STEP_DEG", MAX_STEP_DEG))
            DEADZONE_PX  = int(d.get("DEADZONE_PX", DEADZONE_PX))
            MIN_MOVE_DEG = float(d.get("MIN_MOVE_DEG", MIN_MOVE_DEG))
            SMOOTH_ALPHA = float(d.get("SMOOTH_ALPHA", SMOOTH_ALPHA))
            print("[LOAD] Settings loaded.")
    except Exception as e:
        print("[LOAD] Failed:", e)

# ==============================
# 메인
# ==============================
def main():
    global PAN_OFFSET, TILT_OFFSET
    global Kp, Kv, MAX_STEP_DEG, DEADZONE_PX, MIN_MOVE_DEG, SMOOTH_ALPHA
    global ADJ_STEP

    # 필요하면 부팅 시 저장값 로드
    load_offsets()

    motors = MotorUART(UART_PAN, UART_TILT, BAUDRATE, SER_TIMEOUT)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)})
    picam2.configure(config)
    picam2.start()

    frame_area = FRAME_W * FRAME_H
    AREA_MIN = int(AREA_FRAC_MIN * frame_area)
    AREA_MAX = int(AREA_FRAC_MAX * frame_area)

    ema_cx = ema_cy = None
    prev_t = time.time()
    fps = 0.0

    # 속도 추정 상태
    prev_deg_x = None
    prev_deg_y = None
    v_deg_x = 0.0
    v_deg_y = 0.0

    # 스무딩 프리셋 (숫자 0으로 순환)
    smoothing_presets = [0.20, 0.35, 0.50, 0.65, 0.80]
    smooth_idx = min(range(len(smoothing_presets)),
                     key=lambda i: abs(smoothing_presets[i]-SMOOTH_ALPHA))

    try:
        while True:
            # 프레임 시작 시각
            now = time.time()
            dt = max(1e-3, now - prev_t)  # 제어/속도 추정용 dt
            prev_t = now

            frame_rgb = picam2.capture_array()
            frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
            if MIRROR:
                frame = cv.flip(frame, 1)

            mask = make_mask_from_bgr(frame, use_clahe=True)
            target = pick_best_target(mask, AREA_MIN, AREA_MAX, CIRC_MIN, frame_bgr=frame)

            vis = frame.copy()

            # 중앙 표시 + 총구 표시
            cx0, cy0 = FRAME_W // 2, FRAME_H // 2
            cv.drawMarker(vis, (cx0, cy0), (255, 255, 255), markerType=cv.MARKER_CROSS, markerSize=24, thickness=2)
            cv.circle(vis, (cx0, cy0), 3, (0, 0, 255), -1)
            cv.drawMarker(vis, (MUZZLE_PX, MUZZLE_PY), (0,255,255), markerType=cv.MARKER_TILTED_CROSS, markerSize=24, thickness=2)
            draw_text(vis, "MIRROR: ON" if MIRROR else "MIRROR: OFF", (FRAME_W - 180, 30))

            if target is not None:
                (bx, by, bw, bh) = target["bbox"]
                (cx, cy) = target["center"]
                area = target["area"]
                circ = target["circularity"]
                hull = target["hull"]

                # EMA로 타깃 중심 매끈화 (스무딩)
                if ema_cx is None:
                    ema_cx, ema_cy = cx, cy
                else:
                    ema_cx = int(SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * ema_cx)
                    ema_cy = int(SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * ema_cy)

                # 총구 기준 각도/픽셀 오차
                deg_x, deg_y, dx_px, dy_px = px_to_deg_with_muzzle(ema_cx, ema_cy, MUZZLE_PX, MUZZLE_PY, FRAME_W, FRAME_H)

                # --- 타깃 각속도 추정 (deg/s) ---
                if prev_deg_x is None:
                    prev_deg_x, prev_deg_y = deg_x, deg_y
                    est_vx = est_vy = 0.0
                else:
                    ddeg_x = deg_x - prev_deg_x
                    ddeg_y = deg_y - prev_deg_y
                    est_vx = ddeg_x / dt
                    est_vy = ddeg_y / dt
                    prev_deg_x, prev_deg_y = deg_x, deg_y

                # 속도 EMA
                v_deg_x = ALPHA_V * v_deg_x + (1 - ALPHA_V) * est_vx
                v_deg_y = ALPHA_V * v_deg_y + (1 - ALPHA_V) * est_vy

                # 거리 환산(정보표시용)
                dx_cm, dy_cm = angle_to_cm(deg_x, deg_y, RANGE_CM)

                # 위치기반 필요각(정수화 경로와 척도 맞춤)
                def pos_cmd_from_cm(dx_cm, dy_cm):
                    need_x = dx_cm / H_CM_PER_DEG
                    need_y = dy_cm / V_CM_PER_DEG
                    return need_x, need_y
                need_x, need_y = pos_cmd_from_cm(dx_cm, dy_cm)

                # --- 위치 + 속도 피드포워드 결합 ---
                cmd_x = Kp * need_x + Kv * v_deg_x
                cmd_y = Kp * need_y + Kv * v_deg_y

                # 프레임당 최대 구동각 제한
                cmd_x = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, cmd_x))
                cmd_y = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, cmd_y))

                # 최소 구동각 적용(정수화)
                def quantize_runtime(v):
                    s = 1 if v >= 0 else -1
                    mag = abs(v)
                    if mag < MIN_MOVE_DEG:
                        return 0
                    return int(round(s * mag))

                cmd_deg_x = quantize_runtime(cmd_x)
                cmd_deg_y = quantize_runtime(cmd_y)

                servo_pan, servo_tilt = delta_to_servo_angles(cmd_deg_x, cmd_deg_y)

                # --- 서보 기준 정조준 오프셋 적용 ---
                servo_pan  = PAN_OFFSET  + (servo_pan  - SERVO_CENTER_PAN)
                servo_tilt = TILT_OFFSET + (servo_tilt - SERVO_CENTER_TILT)
                servo_pan  = max(PAN_MIN,  min(PAN_MAX,  int(servo_pan)))
                servo_tilt = max(TILT_MIN, min(TILT_MAX, int(servo_tilt)))

                motors.send_servo(servo_pan, servo_tilt)

                # 시각화
                cv.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                cv.circle(vis, (ema_cx, ema_cy), 5, (0, 0, 255), -1)
                cv.drawContours(vis, [hull], -1, (255, 0, 255), 2)

                y0, dyy = 28, 26
                draw_text(vis, f"area:{area}  circ:{circ:.3f}  score:{target['score']:.3f}",
                          (bx, max(0, by - 10)), color=(0,255,0), scale=0.6)
                draw_text(vis, f"offset_px(mu): ({dx_px:+d}, {dy_px:+d})", (10, y0), color=(50,220,50))
                draw_text(vis, f"offset_deg:     ({deg_x:+.2f}, {deg_y:+.2f})", (10, y0+dyy), color=(50,220,50))
                draw_text(vis, f"offset_cm*:     ({dx_cm:+.1f}, {dy_cm:+.1f})  @R={RANGE_CM:.0f}cm",
                          (10, y0+2*dyy), color=(50,220,50))
                draw_text(vis, f"motor_cmd:      panΔ={cmd_deg_x:+d}°, tiltΔ={cmd_deg_y:+d}°",
                          (10, y0+3*dyy), color=(0,255,255))
                draw_text(vis, f"servo_out:      pan={servo_pan:3d}°, tilt={servo_tilt:3d}°",
                          (10, y0+4*dyy), color=(0,255,255))

            else:
                ema_cx = ema_cy = None
                draw_text(vis, "No target", (10, 30), color=(0,0,255), scale=0.8)

            # FPS & HUD
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
            draw_text(vis, f"FPS: {fps:.1f}", (10, FRAME_H - 10), color=(255,255,0), scale=0.8)

            draw_text(vis, f"ZERO pan:{PAN_OFFSET:+d}° tilt:{TILT_OFFSET:+d}° (step={ADJ_STEP}°)",
                      (FRAME_W - 500, FRAME_H - 32), color=(200,200,255), scale=0.6)
            draw_text(vis, f"Kp:{Kp:.2f} Kv:{Kv:.2f} MAX:{MAX_STEP_DEG}°  DZ:{DEADZONE_PX}px  MIN:{MIN_MOVE_DEG:.1f}°  α:{SMOOTH_ALPHA:.2f}",
                      (FRAME_W - 690, FRAME_H - 60), color=(200,220,255), scale=0.6)

            cv.imshow("frame", vis)
            cv.imshow("mask",  mask)

            key = cv.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            # ---- 키 입력: 영점 미세조정 ----
            if   key == ord('u'):
                TILT_OFFSET -= ADJ_STEP; print(f"[OFFSET] TILT_OFFSET -> {TILT_OFFSET}")
            elif key == ord('d'):
                TILT_OFFSET += ADJ_STEP; print(f"[OFFSET] TILT_OFFSET -> {TILT_OFFSET}")
            elif key == ord('r'):
                PAN_OFFSET  += ADJ_STEP;  print(f"[OFFSET] PAN_OFFSET  -> {PAN_OFFSET}")
            elif key == ord('l'):
                PAN_OFFSET  -= ADJ_STEP;  print(f"[OFFSET] PAN_OFFSET  -> {PAN_OFFSET}")
            elif key == ord('c'):
                PAN_OFFSET, TILT_OFFSET = PAN_OFFSET_INIT, TILT_OFFSET_INIT
                print("[OFFSET] reset to initial:", PAN_OFFSET, TILT_OFFSET)

            # ---- 키 입력: 숫자만으로 튜닝 ----
            elif key == ord('1'):   # Kp down
                Kp = max(0.0, round(Kp - STEP_K, 3)); print(f"[TUNE] Kp = {Kp:.3f}")
            elif key == ord('2'):   # Kp up
                Kp = round(Kp + STEP_K, 3); print(f"[TUNE] Kp = {Kp:.3f}")

            elif key == ord('3'):   # Kv down
                Kv = max(0.0, round(Kv - STEP_K, 3)); print(f"[TUNE] Kv = {Kv:.3f}")
            elif key == ord('4'):   # Kv up
                Kv = round(Kv + STEP_K, 3); print(f"[TUNE] Kv = {Kv:.3f}")

            elif key == ord('5'):   # MAX_STEP_DEG down
                MAX_STEP_DEG = max(1, MAX_STEP_DEG - 1); print(f"[TUNE] MAX_STEP_DEG = {MAX_STEP_DEG}")
            elif key == ord('6'):   # MAX_STEP_DEG up
                MAX_STEP_DEG = min(30, MAX_STEP_DEG + 1); print(f"[TUNE] MAX_STEP_DEG = {MAX_STEP_DEG}")

            elif key == ord('7'):   # DEADZONE_PX down
                DEADZONE_PX = max(0, DEADZONE_PX - STEP_DZ); print(f"[TUNE] DEADZONE_PX = {DEADZONE_PX}")
            elif key == ord('8'):   # DEADZONE_PX up
                DEADZONE_PX = DEADZONE_PX + STEP_DZ; print(f"[TUNE] DEADZONE_PX = {DEADZONE_PX}")

            elif key == ord('9'):   # MIN_MOVE_DEG down
                MIN_MOVE_DEG = max(0.0, round(MIN_MOVE_DEG - STEP_MINMOVE, 2)); print(f"[TUNE] MIN_MOVE_DEG = {MIN_MOVE_DEG:.2f}")
            elif key == ord('0'):   # 스무딩 프리셋 순환
                smooth_idx = (smooth_idx + 1) % len(smoothing_presets)
                SMOOTH_ALPHA = smoothing_presets[smooth_idx]
                print(f"[TUNE] SMOOTH_ALPHA = {SMOOTH_ALPHA:.2f}")

            # ---- 오프셋/스텝 저장/로드 (선택) ----
            elif key == ord('['):   # ADJ_STEP down
                ADJ_STEP = max(1, ADJ_STEP - 1); print(f"[STEP] ADJ_STEP = {ADJ_STEP}")
            elif key == ord(']'):   # ADJ_STEP up
                ADJ_STEP = ADJ_STEP + 1; print(f"[STEP] ADJ_STEP = {ADJ_STEP}")
            elif key == ord('s'):
                save_offsets()
            elif key == ord('o'):
                load_offsets()

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv.destroyAllWindows()
        picam2.stop()
        motors.close()

if __name__ == "__main__":
    main()
