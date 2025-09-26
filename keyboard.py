from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time, math, serial, json, os

# ==============================
# 기본 설정
# ==============================
HFOV_DEG = 66.0
VFOV_DEG = 41.0
FRAME_W, FRAME_H = 1280, 720
MIRROR = True

AREA_FRAC_MIN = 0.0003
AREA_FRAC_MAX = 0.30
CIRC_MIN      = 0.80

# 모폴로지
OPEN_K = 5; CLOSE_K = 7
OPEN_ITERS = 1; CLOSE_ITERS = 2

# 추적/모터 기본
SMOOTH_ALPHA = 0.40       # 타깃 중심 EMA(스무딩, 0~1)
MIN_MOVE_DEG = 1.0        # 정수화 최소 구동각
DEADZONE_PX  = 8          # 총구 기준 데드존
RANGE_CM     = 200.0
H_CM_PER_DEG = 1.32; V_CM_PER_DEG = 1.32

# ===== 실시간 튜닝 (축별) =====
KpX = 1.00; KpY = 1.00    # 위치 오차 비례 이득
KvX = 0.30; KvY = 0.30    # 각속도 이득
MAX_STEP_DEG = 8          # 프레임당 최대 구동각(공통)
ALPHA_V = 0.6             # 속도 EMA

STEP_K  = 0.05            # Kp/Kv 스텝
STEP_MINMOVE = 1.0        # MIN_MOVE_DEG 스텝(원하면 숫자키 추가 매핑 가능)
STEP_DZ = 1               # DEADZONE_PX 스텝(원하면 숫자키 추가 매핑 가능)

# UART
UART_PAN='/dev/ttyAMA2'; UART_TILT='/dev/ttyAMA3'
BAUDRATE=115200; SER_TIMEOUT=0.1

# 서보
SERVO_CENTER_PAN=90; SERVO_CENTER_TILT=90
SIGN_PAN=+1; SIGN_TILT=-1
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 75, 180   # 하드웨어 여유 있으면 60까지 낮춰도 됨

# 총구 좌표 & 영점
MUZZLE_PX = 640; MUZZLE_PY = 600
PAN_OFFSET_INIT=119; TILT_OFFSET_INIT=98
PAN_OFFSET=PAN_OFFSET_INIT; TILT_OFFSET=TILT_OFFSET_INIT
ADJ_STEP=1
SETTINGS="/home/pi/zero_offsets.json"

# 색/타깃 점수
W_CIRC, W_COLOR = 0.40, 0.60
WC_H, WC_S, WC_V = 0.15, 0.65, 0.20
H_LO,H_HI=3.0,11.0; S_LO,S_HI=180.0,199.0; V_LO,V_HI=175.0,199.0
H_SIGMA_OUT,S_SIGMA_OUT,V_SIGMA_OUT = 6.0,15.0,15.0

# ==============================
def draw_text(img, text, org, color=(255,255,255), scale=0.7, thick=2):
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)

def make_mask_from_bgr(frame_bgr, use_clahe=True):
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    if use_clahe:
        h,s,v = cv.split(hsv)
        v = cv.createCLAHE(2.0,(8,8)).apply(v)
        hsv = cv.merge([h,s,v])
    def or_ranges(hsv_img, ranges):
        m=None
        for lo,hi in ranges:
            cur=cv.inRange(hsv_img, np.array(lo,np.uint8), np.array(hi,np.uint8))
            m=cur if m is None else cv.bitwise_or(m,cur)
        return m
    red_bright=[((0,130,80),(15,255,255)),((165,130,80),(180,255,255))]
    red_dark  =[((0, 90,40),(15,255,255)),((165, 90,40),(180,255,255))]
    m_red=cv.bitwise_or(or_ranges(hsv,red_bright), or_ranges(hsv,red_dark))
    k_open=cv.getStructuringElement(cv.MORPH_ELLIPSE,(OPEN_K,OPEN_K))
    k_close=cv.getStructuringElement(cv.MORPH_ELLIPSE,(CLOSE_K,CLOSE_K))
    mask=cv.morphologyEx(m_red,cv.MORPH_OPEN,k_open,iterations=OPEN_ITERS)
    mask=cv.morphologyEx(mask,cv.MORPH_CLOSE,k_close,iterations=CLOSE_ITERS)
    return mask

def circularity_from_contour(cnt):
    area=cv.contourArea(cnt); perim=cv.arcLength(cnt,True)
    if perim<=0: return 0.0,area
    return 4.0*math.pi*area/(perim*perim), area

def mean_hsv_in_hull(frame_bgr, hull):
    hsv=cv.cvtColor(frame_bgr,cv.COLOR_BGR2HSV)
    mask=np.zeros(frame_bgr.shape[:2],np.uint8)
    cv.fillConvexPoly(mask,hull.reshape(-1,2),255)
    m=mask>0
    if not np.any(m): return None
    h,s,v=cv.split(hsv)
    return (float(np.mean(h[m])), float(np.mean(s[m])), float(np.mean(v[m])))

def band_score(val, lo, hi, sigma):
    if lo<=val<=hi: return 1.0
    d=(lo-val) if val<lo else (val-hi)
    return math.exp(-0.5*(d/max(1e-6,sigma))**2)

def color_score_band(Hm,Sm,Vm):
    scH=band_score(Hm,H_LO,H_HI,H_SIGMA_OUT)
    scS=band_score(Sm,S_LO,S_HI,S_SIGMA_OUT)
    scV=band_score(Vm,V_LO,V_HI,V_SIGMA_OUT)
    num=WC_H*scH+WC_S*scS+WC_V*scV; den=WC_H+WC_S+WC_V
    return max(0,min(1,num/den)), (scH,scS,scV)

def pick_best_target(mask, area_min, area_max, circ_min, frame_bgr=None):
    cnts,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    best=None
    for c in cnts:
        hull=cv.convexHull(c)
        circ,area=circularity_from_contour(hull)
        if not (area_min<=area<=area_max): continue
        if circ<circ_min: continue
        M=cv.moments(hull)
        if M["m00"]==0: continue
        cx=int(M["m10"]/M["m00"]); cy=int(M["m01"]/M["m00"])
        col_sc=0.0; scH=scS=scV=0.0; hsv_mean=None
        if frame_bgr is not None:
            hsv_mean=mean_hsv_in_hull(frame_bgr,hull)
            if hsv_mean is not None:
                Hm,Sm,Vm=hsv_mean
                col_sc,(scH,scS,scV)=color_score_band(Hm,Sm,Vm)
        total=(W_CIRC*circ)+(W_COLOR*col_sc)
        cand={"score":total,"score_breakdown":{"circ":circ,"color":col_sc,"H":scH,"S":scS,"V":scV},
              "circularity":circ,"area":int(area),"center":(cx,cy),"hull":hull,"hsv_mean":hsv_mean,
              "bbox":cv.boundingRect(hull)}
        if (best is None) or (cand["score"]>best["score"]) or \
           (abs(cand["score"]-best["score"])<1e-6 and cand["area"]>best["area"]):
            best=cand
    return best

def px_to_deg_with_muzzle(tx,ty, mx,my, w,h):
    dx=tx-mx; dy=ty-my
    if abs(dx)<DEADZONE_PX: dx=0
    if abs(dy)<DEADZONE_PX: dy=0
    return (dx/w)*HFOV_DEG, (dy/h)*VFOV_DEG, dx, dy

def angle_to_cm(dgx,dgy, R):
    return R*math.tan(math.radians(dgx)), R*math.tan(math.radians(dgy))

def delta_to_servo_angles(cmd_x_deg, cmd_y_deg):
    pan  = SERVO_CENTER_PAN  + SIGN_PAN  * cmd_x_deg
    tilt = SERVO_CENTER_TILT + SIGN_TILT * cmd_y_deg
    pan  = max(PAN_MIN,  min(PAN_MAX,  int(pan)))
    tilt = max(TILT_MIN, min(TILT_MAX, int(tilt)))
    return pan, tilt

class MotorUART:
    def __init__(self, p_pan, p_tilt, baud, timeout):
        self.ser_pan=serial.Serial(p_pan,baud,timeout=timeout)
        self.ser_tilt=serial.Serial(p_tilt,baud,timeout=timeout)
        time.sleep(0.2); self.last_pan=None; self.last_tilt=None
    def send_servo(self, pan, tilt):
        pan=max(PAN_MIN,min(PAN_MAX,int(pan)))
        tilt=max(TILT_MIN,min(TILT_MAX,int(tilt)))
        if pan!=self.last_pan:
            self.ser_pan.write(f"{pan}\n".encode('ascii')); self.last_pan=pan
        if tilt!=self.last_tilt:
            self.ser_tilt.write(f"{tilt}\n".encode('ascii')); self.last_tilt=tilt
    def close(self):
        for s in (self.ser_pan,self.ser_tilt):
            try: s.close()
            except: pass

def save_offsets():
    try:
        with open(SETTINGS,"w") as f:
            json.dump({
                "PAN_OFFSET":PAN_OFFSET,"TILT_OFFSET":TILT_OFFSET,"ADJ_STEP":ADJ_STEP,
                "KpX":KpX,"KpY":KpY,"KvX":KvX,"KvY":KvY,
                "MAX_STEP_DEG":MAX_STEP_DEG,"DEADZONE_PX":DEADZONE_PX,
                "MIN_MOVE_DEG":MIN_MOVE_DEG,"SMOOTH_ALPHA":SMOOTH_ALPHA
            },f)
        print("[SAVE] OK:",SETTINGS)
    except Exception as e:
        print("[SAVE] FAIL:",e)

def load_offsets():
    global PAN_OFFSET,TILT_OFFSET,ADJ_STEP
    global KpX,KpY,KvX,KvY,MAX_STEP_DEG,DEADZONE_PX,MIN_MOVE_DEG,SMOOTH_ALPHA
    try:
        if os.path.exists(SETTINGS):
            d=json.load(open(SETTINGS))
            PAN_OFFSET=int(d.get("PAN_OFFSET",PAN_OFFSET))
            TILT_OFFSET=int(d.get("TILT_OFFSET",TILT_OFFSET))
            ADJ_STEP=int(d.get("ADJ_STEP",ADJ_STEP))
            KpX=float(d.get("KpX",KpX)); KpY=float(d.get("KpY",KpY))
            KvX=float(d.get("KvX",KvX)); KvY=float(d.get("KvY",KvY))
            MAX_STEP_DEG=int(d.get("MAX_STEP_DEG",MAX_STEP_DEG))
            DEADZONE_PX=int(d.get("DEADZONE_PX",DEADZONE_PX))
            MIN_MOVE_DEG=float(d.get("MIN_MOVE_DEG",MIN_MOVE_DEG))
            SMOOTH_ALPHA=float(d.get("SMOOTH_ALPHA",SMOOTH_ALPHA))
            print("[LOAD] OK")
    except Exception as e:
        print("[LOAD] FAIL:",e)

def main():
    global PAN_OFFSET,TILT_OFFSET
    global KpX,KpY,KvX,KvY,MAX_STEP_DEG,DEADZONE_PX,MIN_MOVE_DEG,SMOOTH_ALPHA
    global ADJ_STEP

    load_offsets()
    motors=MotorUART(UART_PAN,UART_TILT,BAUDRATE,SER_TIMEOUT)

    cam=Picamera2()
    cfg=cam.create_preview_configuration(main={"size":(FRAME_W,FRAME_H)})
    cam.configure(cfg); cam.start()

    area_min=int(AREA_FRAC_MIN*FRAME_W*FRAME_H)
    area_max=int(AREA_FRAC_MAX*FRAME_W*FRAME_H)

    ema_cx=ema_cy=None
    prev_t=time.time(); fps=0.0
    prev_deg_x=prev_deg_y=None
    v_x=v_y=0.0

    try:
        while True:
            now=time.time(); dt=max(1e-3, now-prev_t); prev_t=now

            frame_rgb=cam.capture_array()
            frame=cv.cvtColor(frame_rgb,cv.COLOR_RGB2BGR)
            if MIRROR: frame=cv.flip(frame,1)

            mask=make_mask_from_bgr(frame, True)
            tgt=pick_best_target(mask, area_min, area_max, CIRC_MIN, frame_bgr=frame)

            vis=frame.copy()
            cx0,cy0=FRAME_W//2, FRAME_H//2
            cv.drawMarker(vis,(cx0,cy0),(255,255,255),cv.MARKER_CROSS,24,2)
            cv.circle(vis,(cx0,cy0),3,(0,0,255),-1)
            cv.drawMarker(vis,(MUZZLE_PX,MUZZLE_PY),(0,255,255),cv.MARKER_TILTED_CROSS,24,2)
            draw_text(vis, "MIRROR: ON" if MIRROR else "MIRROR: OFF", (FRAME_W-180,30))

            clamped_pan=False; clamped_tilt=False

            if tgt is not None:
                (bx,by,bw,bh)=tgt["bbox"]; (cx,cy)=tgt["center"]
                area=tgt["area"]; circ=tgt["circularity"]; hull=tgt["hull"]

                if ema_cx is None: ema_cx,ema_cy=cx,cy
                else:
                    ema_cx=int(SMOOTH_ALPHA*cx+(1-SMOOTH_ALPHA)*ema_cx)
                    ema_cy=int(SMOOTH_ALPHA*cy+(1-SMOOTH_ALPHA)*ema_cy)

                deg_x,deg_y,dx,dy=px_to_deg_with_muzzle(ema_cx,ema_cy,MUZZLE_PX,MUZZLE_PY,FRAME_W,FRAME_H)

                # 속도 추정
                if prev_deg_x is None:
                    prev_deg_x,prev_deg_y=deg_x,deg_y; est_vx=est_vy=0.0
                else:
                    est_vx=(deg_x-prev_deg_x)/dt; est_vy=(deg_y-prev_deg_y)/dt
                    prev_deg_x,prev_deg_y=deg_x,deg_y
                v_x = ALPHA_V*v_x + (1-ALPHA_V)*est_vx
                v_y = ALPHA_V*v_y + (1-ALPHA_V)*est_vy

                # 위치 필요각(연속값)
                need_x = (RANGE_CM*math.tan(math.radians(deg_x))) / H_CM_PER_DEG
                need_y = (RANGE_CM*math.tan(math.radians(deg_y))) / V_CM_PER_DEG

                # 축별 제어: 위치 + 속도
                cmd_x = KpX*need_x + KvX*v_x
                cmd_y = KpY*need_y + KvY*v_y

                # 프레임당 최대 구동각 제한
                cmd_x = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, cmd_x))
                cmd_y = max(-MAX_STEP_DEG, min(MAX_STEP_DEG, cmd_y))

                # 최소 구동각 & 정수화
                def quant(v):
                    s=1 if v>=0 else -1; m=abs(v)
                    if m<MIN_MOVE_DEG: return 0
                    return int(round(s*m))
                cmd_dx=quant(cmd_x); cmd_dy=quant(cmd_y)

                pan,tilt=delta_to_servo_angles(cmd_dx, cmd_dy)

                # 영점 적용
                pan  = PAN_OFFSET  + (pan  - SERVO_CENTER_PAN)
                tilt = TILT_OFFSET + (tilt - SERVO_CENTER_TILT)

                # 클램프 체크
                pan_before=pan; tilt_before=tilt
                pan = max(PAN_MIN,min(PAN_MAX,int(pan)))
                tilt= max(TILT_MIN,min(TILT_MAX,int(tilt)))
                clamped_pan = (pan!=pan_before)
                clamped_tilt= (tilt!=tilt_before)

                motors.send_servo(pan,tilt)

                # 시각화
                cv.rectangle(vis,(bx,by),(bx+bw,by+bh),(0,255,0),2)
                cv.circle(vis,(ema_cx,ema_cy),5,(0,0,255),-1)
                cv.drawContours(vis,[hull],-1,(255,0,255),2)

                y0,dyy=28,26
                draw_text(vis,f"area:{area}  circ:{circ:.3f}  score:{tgt['score']:.3f}",
                          (bx,max(0,by-10)),(0,255,0),0.6)
                draw_text(vis,f"offset_px(mu): ({dx:+d},{dy:+d})",(10,y0),(50,220,50))
                draw_text(vis,f"offset_deg: ({deg_x:+.2f},{deg_y:+.2f})",(10,y0+dyy),(50,220,50))
                draw_text(vis,f"cmdΔ: pan={cmd_dx:+d}°, tilt={cmd_dy:+d}°",(10,y0+2*dyy),(0,255,255))
                draw_text(vis,f"servo: pan={pan:3d}°, tilt={tilt:3d}°",(10,y0+3*dyy),(0,255,255))

            else:
                ema_cx=ema_cy=None
                draw_text(vis,"No target",(10,30),(0,0,255),0.8)

            # FPS & HUD
            fps = 0.9*fps + 0.1*(1.0/dt) if fps>0 else (1.0/dt)
            draw_text(vis, f"FPS: {fps:.1f}", (10, FRAME_H-10), (255,255,0), 0.8)
            draw_text(vis, f"ZERO pan:{PAN_OFFSET:+d}° tilt:{TILT_OFFSET:+d}° (step={ADJ_STEP}°)",
                      (FRAME_W-500, FRAME_H-32), (200,200,255), 0.6)
            draw_text(vis, f"KpX:{KpX:.2f} KvX:{KvX:.2f} | KpY:{KpY:.2f} KvY:{KvY:.2f} | MAX:{MAX_STEP_DEG}°  DZ:{DEADZONE_PX}px  MIN:{MIN_MOVE_DEG:.1f}°  α:{SMOOTH_ALPHA:.2f}",
                      (FRAME_W-780, FRAME_H-60), (200,220,255), 0.6)

            if clamped_pan:  draw_text(vis, "CLAMP PAN",  (FRAME_W-150, 70), (0,255,255), 0.7)
            if clamped_tilt: draw_text(vis, "CLAMP TILT", (FRAME_W-150, 95), (0,255,255), 0.7)

            cv.imshow("frame",vis); cv.imshow("mask",mask)

            key=cv.waitKey(1)&0xFF
            if key in (27, ord('q')): break

            # 영점
            if   key==ord('u'): TILT_OFFSET-=ADJ_STEP; print("[OFFSET] TILT",TILT_OFFSET)
            elif key==ord('d'): TILT_OFFSET+=ADJ_STEP; print("[OFFSET] TILT",TILT_OFFSET)
            elif key==ord('r'): PAN_OFFSET +=ADJ_STEP; print("[OFFSET] PAN",PAN_OFFSET)
            elif key==ord('l'): PAN_OFFSET -=ADJ_STEP; print("[OFFSET] PAN",PAN_OFFSET)
            elif key==ord('c'): PAN_OFFSET,TILT_OFFSET=PAN_OFFSET_INIT,TILT_OFFSET_INIT; print("[OFFSET] reset")

            # 숫자키 튜닝(축별)
            elif key==ord('1'): KpX=max(0.0,round(KpX-STEP_K,3)); print("[KpX]",KpX)
            elif key==ord('2'): KpX=round(KpX+STEP_K,3);         print("[KpX]",KpX)
            elif key==ord('3'): KpY=max(0.0,round(KpY-STEP_K,3)); print("[KpY]",KpY)
            elif key==ord('4'): KpY=round(KpY+STEP_K,3);         print("[KpY]",KpY)
            elif key==ord('5'): KvX=max(0.0,round(KvX-STEP_K,3)); print("[KvX]",KvX)
            elif key==ord('6'): KvX=round(KvX+STEP_K,3);         print("[KvX]",KvX)
            elif key==ord('7'): KvY=max(0.0,round(KvY-STEP_K,3)); print("[KvY]",KvY)
            elif key==ord('8'): KvY=round(KvY+STEP_K,3);         print("[KvY]",KvY)
            elif key==ord('9'): MAX_STEP_DEG=max(1,MAX_STEP_DEG-1); print("[MAX]",MAX_STEP_DEG)
            elif key==ord('0'): MAX_STEP_DEG=min(30,MAX_STEP_DEG+1); print("[MAX]",MAX_STEP_DEG)

            # 저장/로드 & 스텝
            elif key==ord('['): ADJ_STEP=max(1,ADJ_STEP-1); print("[STEP]",ADJ_STEP)
            elif key==ord(']'): ADJ_STEP+=1; print("[STEP]",ADJ_STEP)
            elif key==ord('s'): save_offsets()
            elif key==ord('o'): load_offsets()

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        cv.destroyAllWindows(); cam.stop(); motors.close()

if __name__=="__main__":
    main()
