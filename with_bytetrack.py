import os, time
import numpy as np
import jetson_utils_python as jetson_utils
import jetson_inference
import vlc

# =========================
# Box 변환 유틸
# =========================
def tlbr_to_xyah(tlbr):
    # tlbr: [x1,y1,x2,y2] -> [cx,cy,a,h] (a = w/h)
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]
    cx = tlbr[0] + w * 0.5
    cy = tlbr[1] + h * 0.5
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)

def xyah_to_tlbr(xyah):
    cx, cy, a, h = xyah
    w = a * h
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# =========================
# IoU & Mahalanobis
# =========================
def iou_xyxy(a, b):
    N, M = a.shape[0], b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)

def mahalanobis_squared(y, S_inv):
    # y: (K,4), S_inv: (4,4)
    # returns (K,)
    return np.einsum("ki,ij,kj->k", y, S_inv, y)

# =========================
# 칼만 필터 (SORT/ByteTrack 스타일)
# 상태: [cx, cy, a, h, vx, vy, va, vh]  (8D)
# 관측: [cx, cy, a, h]                  (4D)
# =========================
class KalmanFilterXYAH:
    def __init__(self, dt=1.0):
        self.dt = dt
        # 상태 전이 F
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i+4] = dt
        # 관측 H
        self._H = np.zeros((4, 8), dtype=np.float32)
        self._H[0,0] = self._H[1,1] = self._H[2,2] = self._H[3,3] = 1.0

        # 프로세스 잡음 Q (가벼운 움직임 기준)
        q_pos = 1.0
        q_vel = 10.0
        self._Q = np.diag([q_pos, q_pos, 1e-2, q_pos, q_vel, q_vel, 1e-3, q_vel]).astype(np.float32)

        # 관측 잡음 R
        self._R = np.diag([1.0, 1.0, 1e-2, 1.0]).astype(np.float32)

        self._I8 = np.eye(8, dtype=np.float32)

    def initiate(self, xyah):
        mean = np.zeros((8,), dtype=np.float32)
        mean[:4] = xyah
        P = np.diag([10, 10, 1e-1, 10, 100, 100, 1e-2, 100]).astype(np.float32)
        return mean, P

    def predict(self, mean, P):
        mean = self._F @ mean
        P = self._F @ P @ self._F.T + self._Q
        return mean, P

    def project(self, mean, P):
        S = self._H @ P @ self._H.T + self._R
        z = self._H @ mean
        return z, S

    def update(self, mean, P, z_obs):
        z_pred, S = self.project(mean, P)
        K = P @ self._H.T @ np.linalg.inv(S)
        mean = mean + K @ (z_obs - z_pred)
        P = (self._I8 - K @ self._H) @ P
        return mean, P

# =========================
# 트랙 / 트래커 (ByteTrack-KF 경량)
# =========================
class TrackKF:
    __slots__ = ("mean","cov","score","id","age","miss","hit","class_id","activated","kf")
    def __init__(self, xyah, score, class_id, tid, kf: KalmanFilterXYAH):
        self.kf = kf
        self.mean, self.cov = kf.initiate(xyah)
        self.score = float(score)
        self.class_id = int(class_id)
        self.id = tid
        self.age = 0
        self.miss = 0
        self.hit = 1
        self.activated = True

    @property
    def tlbr(self):
        return xyah_to_tlbr(self.mean[:4])

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.miss += 1

    def update(self, tlbr, score):
        xyah = tlbr_to_xyah(tlbr)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyah)
        self.score = float(score)
        self.hit += 1
        self.miss = 0
        self.activated = True

class BYTETrackerKF:
    def __init__(self, track_thresh=0.5, match_thresh=0.7, match_thresh_low=0.5,
                 buffer_ttl=30, min_box_area=10, gate_maha=True, maha_th=25.0,  # ~5^2
                 dt=1.0, max_ids=1<<30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.match_thresh_low = match_thresh_low
        self.buffer_ttl = buffer_ttl
        self.min_box_area = min_box_area
        self.gate_maha = gate_maha
        self.maha_th = maha_th
        self.kf = KalmanFilterXYAH(dt=dt)
        self.tracks = []
        self._next_id = 1
        self._max_ids = max_ids

    def _new_id(self):
        tid = self._next_id
        self._next_id += 1
        if self._next_id >= self._max_ids:
            self._next_id = 1
        return tid

    @staticmethod
    def _greedy_match(iou_mat, iou_th):
        matches, u_a, u_b = [], list(range(iou_mat.shape[0])), list(range(iou_mat.shape[1]))
        if iou_mat.size == 0:
            return matches, u_a, u_b
        iou_copy = iou_mat.copy()
        while True:
            maxv = iou_copy.max() if iou_copy.size else 0.0
            if maxv < iou_th or maxv <= 0:
                break
            i, j = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
            matches.append((int(i), int(j)))
            iou_copy[i, :] = -1.0
            iou_copy[:, j] = -1.0
        matched_a = {m[0] for m in matches}
        matched_b = {m[1] for m in matches}
        u_a = [i for i in range(iou_mat.shape[0]) if i not in matched_a]
        u_b = [j for j in range(iou_mat.shape[1]) if j not in matched_b]
        return matches, u_a, u_b

    def _gating_mask(self, dets_xyah, tracks):
        # Mahalanobis 거리로 게이팅 (옵션)
        if not self.gate_maha or len(tracks) == 0 or len(dets_xyah) == 0:
            return np.ones((len(tracks), len(dets_xyah)), dtype=bool)
        mask = np.zeros((len(tracks), len(dets_xyah)), dtype=bool)
        for i, t in enumerate(tracks):
            z_pred, S = self.kf.project(t.mean, t.cov)
            S_inv = np.linalg.inv(S)
            y = dets_xyah - z_pred[None, :]   # (K,4)
            d2 = mahalanobis_squared(y, S_inv) # (K,)
            mask[i] = d2 < self.maha_th
        return mask

    def update(self, dets_tlbr_scores, class_ids=None):
        # 예측(칼만)
        for t in self.tracks:
            t.predict()
        # 디텍션 정리
        if dets_tlbr_scores is None or len(dets_tlbr_scores) == 0:
            self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]
            return [t for t in self.tracks if t.miss == 0 and t.activated]

        dets = np.asarray(dets_tlbr_scores, dtype=np.float32)
        if class_ids is None:
            cls_arr = np.zeros((dets.shape[0],), dtype=np.int32)
        else:
            cls_arr = np.asarray(class_ids, dtype=np.int32)

        high_mask = dets[:, 4] >= self.track_thresh
        dets_high = dets[high_mask]
        dets_low  = dets[~high_mask]
        cls_high  = cls_arr[high_mask]
        cls_low   = cls_arr[~high_mask]

        # 1단계: high-score 매칭 (칼만 예측 박스 vs 디텍 박스)
        active_idx = list(range(len(self.tracks)))
        tr_tlbr = np.array([self.tracks[i].tlbr for i in active_idx], dtype=np.float32) if active_idx else np.zeros((0,4), np.float32)
        iou_mat = iou_xyxy(tr_tlbr, dets_high[:, :4]) if dets_high.size else np.zeros((tr_tlbr.shape[0], 0), np.float32)
        # 게이팅 적용 (마할라노비스)
        if dets_high.size and len(self.tracks):
            dets_high_xyah = np.stack([tlbr_to_xyah(b) for b in dets_high[:, :4]], axis=0)  # (K,4)
            gate = self._gating_mask(dets_high_xyah, [self.tracks[i] for i in active_idx])
            iou_mat = np.where(gate, iou_mat, -1.0)
        matches, u_tr, u_dt = self._greedy_match(iou_mat, self.match_thresh)

        for (ti, di) in matches:
            t = self.tracks[active_idx[ti]]
            t.update(dets_high[di, :4], dets_high[di, 4])
            t.class_id = int(cls_high[di])

        # 2단계: 남은 트랙 vs low-score 매칭 (낮은 IoU 허들)
        if len(u_tr) and len(dets_low):
            tr_rest = np.array([self.tracks[active_idx[i]].tlbr for i in u_tr], dtype=np.float32)
            iou_low = iou_xyxy(tr_rest, dets_low[:, :4])
            if len(self.tracks):
                dets_low_xyah = np.stack([tlbr_to_xyah(b) for b in dets_low[:, :4]], axis=0)
                gate2 = self._gating_mask(dets_low_xyah, [self.tracks[active_idx[i]] for i in u_tr])
                iou_low = np.where(gate2, iou_low, -1.0)
            matches2, u_tr2, u_dt2 = self._greedy_match(iou_low, self.match_thresh_low)
            for (ti2, di2) in matches2:
                t = self.tracks[active_idx[u_tr[ti2]]]
                t.update(dets_low[di2, :4], dets_low[di2, 4])
                t.class_id = int(cls_low[di2])
            u_tr_final = [u_tr[i] for i in u_tr2]
            u_dt_high_final = [i for i in range(len(dets_high)) if i not in u_dt]
            u_dt_low_final  = u_dt2
        else:
            u_tr_final = u_tr
            u_dt_high_final = u_dt
            u_dt_low_final  = list(range(len(dets_low)))

        # 신규 트랙 생성 (high-score에서 미매칭된 것)
        for di in u_dt_high_final:
            tlbr = dets_high[di, :4]
            if (tlbr[2]-tlbr[0])*(tlbr[3]-tlbr[1]) >= self.min_box_area:
                xyah = tlbr_to_xyah(tlbr)
                tid = self._new_id()
                self.tracks.append(TrackKF(xyah, dets_high[di, 4], int(cls_high[di]), tid, self.kf))

        # 오래 실종된 트랙 제거
        self.tracks = [t for t in self.tracks if t.miss <= self.buffer_ttl]

        # 활성 트랙
        active_tracks = [t for t in self.tracks if t.miss == 0 and t.activated]
        return active_tracks

# ============== 오디오 설정 ==============
MP3_FILE = "/home/jetson/vigil/ji/lion_roar_3sec.mp3"
AUDIO_LEN_SEC = 3.2
assert os.path.exists(MP3_FILE), f"MP3 not found: {MP3_FILE}"
vlc_instance = vlc.Instance('--no-xlib', '--no-video', '--aout=pulse')
player = vlc_instance.media_player_new()
def play_mp3_once():
    media = vlc_instance.media_new(MP3_FILE)
    player.set_media(media)
    player.stop()
    player.play()
    time.sleep(0.05)

# ============== 라벨/환경 ==============
WIDTH, HEIGHT = 640, 480
LABELS_PATH = "/home/jetson/vigil/ji/labels.txt"
TARGET_NAME = "dog"

def normalize(s):
    return " ".join(s.strip().lower().replace("_", " ").split())

label_to_id = {}
with open(LABELS_PATH, "r") as f:
    for idx, line in enumerate(f):
        name = line.strip()
        if name:
            label_to_id[normalize(name)] = idx

target_id = None
for cand in [TARGET_NAME, "dog"]:
    key = normalize(cand)
    if key in label_to_id:
        target_id = label_to_id[key]; break
if target_id is None:
    print(f"[WARN] '{TARGET_NAME}' not in labels. first keys: {list(label_to_id.keys())[:10]}")

# ============== 카메라/표시/모델 ==============
camera = jetson_utils.gstCamera(WIDTH, HEIGHT, "/dev/video0")
display_local = jetson_utils.videoOutput()
display_rtmp  = jetson_utils.videoOutput("rtmp://192.168.68.87/live/jetson1")
net = jetson_inference.detectNet(argv=[
    "--model=/home/jetson/vigil/ji/mb2-ssd-lite_300.onnx",
    f"--labels={LABELS_PATH}",
    "--input-blob=input_0",
    "--output-cvg=scores",
    "--output-bbox=boxes",
    "--threshold=0.3",
    "--verbose"
])

# ============== 트래커/폰트 ==============
# dt는 프레임 간 시간(초). 카메라 FPS가 20이면 dt≈0.05 로 설정해도 좋음.
tracker = BYTETrackerKF(track_thresh=0.5, match_thresh=0.7, match_thresh_low=0.5,
                        buffer_ttl=30, min_box_area=12, gate_maha=True, maha_th=25.0,
                        dt=1.0)
font = jetson_utils.cudaFont()

print("=== Starting ===")
print("DISPLAY:", os.environ.get("DISPLAY"))
print("XDG_SESSION_TYPE:", os.environ.get("XDG_SESSION_TYPE"))

# ============== 타이머/오디오 상태 ==============
present_since = None
audio_playing = False
audio_started = 0.0
start = time.time()

try:
    while True:
        img, w, h = camera.CaptureRGBA()
        if img is None:
            print("Capture failed"); break

        # 1) Detect
        dets = net.Detect(img, w, h)

        # 2) target 클래스만 트래킹
        det_list, cls_list = [], []
        for d in dets:
            if target_id is not None and d.ClassID != target_id:
                continue
            x1, y1, x2, y2 = float(d.Left), float(d.Top), float(d.Right), float(d.Bottom)
            score = float(d.Confidence)
            det_list.append([x1, y1, x2, y2, score])
            cls_list.append(int(d.ClassID))

        det_arr = np.array(det_list, dtype=np.float32) if len(det_list) else np.zeros((0,5), np.float32)
        tracks = tracker.update(det_arr, class_ids=np.array(cls_list) if len(cls_list) else None)

        # 3) 오버레이
        for t in tracks:
            x1, y1, x2, y2 = t.tlbr
            jetson_utils.cudaDrawRect(img, (int(x1), int(y1), int(x2), int(y2)), (0,255,0,255))
            font.OverlayText(img, w, h, f"id:{t.id} s:{t.score:.2f}",
                             int(x1), max(0, int(y1)-18),
                             font.White, font.Gray40)

        # 4) 2초 연속 감지(트랙 기반)
        now = time.time()
        is_present = len(tracks) > 0
        if is_present:
            if present_since is None:
                present_since = now
            elif (now - present_since) >= 2.0 and not audio_playing:
                play_mp3_once()
                audio_started = time.time()
                audio_playing = True
        else:
            present_since = None

        # 5) 오디오 종료
        if audio_playing and (time.time() - audio_started) >= AUDIO_LEN_SEC:
            player.stop()
            audio_playing = False

        # 6) 출력
        if display_local and display_local.IsStreaming():
            display_local.Render(img)
            display_local.SetStatus(f"{net.GetNetworkFPS():.0f} FPS | det={len(dets)} | tracks={len(tracks)}")
        if display_rtmp and display_rtmp.IsStreaming():
            display_rtmp.Render(img)

        if time.time() - start > 100:
            print("Timeout exit"); break

except KeyboardInterrupt:
    print("Interrupted")
finally:
    try: camera.Close()
    except: pass
    try:
        if display_local: display_local.Close()
    except: pass
    try:
        if display_rtmp: display_rtmp.Close()
    except: pass
    try: player.stop()
    except: pass
    print("Done.")
