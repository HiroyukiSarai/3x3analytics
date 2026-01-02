"""
3x3バスケットボール プレー分析ビューワー v4
スライダーでコート4点を指定（確実に動作）

使用方法:
    streamlit run play_analyzer_v4.py
"""

import streamlit as st
import json
import numpy as np
import cv2
import tempfile
import os
import plotly.graph_objects as go
from collections import defaultdict
from PIL import Image

st.set_page_config(page_title="3x3 Play Analyzer v4", page_icon="🏀", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%); }
    h1, h2, h3 { color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

PLAYER_COLORS = ['#ff6b6b', '#ffa94d', '#ffd43b', '#4dabf7', '#69db7c', '#b197fc']
COURT_WIDTH, COURT_HEIGHT = 15.0, 11.0
POINT_COLORS = [(255,0,0), (255,165,0), (255,255,0), (0,255,0)]
POINT_NAMES = ['① ゴール左(奥左)', '② ゴール右(奥右)', '③ トップ右(手前右)', '④ トップ左(手前左)']


class HomographyTransformer:
    def __init__(self):
        self.matrix = None
    
    def set_points(self, image_points):
        # ゴール側が奥（Y=11）、トップ側が手前（Y=0）
        court_points = [(0, COURT_HEIGHT), (COURT_WIDTH, COURT_HEIGHT), 
                        (COURT_WIDTH, 0), (0, 0)]
        src = np.float32(image_points)
        dst = np.float32(court_points)
        self.matrix, _ = cv2.findHomography(src, dst)
        return self.matrix is not None
    
    def transform(self, x, y):
        if self.matrix is None:
            return x, y
        pt = np.array([[[x, y]]], dtype=np.float32)
        t = cv2.perspectiveTransform(pt, self.matrix)
        return float(t[0][0][0]), float(t[0][0][1])


class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
    
    def update(self, dets):
        results = []
        matched = set()
        for det in dets:
            best_id, best_iou = None, 0.3
            for tid, t in self.tracks.items():
                if tid in matched:
                    continue
                iou = self._iou(det, t['box'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id:
                matched.add(best_id)
                self.tracks[best_id] = {'box': det, 'age': 0}
                results.append((best_id, *det))
            else:
                self.tracks[self.next_id] = {'box': det, 'age': 0}
                results.append((self.next_id, *det))
                self.next_id += 1
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > 30:
                    del self.tracks[tid]
        return results
    
    def _iou(self, a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / area if area > 0 else 0


def get_frame(path):
    cap = cv2.VideoCapture(path)
    ret, f = cap.read()
    cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret else None


def draw_points(img, points):
    out = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 12, POINT_COLORS[i], -1)
        cv2.circle(out, (int(x), int(y)), 12, (255,255,255), 2)
        cv2.putText(out, str(i+1), (int(x)-6, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    if len(points) == 4:
        pts = [(int(p[0]), int(p[1])) for p in points]
        for i in range(4):
            cv2.line(out, pts[i], pts[(i+1)%4], (255,255,255), 2)
    return out


def analyze(path, trans, prog):
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
    except:
        st.error("YOLO11がありません")
        return None
    
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = SimpleTracker()
    trajs = defaultdict(list)
    idx = 0
    
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if idx % 3 == 0:
            res = model(f, verbose=False)[0]
            dets = [tuple(b.xyxy[0].cpu().numpy()) for b in res.boxes 
                    if int(b.cls[0]) == 0 and float(b.conf[0]) > 0.5]
            for tid, x1, y1, x2, y2 in tracker.update(dets)[:6]:
                cx, cy = trans.transform((x1+x2)/2, y2)
                if 0 <= cx <= COURT_WIDTH and 0 <= cy <= COURT_HEIGHT:
                    pid = ((tid-1) % 6) + 1
                    if trajs[pid] and np.hypot(cx-trajs[pid][-1][0], cy-trajs[pid][-1][1]) > 2:
                        continue
                    trajs[pid].append((cx, cy))
        prog.progress(min(idx/total, 1.0))
        idx += 1
    cap.release()
    
    # スムージング
    def smooth(pts):
        if len(pts) < 5:
            return pts
        return [(np.mean([p[0] for p in pts[max(0,i-2):i+3]]),
                 np.mean([p[1] for p in pts[max(0,i-2):i+3]])) for i in range(len(pts))]
    
    return {'trajectories': {k: smooth(v) for k, v in trajs.items()},
            'duration': total/fps if fps else 0, 'fps': fps}


def court_fig(trajs):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=15, y1=11,
                  fillcolor="rgba(34,139,34,0.9)", line=dict(color="white", width=3))
    fig.add_shape(type="circle", x0=7, y0=10, x1=8, y1=11,
                  fillcolor="orange", line=dict(color="white", width=2))
    theta = np.linspace(0, np.pi, 50)
    fig.add_trace(go.Scatter(x=7.5+6.75*np.cos(theta), y=11-6.75*np.sin(theta),
                             mode='lines', line=dict(color='white', width=2), showlegend=False))
    for pid, pts in trajs.items():
        if len(pts) < 2:
            continue
        c = PLAYER_COLORS[int(pid)-1]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts],
                                 mode='lines', line=dict(color=c, width=3), name=f'選手{pid}'))
    fig.update_layout(xaxis=dict(range=[-0.5,15.5], showgrid=False),
                      yaxis=dict(range=[-0.5,11.5], showgrid=False, scaleanchor="x"),
                      height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


def main():
    st.title("🏀 3x3 Play Analyzer v4")
    st.caption("スライダーでコートの4隅を指定")
    
    # 初期化
    if 'frame' not in st.session_state:
        st.session_state.frame = None
    if 'path' not in st.session_state:
        st.session_state.path = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # サイドバー
    with st.sidebar:
        st.header("📹 動画選択")
        path = st.text_input("ファイルパス", placeholder="/path/to/video.mp4")
        if path and os.path.exists(path):
            st.success("✅ OK")
            if st.button("フレーム取得", use_container_width=True):
                st.session_state.frame = get_frame(path)
                st.session_state.path = path
                st.session_state.result = None
                st.rerun()
    
    if st.session_state.frame is None:
        st.info("👈 サイドバーで動画を選択")
        return
    
    frame = st.session_state.frame
    h, w = frame.shape[:2]
    
    st.markdown("### 📍 コートの4隅を指定（スライダーで調整）")
    st.markdown("**赤い点を動画上のコート角に合わせてください**")
    
    # 4点のスライダー
    cols = st.columns(4)
    points = []
    
    defaults = [
        (int(w*0.15), int(h*0.35)),  # ゴール左(奥左)
        (int(w*0.75), int(h*0.35)),  # ゴール右(奥右)
        (int(w*0.95), int(h*0.85)),  # トップ右(手前右)
        (int(w*0.05), int(h*0.75)),  # トップ左(手前左)
    ]
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{POINT_NAMES[i]}**")
            x = st.slider(f"X{i+1}", 0, w, defaults[i][0], key=f"x{i}")
            y = st.slider(f"Y{i+1}", 0, h, defaults[i][1], key=f"y{i}")
            points.append((x, y))
    
    # プレビュー
    preview = draw_points(frame, points)
    st.image(preview, caption="プレビュー（4つの点がコートの角に合うように調整）", use_container_width=True)
    
    # 分析ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 分析開始", type="primary", use_container_width=True):
            trans = HomographyTransformer()
            if trans.set_points(points):
                prog = st.progress(0)
                st.session_state.result = analyze(st.session_state.path, trans, prog)
                st.rerun()
    with col2:
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.result = None
            st.rerun()
    
    # 結果
    if st.session_state.result:
        st.markdown("---")
        st.markdown("### 📊 分析結果")
        
        res = st.session_state.result
        st.plotly_chart(court_fig(res['trajectories']), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("時間", f"{res['duration']:.1f}秒")
        with c2:
            for pid, pts in res['trajectories'].items():
                if len(pts) > 1:
                    d = sum(np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]) 
                           for i in range(1, len(pts)))
                    st.text(f"選手{pid}: {d:.1f}m")
        
        st.download_button("📥 JSON", json.dumps({
            'trajectories': {k: list(v) for k, v in res['trajectories'].items()},
            'points': points, 'duration': res['duration']
        }, indent=2), "result.json")


if __name__ == "__main__":
    main()

