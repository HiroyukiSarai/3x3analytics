"""
3x3バスケットボール プレー分析ビューワー v3
クリックでコート4点を指定可能

使用方法:
    streamlit run play_analyzer_v3.py
"""

import streamlit as st
import json
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict
import base64
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(
    page_title="3x3 Play Analyzer v3",
    page_icon="🏀",
    layout="wide"
)

# スタイル
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .point-marker {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 定数
PLAYER_COLORS = ['#ff6b6b', '#ffa94d', '#ffd43b', '#4dabf7', '#69db7c', '#b197fc']
COURT_WIDTH = 15.0
COURT_HEIGHT = 11.0
POINT_COLORS = ['#ff0000', '#ff8800', '#ffff00', '#00ff00']
POINT_NAMES = ['① トップ左', '② トップ右', '③ ゴール右', '④ ゴール左']


class HomographyTransformer:
    def __init__(self):
        self.matrix = None
    
    def set_points(self, image_points):
        court_points = [
            (0, 0), (COURT_WIDTH, 0),
            (COURT_WIDTH, COURT_HEIGHT), (0, COURT_HEIGHT)
        ]
        src = np.float32(image_points)
        dst = np.float32(court_points)
        self.matrix, _ = cv2.findHomography(src, dst)
        return self.matrix is not None
    
    def transform_point(self, x, y):
        if self.matrix is None:
            return x, y
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)
        return float(transformed[0][0][0]), float(transformed[0][0][1])


def smooth_trajectory(points, window=3):
    if len(points) < window:
        return points
    smoothed = []
    for i in range(len(points)):
        start = max(0, i - window // 2)
        end = min(len(points), i + window // 2 + 1)
        avg_x = np.mean([p[0] for p in points[start:end]])
        avg_y = np.mean([p[1] for p in points[start:end]])
        smoothed.append((avg_x, avg_y))
    return smoothed


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


class SimpleTracker:
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
    
    def update(self, detections):
        results = []
        matched_tracks = set()
        matched_dets = set()
        
        for det_idx, det in enumerate(detections):
            best_iou = 0.3
            best_track_id = None
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = compute_iou(det, track['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                matched_tracks.add(best_track_id)
                matched_dets.add(det_idx)
                self.tracks[best_track_id] = {'box': det, 'age': 0}
                results.append((best_track_id, *det))
        
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {'box': det, 'age': 0}
                results.append((track_id, *det))
        
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        return results


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def draw_points_on_image(image, points, scale=1.0):
    """画像にポイントを描画"""
    img = image.copy()
    
    for i, (px, py) in enumerate(points):
        color_bgr = tuple(int(POINT_COLORS[i][j:j+2], 16) for j in (1, 3, 5))[::-1]
        color_rgb = color_bgr[::-1]
        cv2.circle(img, (int(px), int(py)), 15, color_rgb, -1)
        cv2.circle(img, (int(px), int(py)), 15, (255, 255, 255), 3)
        cv2.putText(img, str(i+1), (int(px)-8, int(py)+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # 4点あれば線で結ぶ
    if len(points) == 4:
        pts = [(int(p[0]), int(p[1])) for p in points]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i+1) % 4], (255, 255, 255), 2)
    
    return img


def create_court_figure(trajectories):
    fig = go.Figure()
    
    # コート
    fig.add_shape(type="rect", x0=0, y0=0, x1=COURT_WIDTH, y1=COURT_HEIGHT,
                  fillcolor="rgba(34, 139, 34, 0.9)", line=dict(color="white", width=3))
    fig.add_shape(type="rect", x0=4.5, y0=5.5, x1=10.5, y1=9.5,
                  fillcolor="rgba(34, 139, 34, 0.7)", line=dict(color="white", width=2))
    fig.add_shape(type="circle", x0=7.0, y0=10.0, x1=8.0, y1=11.0,
                  fillcolor="rgba(255, 165, 0, 0.9)", line=dict(color="white", width=2))
    
    # 3ポイントアーク
    theta = np.linspace(0, np.pi, 50)
    x_arc = 7.5 + 6.75 * np.cos(theta)
    y_arc = 11 - 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines',
                             line=dict(color='white', width=2), showlegend=False))
    
    # 軌跡
    for player_id, points in trajectories.items():
        if not points or len(points) < 2:
            continue
        color = PLAYER_COLORS[int(player_id) - 1] if int(player_id) <= 6 else '#888888'
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='lines',
                                 line=dict(color=color, width=3),
                                 name=f'選手{player_id}', opacity=0.8))
        fig.add_trace(go.Scatter(x=[x_coords[0]], y=[y_coords[0]], mode='markers',
                                 marker=dict(size=12, color=color, symbol='circle'),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=[x_coords[-1]], y=[y_coords[-1]], mode='markers',
                                 marker=dict(size=10, color=color, symbol='triangle-up'),
                                 showlegend=False))
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, COURT_WIDTH + 0.5], showgrid=False, title="横 (m)"),
        yaxis=dict(range=[-0.5, COURT_HEIGHT + 0.5], showgrid=False, title="縦 (m)", scaleanchor="x"),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=450, showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='white'))
    )
    return fig


def analyze_video(video_path, transformer, progress_callback=None):
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
    except:
        st.error("❌ YOLO11がインストールされていません")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = SimpleTracker(max_age=30)
    trajectories = defaultdict(list)
    first_frame = None
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 3 != 0:  # 3フレームごと
            frame_idx += 1
            continue
        
        if first_frame is None:
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                detections.append(tuple(box.xyxy[0].cpu().numpy()))
        
        tracked = tracker.update(detections)
        
        for track_id, x1, y1, x2, y2 in tracked[:6]:
            foot_x, foot_y = (x1 + x2) / 2, y2
            court_x, court_y = transformer.transform_point(foot_x, foot_y)
            
            if 0 <= court_x <= COURT_WIDTH and 0 <= court_y <= COURT_HEIGHT:
                player_id = ((track_id - 1) % 6) + 1
                # 外れ値フィルタ
                if trajectories[player_id]:
                    last = trajectories[player_id][-1]
                    if np.sqrt((court_x-last[0])**2 + (court_y-last[1])**2) > 2.0:
                        continue
                trajectories[player_id].append((court_x, court_y))
        
        if progress_callback:
            progress_callback(frame_idx / total_frames)
        frame_idx += 1
    
    cap.release()
    
    # スムージング
    result = {}
    for pid, traj in trajectories.items():
        result[pid] = smooth_trajectory(traj, 5) if len(traj) >= 3 else traj
    
    return {
        'trajectories': result,
        'duration': total_frames / fps if fps > 0 else 0,
        'fps': fps,
        'first_frame': first_frame
    }


def main():
    st.markdown("<h1 style='text-align:center'>🏀 3x3 Play Analyzer v3</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#94a3b8'>画像をクリックしてコートの4隅を指定</p>", unsafe_allow_html=True)
    
    # 初期化
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'frame' not in st.session_state:
        st.session_state.frame = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    
    # サイドバー
    with st.sidebar:
        st.markdown("### 📹 動画を選択")
        video_path = st.text_input("動画ファイルパス", placeholder="/path/to/video.mp4")
        
        if video_path and os.path.exists(video_path):
            st.success(f"✅ ファイル確認OK")
            if st.button("📷 フレーム取得", use_container_width=True):
                frame = extract_first_frame(video_path)
                if frame is not None:
                    st.session_state.frame = frame
                    st.session_state.video_path = video_path
                    st.session_state.points = []
                    st.session_state.result = None
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 📍 指定済みポイント")
        for i, (px, py) in enumerate(st.session_state.points):
            st.markdown(f"<span style='color:{POINT_COLORS[i]}'>{POINT_NAMES[i]}</span>: ({int(px)}, {int(py)})", 
                       unsafe_allow_html=True)
        
        if st.session_state.points:
            if st.button("🔄 リセット", use_container_width=True):
                st.session_state.points = []
                st.rerun()
            if st.button("↩️ 1つ戻す", use_container_width=True):
                st.session_state.points.pop()
                st.rerun()
    
    # メインエリア
    if st.session_state.frame is None:
        st.info("👈 サイドバーから動画を選択して「フレーム取得」をクリック")
        return
    
    frame = st.session_state.frame
    h, w = frame.shape[:2]
    
    # 表示用にリサイズ
    max_width = 900
    scale = min(max_width / w, 1.0)
    display_w = int(w * scale)
    display_h = int(h * scale)
    
    display_frame = cv2.resize(frame, (display_w, display_h))
    
    # ポイントをスケーリングして描画
    scaled_points = [(p[0] * scale, p[1] * scale) for p in st.session_state.points]
    display_frame = draw_points_on_image(display_frame, scaled_points)
    
    # 4点未満ならクリック受付
    if len(st.session_state.points) < 4:
        st.markdown(f"### 📍 {POINT_NAMES[len(st.session_state.points)]} をクリック")
        st.markdown("**コートの角（白いラインの交点）をクリックしてください**")
        
        # クリック可能な画像
        coords = streamlit_image_coordinates(
            Image.fromarray(display_frame),
            key=f"court_click_{len(st.session_state.points)}"
        )
        
        if coords is not None:
            # スケールを戻して元の座標に
            orig_x = coords["x"] / scale
            orig_y = coords["y"] / scale
            st.session_state.points.append((orig_x, orig_y))
            st.rerun()
    
    else:
        # 4点指定済み
        st.image(display_frame, caption="コート4点を指定済み", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 分析開始", type="primary", use_container_width=True):
                transformer = HomographyTransformer()
                if transformer.set_points(st.session_state.points):
                    progress = st.progress(0)
                    st.session_state.result = analyze_video(
                        st.session_state.video_path,
                        transformer,
                        progress_callback=lambda p: progress.progress(min(p, 1.0))
                    )
                    st.rerun()
        with col2:
            if st.button("🔄 やり直し", use_container_width=True):
                st.session_state.points = []
                st.rerun()
    
    # 結果表示
    if st.session_state.result:
        st.markdown("---")
        st.markdown("### 📊 分析結果")
        
        result = st.session_state.result
        fig = create_court_figure(result['trajectories'])
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**動画情報**")
            st.text(f"時間: {result['duration']:.1f}秒")
            st.text(f"FPS: {result['fps']:.1f}")
        with col2:
            st.markdown("**移動距離**")
            for pid, traj in result['trajectories'].items():
                if len(traj) > 1:
                    dist = sum(np.sqrt((traj[i][0]-traj[i-1][0])**2 + 
                                      (traj[i][1]-traj[i-1][1])**2) 
                              for i in range(1, len(traj)))
                    team = '🔴' if int(pid) <= 3 else '🔵'
                    st.text(f"{team} 選手{pid}: {dist:.1f}m")
        
        # ダウンロード
        st.download_button(
            "📥 JSONダウンロード",
            json.dumps({
                'trajectories': {k: list(v) for k, v in result['trajectories'].items()},
                'court_points': st.session_state.points,
                'duration': result['duration']
            }, indent=2),
            "trajectory.json", "application/json"
        )


if __name__ == "__main__":
    main()

