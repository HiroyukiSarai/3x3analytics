"""
3x3バスケットボール プレー分析ビューワー v2
ホモグラフィ変換対応版（斜めアングル対応）

使用方法:
    streamlit run play_analyzer_v2.py
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

st.set_page_config(
    page_title="3x3 Play Analyzer v2",
    page_icon="🏀",
    layout="wide"
)

# スタイル
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;500;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .court-point {
        background: rgba(255, 0, 0, 0.8);
        border-radius: 50%;
        width: 20px;
        height: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 定数
PLAYER_COLORS = [
    '#ff6b6b', '#ffa94d', '#ffd43b',  # チームA
    '#4dabf7', '#69db7c', '#b197fc',  # チームB
]

# 3x3コートの実際のサイズ（メートル）
COURT_WIDTH = 15.0   # 横幅
COURT_HEIGHT = 11.0  # 縦（ゴールからトップまで）


class HomographyTransformer:
    """ホモグラフィ変換クラス"""
    
    def __init__(self):
        self.matrix = None
        self.src_points = None  # 画像上の4点
        self.dst_points = None  # コート上の4点
    
    def set_points(self, image_points, court_points=None):
        """
        4点を設定してホモグラフィ行列を計算
        
        image_points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] - 画像上の4点
                      順序: 左上、右上、右下、左下（コートのコーナー）
        court_points: コート上の対応点（デフォルトはコート全体）
        """
        if court_points is None:
            # デフォルト: コート全体（15m x 11m）
            court_points = [
                (0, 0),           # 左上（トップ左）
                (COURT_WIDTH, 0), # 右上（トップ右）
                (COURT_WIDTH, COURT_HEIGHT),  # 右下（ゴール右）
                (0, COURT_HEIGHT) # 左下（ゴール左）
            ]
        
        self.src_points = np.float32(image_points)
        self.dst_points = np.float32(court_points)
        
        # ホモグラフィ行列を計算
        self.matrix, _ = cv2.findHomography(self.src_points, self.dst_points)
        
        return self.matrix is not None
    
    def transform_point(self, x, y):
        """画像座標をコート座標に変換"""
        if self.matrix is None:
            return x, y
        
        point = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)
        
        return float(transformed[0][0][0]), float(transformed[0][0][1])
    
    def transform_points(self, points):
        """複数の点を一度に変換"""
        if self.matrix is None or len(points) == 0:
            return points
        
        pts = np.array([[p] for p in points], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.matrix)
        
        return [(float(p[0][0]), float(p[0][1])) for p in transformed]


def smooth_trajectory(points, window=3):
    """軌跡を移動平均でスムージング"""
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
    """2つのボックスのIoUを計算"""
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
    """シンプルなIoUベースのトラッカー"""
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
    """動画から最初のフレームを抽出"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # BGRからRGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def create_court_figure(trajectories, title=""):
    """2Dコート図を作成"""
    fig = go.Figure()
    
    # コート背景
    fig.add_shape(type="rect", x0=0, y0=0, x1=COURT_WIDTH, y1=COURT_HEIGHT,
                  fillcolor="rgba(34, 139, 34, 0.9)", line=dict(color="white", width=3))
    
    # フリースローエリア
    fig.add_shape(type="rect", x0=4.5, y0=5.5, x1=10.5, y1=9.5,
                  fillcolor="rgba(34, 139, 34, 0.7)", line=dict(color="white", width=2))
    
    # ゴール（下部）
    fig.add_shape(type="circle", x0=7.0, y0=10.0, x1=8.0, y1=11.0,
                  fillcolor="rgba(255, 165, 0, 0.9)", line=dict(color="white", width=2))
    
    # 3ポイントアーク
    theta = np.linspace(0, np.pi, 50)
    x_arc = 7.5 + 6.75 * np.cos(theta)
    y_arc = 11 - 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines',
                             line=dict(color='white', width=2), showlegend=False))
    
    # トップライン
    fig.add_shape(type="line", x0=0, y0=0, x1=COURT_WIDTH, y1=0,
                  line=dict(color="yellow", width=3, dash="dash"))
    fig.add_annotation(x=7.5, y=-0.5, text="トップ（ボールチェック）", 
                      font=dict(color="yellow", size=12), showarrow=False)
    
    # 軌跡を描画
    for player_id, points in trajectories.items():
        if not points or len(points) < 2:
            continue
        
        color = PLAYER_COLORS[int(player_id) - 1] if int(player_id) <= 6 else '#888888'
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # 軌跡ライン
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color=color, width=3),
            name=f'選手{player_id}',
            opacity=0.8
        ))
        
        # 開始点
        fig.add_trace(go.Scatter(
            x=[x_coords[0]], y=[y_coords[0]],
            mode='markers+text',
            marker=dict(size=15, color=color, symbol='circle',
                       line=dict(color='white', width=2)),
            text=[str(player_id)],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            showlegend=False
        ))
        
        # 終了点
        fig.add_trace(go.Scatter(
            x=[x_coords[-1]], y=[y_coords[-1]],
            mode='markers',
            marker=dict(size=12, color=color, symbol='triangle-up',
                       line=dict(color='white', width=2)),
            showlegend=False
        ))
    
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-1, COURT_WIDTH + 1], showgrid=False, zeroline=False, 
                   showticklabels=True, title="横 (m)"),
        yaxis=dict(range=[-1, COURT_HEIGHT + 1], showgrid=False, zeroline=False, 
                   showticklabels=True, title="縦 (m)", scaleanchor="x"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=10, t=40, b=40),
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                   bgcolor='rgba(0,0,0,0.5)', font=dict(color='white', size=10))
    )
    
    return fig


def analyze_video_with_homography(video_path, transformer, max_frames=None, progress_callback=None):
    """ホモグラフィ変換を使って動画を分析"""
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        yolo_available = True
    except:
        yolo_available = False
        st.error("❌ YOLO11がインストールされていません")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    tracker = SimpleTracker(max_age=30)
    trajectories = defaultdict(list)
    first_frame = None
    
    frame_idx = 0
    frame_skip = 3
    
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # 最初のフレームを保存
        if first_frame is None:
            first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLO検出
        results = model(frame, verbose=False)[0]
        player_detections = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if cls == 0 and conf > 0.5:  # person
                player_detections.append(tuple(xyxy))
        
        # トラッキング
        tracked_players = tracker.update(player_detections)
        
        # 座標変換と記録
        for track_id, x1, y1, x2, y2 in tracked_players[:6]:
            # 足元の位置
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            # ホモグラフィ変換
            court_x, court_y = transformer.transform_point(foot_x, foot_y)
            
            # コート範囲内のみ記録
            if 0 <= court_x <= COURT_WIDTH and 0 <= court_y <= COURT_HEIGHT:
                player_id = ((track_id - 1) % 6) + 1
                
                # 外れ値フィルタリング（急激なワープを除去）
                if trajectories[player_id]:
                    last_x, last_y = trajectories[player_id][-1]
                    distance = np.sqrt((court_x - last_x)**2 + (court_y - last_y)**2)
                    # 1フレームで2m以上移動は異常（約20m/s = 72km/h以上）
                    if distance > 2.0:
                        continue  # この点はスキップ
                
                trajectories[player_id].append((court_x, court_y))
        
        if progress_callback:
            progress_callback(frame_idx / total_frames)
        
        frame_idx += 1
    
    cap.release()
    
    # 軌跡をスムージング
    smoothed_trajectories = {}
    for player_id, traj in trajectories.items():
        if len(traj) >= 3:
            smoothed_trajectories[player_id] = smooth_trajectory(traj, window=5)
        else:
            smoothed_trajectories[player_id] = traj
    
    return {
        'trajectories': smoothed_trajectories,
        'duration': total_frames / fps if fps > 0 else 0,
        'fps': fps,
        'total_frames': total_frames,
        'first_frame': first_frame
    }


def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🏀 3x3 Play Analyzer v2</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            ホモグラフィ変換対応（斜めアングル対応）
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ステップ管理
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'court_points' not in st.session_state:
        st.session_state.court_points = []
    if 'video_path_v2' not in st.session_state:
        st.session_state.video_path_v2 = None
    if 'first_frame' not in st.session_state:
        st.session_state.first_frame = None
    
    # サイドバー
    with st.sidebar:
        st.markdown("### 📹 動画を選択")
        
        video_path_input = st.text_input(
            "動画ファイルのパス",
            placeholder="/path/to/video.mp4",
            key="video_path_input_v2"
        )
        
        if video_path_input and os.path.exists(video_path_input):
            file_size = os.path.getsize(video_path_input) / (1024**3)
            st.success(f"✅ {file_size:.2f} GB")
            
            if st.button("📷 最初のフレームを取得", use_container_width=True):
                frame = extract_first_frame(video_path_input)
                if frame is not None:
                    st.session_state.first_frame = frame
                    st.session_state.video_path_v2 = video_path_input
                    st.session_state.step = 2
                    st.session_state.court_points = []
                    st.rerun()
        elif video_path_input:
            st.error("❌ ファイルが見つかりません")
        
        st.markdown("---")
        st.markdown("### 📤 またはアップロード")
        uploaded_video = st.file_uploader("動画ファイル", type=['mp4', 'mov', 'avi'])
        
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
            
            if st.button("📷 最初のフレームを取得 ", use_container_width=True):
                frame = extract_first_frame(tmp_path)
                if frame is not None:
                    st.session_state.first_frame = frame
                    st.session_state.video_path_v2 = tmp_path
                    st.session_state.step = 2
                    st.session_state.court_points = []
                    st.rerun()
    
    # メインエリア
    st.markdown("---")
    
    # ステップ1: 動画選択
    if st.session_state.step == 1:
        st.markdown("### 📍 ステップ 1: 動画を選択")
        st.info("👈 サイドバーから動画を選択してください")
    
    # ステップ2: コートの4点を指定
    elif st.session_state.step == 2:
        st.markdown("### 📍 ステップ 2: コートの4隅をクリック")
        st.markdown("""
        **以下の順序でコートの4隅をクリックしてください：**
        1. 🔴 トップ左（ボールチェックライン左端）
        2. 🟠 トップ右（ボールチェックライン右端）
        3. 🟡 ゴール右（エンドライン右端）
        4. 🟢 ゴール左（エンドライン左端）
        """)
        
        if st.session_state.first_frame is not None:
            frame = st.session_state.first_frame
            h, w = frame.shape[:2]
            
            # 縮小表示（幅800px）
            display_width = 800
            scale = display_width / w
            display_height = int(h * scale)
            
            # 画像をPILに変換
            pil_image = Image.fromarray(frame)
            pil_image = pil_image.resize((display_width, display_height))
            
            # 既存のポイントを描画
            draw_frame = np.array(pil_image).copy()
            colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0)]
            labels = ['1.トップ左', '2.トップ右', '3.ゴール右', '4.ゴール左']
            
            for i, (px, py) in enumerate(st.session_state.court_points):
                scaled_x = int(px * scale)
                scaled_y = int(py * scale)
                cv2.circle(draw_frame, (scaled_x, scaled_y), 10, colors[i], -1)
                cv2.putText(draw_frame, labels[i], (scaled_x + 15, scaled_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
            
            # 4点揃ったら線で結ぶ
            if len(st.session_state.court_points) == 4:
                pts = [(int(p[0] * scale), int(p[1] * scale)) for p in st.session_state.court_points]
                for i in range(4):
                    cv2.line(draw_frame, pts[i], pts[(i+1) % 4], (255, 255, 255), 2)
            
            # 画像を表示
            st.image(draw_frame, caption="クリックしてコートの4隅を指定", use_container_width=True)
            
            # クリック座標入力（手動）
            col1, col2, col3 = st.columns(3)
            with col1:
                click_x = st.number_input("X座標", 0, w, w//2, key="click_x")
            with col2:
                click_y = st.number_input("Y座標", 0, h, h//2, key="click_y")
            with col3:
                if st.button("➕ ポイント追加"):
                    if len(st.session_state.court_points) < 4:
                        st.session_state.court_points.append((click_x, click_y))
                        st.rerun()
            
            # 現在のポイント表示
            st.markdown(f"**指定済みポイント: {len(st.session_state.court_points)}/4**")
            for i, (px, py) in enumerate(st.session_state.court_points):
                st.text(f"  {labels[i]}: ({px}, {py})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 リセット"):
                    st.session_state.court_points = []
                    st.rerun()
            with col2:
                if st.button("↩️ 1つ戻す"):
                    if st.session_state.court_points:
                        st.session_state.court_points.pop()
                        st.rerun()
            with col3:
                if len(st.session_state.court_points) == 4:
                    if st.button("✅ 確定して分析", type="primary"):
                        st.session_state.step = 3
                        st.rerun()
    
    # ステップ3: 分析実行
    elif st.session_state.step == 3:
        st.markdown("### 📍 ステップ 3: 分析実行")
        
        if len(st.session_state.court_points) == 4:
            # ホモグラフィ変換器を作成
            transformer = HomographyTransformer()
            success = transformer.set_points(st.session_state.court_points)
            
            if success:
                st.success("✅ ホモグラフィ行列を計算しました")
                
                # 分析実行
                progress = st.progress(0)
                status = st.empty()
                
                def update_progress(p):
                    progress.progress(p)
                    status.text(f"分析中... {int(p*100)}%")
                
                result = analyze_video_with_homography(
                    st.session_state.video_path_v2,
                    transformer,
                    progress_callback=update_progress
                )
                
                if result:
                    status.text("✅ 分析完了！")
                    st.session_state.analysis_result = result
                    st.session_state.step = 4
                    st.rerun()
            else:
                st.error("❌ ホモグラフィ行列の計算に失敗しました")
    
    # ステップ4: 結果表示
    elif st.session_state.step == 4:
        st.markdown("### 📊 分析結果")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            
            # 軌跡図を表示
            fig = create_court_figure(result['trajectories'], "選手の軌跡")
            st.plotly_chart(fig, use_container_width=True)
            
            # 統計情報
            st.markdown("### 📈 統計")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**動画情報**")
                st.text(f"再生時間: {result['duration']:.1f}秒")
                st.text(f"FPS: {result['fps']:.1f}")
                st.text(f"フレーム数: {result['total_frames']}")
            
            with col2:
                st.markdown("**選手の動き**")
                for player_id, traj in result['trajectories'].items():
                    if traj and len(traj) > 1:
                        dist = sum(
                            np.sqrt((traj[i][0]-traj[i-1][0])**2 + (traj[i][1]-traj[i-1][1])**2)
                            for i in range(1, len(traj))
                        )
                        team = 'A' if int(player_id) <= 3 else 'B'
                        color = '🔴' if team == 'A' else '🔵'
                        st.text(f"{color} 選手{player_id}: {dist:.1f}m移動")
            
            # 最初のフレームも表示
            if result.get('first_frame') is not None:
                st.markdown("### 📷 分析したフレーム")
                st.image(result['first_frame'], use_container_width=True)
            
            # リセットボタン
            if st.button("🔄 新しい動画を分析"):
                st.session_state.step = 1
                st.session_state.court_points = []
                st.session_state.first_frame = None
                st.session_state.video_path_v2 = None
                if 'analysis_result' in st.session_state:
                    del st.session_state.analysis_result
                st.rerun()
            
            # データダウンロード
            st.download_button(
                "📥 軌跡データをダウンロード (JSON)",
                data=json.dumps({
                    'trajectories': {k: list(v) for k, v in result['trajectories'].items()},
                    'duration': result['duration'],
                    'court_points': st.session_state.court_points
                }, indent=2),
                file_name="trajectory_data.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()

