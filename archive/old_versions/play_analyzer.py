"""
3x3バスケットボール プレー分析ビューワー
ボールチェックからシュートまでのプレーを切り出し、頻度順に表示

使用方法:
    streamlit run play_analyzer.py
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
from io import BytesIO
from PIL import Image

st.set_page_config(
    page_title="3x3 Play Analyzer",
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
    
    .play-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a4a);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 定数
PLAYER_COLORS = [
    '#ff6b6b', '#ffa94d', '#ffd43b',  # チームA
    '#4dabf7', '#69db7c', '#b197fc',  # チームB
]

# コートの位置定義
COURT_TOP = (7.5, 2)      # トップ（ボールチェック位置）
COURT_GOAL = (7.5, 10)    # ゴール位置


def create_court_with_trajectories(trajectories, highlight_start=True, highlight_end=True):
    """軌跡付きコート図を作成"""
    fig = go.Figure()
    
    # コート背景
    fig.add_shape(type="rect", x0=0, y0=0, x1=15, y1=11,
                  fillcolor="rgba(34, 139, 34, 0.9)", line=dict(color="white", width=3))
    
    # フリースローエリア
    fig.add_shape(type="rect", x0=4.5, y0=3.5, x1=10.5, y1=7.5,
                  fillcolor="rgba(34, 139, 34, 0.7)", line=dict(color="white", width=2))
    
    # ゴール
    fig.add_shape(type="circle", x0=7.0, y0=9.5, x1=8.0, y1=10.5,
                  fillcolor="rgba(255, 165, 0, 0.9)", line=dict(color="white", width=2))
    
    # トップ位置（ボールチェック）
    fig.add_shape(type="circle", x0=6.5, y0=1.5, x1=8.5, y1=3.5,
                  fillcolor="rgba(255, 255, 255, 0.1)", 
                  line=dict(color="yellow", width=2, dash="dash"))
    fig.add_annotation(x=7.5, y=0.5, text="ボールチェック", 
                      font=dict(color="yellow", size=12), showarrow=False)
    
    # 3ポイントアーク
    theta = np.linspace(np.pi, 2*np.pi, 50)
    x_arc = 7.5 + 6.75 * np.cos(theta)
    y_arc = 10 + 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines',
                             line=dict(color='white', width=2), showlegend=False))
    
    # 軌跡を描画
    for player_id, points in trajectories.items():
        if not points:
            continue
        
        color = PLAYER_COLORS[int(player_id) - 1] if int(player_id) <= 6 else '#888888'
        team = 'A' if int(player_id) <= 3 else 'B'
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        # 軌跡ライン
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color=color, width=4),
            name=f'選手{player_id}',
            opacity=0.8
        ))
        
        # 開始点
        if highlight_start and x_coords:
            fig.add_trace(go.Scatter(
                x=[x_coords[0]], y=[y_coords[0]],
                mode='markers+text',
                marker=dict(size=20, color=color, symbol='circle',
                           line=dict(color='white', width=2)),
                text=[str(player_id)],
                textposition='middle center',
                textfont=dict(color='white', size=10),
                showlegend=False
            ))
        
        # 終了点（矢印）
        if highlight_end and len(x_coords) > 1:
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]], y=[y_coords[-1]],
                mode='markers',
                marker=dict(size=15, color=color, symbol='triangle-up',
                           line=dict(color='white', width=2)),
                showlegend=False
            ))
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, 15.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.5, 12], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99,
                   bgcolor='rgba(0,0,0,0.5)', font=dict(color='white', size=10))
    )
    
    return fig


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
        self.tracks = {}  # track_id -> {'box': [x1,y1,x2,y2], 'age': int}
        self.next_id = 1
        self.max_age = max_age
    
    def update(self, detections):
        """
        検出結果でトラックを更新
        detections: [(x1, y1, x2, y2), ...]
        returns: [(track_id, x1, y1, x2, y2), ...]
        """
        results = []
        matched_tracks = set()
        matched_dets = set()
        
        # IoUでマッチング
        for det_idx, det in enumerate(detections):
            best_iou = 0.3  # 最低IoU閾値
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
        
        # マッチしなかった検出は新規トラック
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_dets:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {'box': det, 'age': 0}
                results.append((track_id, *det))
        
        # 古いトラックを削除
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        return results


def detect_plays(video_path, max_plays=20, progress_callback=None):
    """
    動画からプレー（ボールチェック→シュート）を検出
    
    Returns:
        plays: [{
            'start_frame': int,
            'end_frame': int,
            'trajectories': {player_id: [(x, y), ...]},
            'duration': float,
            'thumbnail': base64_image
        }, ...]
    """
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
        yolo_available = True
    except:
        yolo_available = False
        st.warning("⚠️ YOLO11がインストールされていません。サンプルデータを生成します。")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # コートサイズ（実際の3x3コート: 15m x 11m）
    COURT_WIDTH = 15.0
    COURT_HEIGHT = 11.0
    
    plays = []
    current_play = None
    tracker = SimpleTracker(max_age=30)  # トラッカー初期化
    
    # ボールチェック位置の検出用（コート上部中央）
    top_zone_y = height * 0.3  # 上部30%
    goal_zone_y = height * 0.85  # 下部15%
    
    frame_idx = 0
    play_count = 0
    frame_skip = 3  # 3フレームごとに処理（高速化）
    
    # 全体の軌跡も保存
    full_trajectories = defaultdict(list)
    first_frame = None
    
    # 時間ベースでプレーを区切る設定
    PLAY_DURATION_FRAMES = int(fps * 12)  # 12秒ごとにプレーを区切る
    
    # 動画全体を1つのプレーとして処理（シンプル版）
    current_play = {
        'start_frame': 0,
        'trajectories': defaultdict(list),
        'frames': []
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームスキップ
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        player_detections = []  # [(x1, y1, x2, y2), ...]
        
        if yolo_available:
            results = model(frame, verbose=False)[0]
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                if cls == 0 and conf > 0.5:  # person
                    player_detections.append(tuple(xyxy))
        
        # トラッキング更新
        tracked_players = tracker.update(player_detections)
        
        # サムネイル用（最初のフレーム）
        if first_frame is None and len(player_detections) >= 2:
            first_frame = frame.copy()
            current_play['frames'].append(first_frame)
        
        # 全選手の位置を記録
        for track_id, x1, y1, x2, y2 in tracked_players[:6]:  # 最大6人
            # 足元の位置をコート座標に変換
            foot_x = (x1 + x2) / 2
            foot_y = y2  # 足元
            
            # ピクセル座標 → コート座標（正規化）
            court_x = (foot_x / width) * COURT_WIDTH
            court_y = (foot_y / height) * COURT_HEIGHT
            
            # トラックIDを1-6の範囲にマッピング
            player_id = ((track_id - 1) % 6) + 1
            current_play['trajectories'][player_id].append((court_x, court_y))
            full_trajectories[player_id].append((court_x, court_y))
        
        if progress_callback:
            progress_callback(frame_idx / total_frames)
        
        frame_idx += 1
    
    cap.release()
    
    # YOLOがない場合のみサンプルデータ
    if not yolo_available:
        plays = generate_sample_plays()
    else:
        # 動画全体を1つのプレーとして処理
        duration = total_frames / fps if fps > 0 else 0
        
        # サムネイル作成
        thumbnail = None
        if current_play.get('frames'):
            try:
                thumb = current_play['frames'][0]
                thumb = cv2.resize(thumb, (320, 180))
                _, buffer = cv2.imencode('.jpg', thumb)
                thumbnail = base64.b64encode(buffer).decode()
            except:
                pass
        
        # 軌跡データがあれば1つのプレーとして追加
        if sum(len(t) for t in current_play['trajectories'].values()) > 5:
            plays = [{
                'start_frame': 0,
                'end_frame': total_frames,
                'trajectories': dict(current_play['trajectories']),
                'duration': duration,
                'thumbnail': thumbnail,
                'pattern_type': '全プレー'
            }]
        else:
            st.warning(f"⚠️ 選手が検出されませんでした（検出数: {sum(len(t) for t in current_play['trajectories'].values())}）")
            plays = []
    
    return plays, {'fps': fps, 'width': width, 'height': height, 'total_frames': total_frames}


def generate_sample_plays():
    """サンプルプレーデータを生成"""
    plays = []
    
    # パターン1: ピック&ロール風
    for i in range(5):
        trajectories = {}
        for player_id in range(1, 7):
            points = []
            if player_id <= 3:  # チームA（攻撃）
                if player_id == 1:  # ボールハンドラー
                    for t in range(30):
                        x = 7.5 + np.sin(t * 0.2) * 2
                        y = 2 + t * 0.25
                        points.append((x, y))
                elif player_id == 2:  # スクリーナー
                    for t in range(30):
                        x = 6 + (t > 10) * 2
                        y = 5 + (t > 15) * 3
                        points.append((x, y))
                else:  # コーナー
                    for t in range(30):
                        x = 12 + np.sin(t * 0.1)
                        y = 4 + np.cos(t * 0.1)
                        points.append((x, y))
            else:  # チームB（守備）
                for t in range(30):
                    x = 5 + (player_id - 4) * 3 + np.random.randn() * 0.2
                    y = 6 + np.random.randn() * 0.3
                    points.append((x, y))
            trajectories[player_id] = points
        
        plays.append({
            'start_frame': i * 300,
            'end_frame': i * 300 + 90,
            'duration': 3.0,
            'trajectories': trajectories,
            'thumbnail': None,
            'pattern_type': 'ピック&ロール'
        })
    
    # パターン2: カッティング風
    for i in range(3):
        trajectories = {}
        for player_id in range(1, 7):
            points = []
            if player_id <= 3:
                if player_id == 1:  # カッター
                    for t in range(25):
                        x = 3 + t * 0.4
                        y = 3 + t * 0.3
                        points.append((x, y))
                elif player_id == 2:
                    for t in range(25):
                        x = 7.5
                        y = 2 + t * 0.1
                        points.append((x, y))
                else:
                    for t in range(25):
                        x = 12
                        y = 4
                        points.append((x, y))
            else:
                for t in range(25):
                    x = 5 + (player_id - 4) * 3
                    y = 5 + np.random.randn() * 0.2
                    points.append((x, y))
            trajectories[player_id] = points
        
        plays.append({
            'start_frame': 1500 + i * 300,
            'end_frame': 1500 + i * 300 + 75,
            'duration': 2.5,
            'trajectories': trajectories,
            'thumbnail': None,
            'pattern_type': 'カッティング'
        })
    
    # パターン3: アイソレーション風
    for i in range(2):
        trajectories = {}
        for player_id in range(1, 7):
            points = []
            if player_id == 1:  # アイソ
                for t in range(35):
                    x = 7.5 + np.sin(t * 0.3) * 1.5
                    y = 3 + t * 0.2
                    points.append((x, y))
            elif player_id <= 3:
                for t in range(35):
                    x = 3 if player_id == 2 else 12
                    y = 4
                    points.append((x, y))
            else:
                for t in range(35):
                    x = 5 + (player_id - 4) * 3
                    y = 5
                    points.append((x, y))
            trajectories[player_id] = points
        
        plays.append({
            'start_frame': 3000 + i * 300,
            'end_frame': 3000 + i * 300 + 105,
            'duration': 3.5,
            'trajectories': trajectories,
            'thumbnail': None,
            'pattern_type': 'アイソレーション'
        })
    
    return plays


def cluster_plays(plays, n_clusters=5):
    """
    プレーを類似度でクラスタリング
    
    Returns:
        clusters: {pattern_name: [play_indices]}
    """
    if len(plays) < 2:
        return {'全てのプレー': list(range(len(plays)))}
    
    # 特徴ベクトル抽出（簡易版：各選手の開始・終了位置）
    features = []
    for play in plays:
        feat = []
        for player_id in range(1, 7):
            traj = play['trajectories'].get(player_id, [])
            if traj:
                feat.extend([traj[0][0], traj[0][1], traj[-1][0], traj[-1][1]])
            else:
                feat.extend([0, 0, 0, 0])
        features.append(feat)
    
    features = np.array(features)
    
    # 簡易クラスタリング（K-means）
    try:
        from sklearn.cluster import KMeans
        n_clusters = min(n_clusters, len(plays))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
    except:
        # sklearn がない場合は均等分割
        labels = [i % n_clusters for i in range(len(plays))]
    
    # クラスタごとにグループ化
    clusters = defaultdict(list)
    pattern_names = ['パターンA', 'パターンB', 'パターンC', 'パターンD', 'パターンE']
    
    for i, label in enumerate(labels):
        name = plays[i].get('pattern_type', pattern_names[label % len(pattern_names)])
        clusters[name].append(i)
    
    # 頻度順にソート
    sorted_clusters = dict(sorted(clusters.items(), key=lambda x: -len(x[1])))
    
    return sorted_clusters


def extract_video_clip(video_path, start_frame, end_frame, output_path):
    """動画クリップを抽出"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path


def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🏀 3x3 Play Analyzer</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">
            ボールチェック → シュートまでのプレーを分析
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown("### 📹 動画を選択")
        
        video_path_input = None
        uploaded_video = None
        
        # 方法1: ファイルパス指定
        st.markdown("#### 📂 方法1: ファイルパス")
        sample_path = "/Users/saraihiroyuki/3x3analytics/sample_videos/test_sample.mp4"
        default_path = sample_path if os.path.exists(sample_path) else ""
        
        video_path_input = st.text_input(
            "動画ファイルのパス",
            value=default_path,
            placeholder="/path/to/video.mp4",
            key="video_path"
        )
        if video_path_input and not os.path.exists(video_path_input):
            st.error("❌ ファイルが見つかりません")
            video_path_input = None
        elif video_path_input:
            file_size = os.path.getsize(video_path_input) / (1024**3)
            st.success(f"✅ {file_size:.2f} GB")
        
        st.markdown("---")
        
        # 方法2: アップロード
        st.markdown("#### 📤 方法2: アップロード")
        uploaded_video = st.file_uploader(
            "動画ファイル（5GBまで）",
            type=['mp4', 'mov', 'avi'],
            key="video_upload"
        )
        if uploaded_video:
            st.success(f"✅ {uploaded_video.name}")
        
        st.markdown("---")
        
        max_plays = st.slider("最大プレー数", 5, 50, 20)
        
        # どちらかが指定されていれば分析可能
        can_analyze = bool(video_path_input) or bool(uploaded_video)
        
        if can_analyze:
            analyze_btn = st.button("🚀 プレーを検出", type="primary", use_container_width=True)
        else:
            analyze_btn = False
            st.info("👆 動画を指定してください")
        
        st.markdown("---")
        st.markdown("### 📊 表示設定")
        show_all_trajectories = st.checkbox("全選手の軌跡を表示", value=True)
        show_video_clip = st.checkbox("該当動画を表示", value=True)
    
    # 分析実行
    plays = None
    video_info = None
    video_path_temp = None
    
    if analyze_btn and (uploaded_video or video_path_input):
        # 動画パスを決定（アップロード優先）
        if uploaded_video:
            st.info("📤 アップロードされた動画を使用します")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                video_path_temp = tmp.name
        elif video_path_input:
            st.info(f"📂 ファイルパスの動画を使用: {video_path_input}")
            video_path_temp = video_path_input
        
        st.markdown("### 🔄 プレー検出中...")
        progress = st.progress(0)
        
        plays, video_info = detect_plays(
            video_path_temp, max_plays,
            progress_callback=lambda p: progress.progress(p)
        )
        
        st.success(f"✅ {len(plays)}個のプレーを検出しました！")
        st.session_state['plays'] = plays
        st.session_state['video_info'] = video_info
        st.session_state['analyzed_video_path'] = video_path_temp  # キー名を変更
    
    elif 'plays' in st.session_state:
        plays = st.session_state['plays']
        video_info = st.session_state.get('video_info')
        video_path_temp = st.session_state.get('analyzed_video_path')
    
    else:
        # サンプルデータ
        st.info("👆 動画をアップロードしてプレーを検出してください。サンプルデータを表示中...")
        plays = generate_sample_plays()
        video_info = {'fps': 30}
    
    if plays:
        st.markdown("---")
        
        # クラスタリング
        clusters = cluster_plays(plays)
        
        # パターン別タブ
        st.markdown("### 📊 プレーパターン（頻度順）")
        
        tabs = st.tabs([f"{name} ({len(indices)}回)" for name, indices in clusters.items()])
        
        for tab, (pattern_name, play_indices) in zip(tabs, clusters.items()):
            with tab:
                st.markdown(f"#### {pattern_name} - {len(play_indices)}回検出")
                
                for idx in play_indices:
                    play = plays[idx]
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="play-card">
                            <strong>プレー #{idx + 1}</strong> | 
                            時間: {play['duration']:.1f}秒 | 
                            フレーム: {play['start_frame']} - {play['end_frame']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # 軌跡表示
                            if show_all_trajectories:
                                fig = create_court_with_trajectories(play['trajectories'])
                            else:
                                # チームAのみ
                                traj_a = {k: v for k, v in play['trajectories'].items() if int(k) <= 3}
                                fig = create_court_with_trajectories(traj_a)
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"court_{idx}")
                        
                        with col2:
                            # サムネイル or 動画クリップ
                            if play.get('thumbnail'):
                                img_data = base64.b64decode(play['thumbnail'])
                                st.image(img_data, caption="プレー開始時点", use_container_width=True)
                            
                            if show_video_clip and video_path_temp and os.path.exists(video_path_temp):
                                if st.button(f"🎬 動画を見る", key=f"video_btn_{idx}"):
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as clip_tmp:
                                        extract_video_clip(
                                            video_path_temp,
                                            play['start_frame'],
                                            play['end_frame'],
                                            clip_tmp.name
                                        )
                                        st.video(clip_tmp.name)
                            
                            # 統計
                            st.markdown("**📍 選手の動き**")
                            for player_id, traj in play['trajectories'].items():
                                if traj:
                                    # 移動距離
                                    dist = sum(
                                        np.sqrt((traj[i][0]-traj[i-1][0])**2 + (traj[i][1]-traj[i-1][1])**2)
                                        for i in range(1, len(traj))
                                    )
                                    team = 'A' if int(player_id) <= 3 else 'B'
                                    color = '🔴' if team == 'A' else '🔵'
                                    st.text(f"{color} 選手{player_id}: {dist:.1f}m移動")
                        
                        st.markdown("---")
        
        # サマリー
        st.markdown("### 📈 パターンサマリー")
        
        summary_data = []
        for pattern_name, play_indices in clusters.items():
            avg_duration = np.mean([plays[i]['duration'] for i in play_indices])
            summary_data.append({
                'パターン': pattern_name,
                '出現回数': len(play_indices),
                '平均時間(秒)': f"{avg_duration:.1f}"
            })
        
        st.dataframe(summary_data, use_container_width=True)
        
        # データダウンロード
        st.markdown("---")
        export_data = {
            'plays': [
                {
                    'start_frame': p['start_frame'],
                    'end_frame': p['end_frame'],
                    'duration': p['duration'],
                    'trajectories': {k: list(v) for k, v in p['trajectories'].items()}
                }
                for p in plays
            ],
            'clusters': {k: v for k, v in clusters.items()},
            'video_info': video_info
        }
        
        st.download_button(
            "📥 分析結果をダウンロード (JSON)",
            data=json.dumps(export_data, indent=2, default=str),
            file_name="play_analysis.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()

