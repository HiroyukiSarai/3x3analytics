"""
3x3バスケットボール 軌跡ビューワー
選手の動きを2Dコート上で連続的に可視化

使用方法:
    streamlit run trajectory_viewer.py
"""

import streamlit as st
import json
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(
    page_title="3x3 Trajectory Viewer",
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
</style>
""", unsafe_allow_html=True)


# チームカラー
TEAM_COLORS = {
    'A': ['#ff6b6b', '#ff8787', '#ffa8a8'],  # 赤系
    'B': ['#4dabf7', '#74c0fc', '#a5d8ff'],  # 青系
}

PLAYER_COLORS = [
    '#ff6b6b', '#ffa94d', '#ffd43b',  # チームA: 赤、オレンジ、黄
    '#4dabf7', '#69db7c', '#b197fc',  # チームB: 青、緑、紫
]


def create_court_base():
    """3x3コートのベース図形を作成"""
    shapes = [
        # コート外枠
        dict(type="rect", x0=0, y0=0, x1=15, y1=11,
             fillcolor="rgba(34, 139, 34, 0.9)", line=dict(color="white", width=3)),
        # フリースローエリア
        dict(type="rect", x0=4.5, y0=3.5, x1=10.5, y1=7.5,
             fillcolor="rgba(34, 139, 34, 0.7)", line=dict(color="white", width=2)),
        # フリースローライン
        dict(type="line", x0=4.5, y0=5.5, x1=10.5, y1=5.5,
             line=dict(color="white", width=2)),
        # ゴール
        dict(type="circle", x0=7.0, y0=9.5, x1=8.0, y1=10.5,
             fillcolor="rgba(255, 165, 0, 0.9)", line=dict(color="white", width=2)),
    ]
    return shapes


def create_trajectory_figure(trajectories, current_frame=None, show_all=True):
    """
    軌跡を表示するPlotly図を作成
    
    Args:
        trajectories: {player_id: [(x, y, frame), ...]}
        current_frame: 現在のフレーム（Noneで全体表示）
        show_all: 全軌跡を表示するか
    """
    fig = go.Figure()
    
    # コートを追加
    for shape in create_court_base():
        fig.add_shape(**shape)
    
    # 3ポイントアーク
    theta = np.linspace(np.pi, 2*np.pi, 50)
    x_arc = 7.5 + 6.75 * np.cos(theta)
    y_arc = 10 + 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_arc, y=y_arc, mode='lines',
        line=dict(color='white', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    
    # 各選手の軌跡を描画
    for player_id, points in trajectories.items():
        if not points:
            continue
        
        color = PLAYER_COLORS[player_id - 1] if player_id <= 6 else '#888888'
        team = 'A' if player_id <= 3 else 'B'
        
        # 表示するポイントをフィルタリング
        if current_frame is not None and not show_all:
            display_points = [(x, y, f) for x, y, f in points if f <= current_frame]
        else:
            display_points = points
        
        if not display_points:
            continue
        
        x_coords = [p[0] for p in display_points]
        y_coords = [p[1] for p in display_points]
        
        # 軌跡ライン
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode='lines',
            line=dict(color=color, width=3),
            name=f'選手{player_id} (チーム{team})',
            opacity=0.7,
            hovertemplate=f'選手{player_id}<br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<extra></extra>'
        ))
        
        # 開始点（大きい丸）
        fig.add_trace(go.Scatter(
            x=[x_coords[0]], y=[y_coords[0]],
            mode='markers+text',
            marker=dict(size=20, color=color, line=dict(color='white', width=2)),
            text=[str(player_id)],
            textposition='middle center',
            textfont=dict(color='white', size=12, family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # 現在位置（現在フレーム指定時）
        if current_frame is not None and display_points:
            fig.add_trace(go.Scatter(
                x=[x_coords[-1]], y=[y_coords[-1]],
                mode='markers',
                marker=dict(size=25, color=color, 
                           line=dict(color='white', width=3),
                           symbol='circle'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        xaxis=dict(range=[-0.5, 15.5], showgrid=False, zeroline=False, 
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-0.5, 11.5], showgrid=False, zeroline=False, 
                   showticklabels=False, scaleanchor="x", fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')
        ),
        title=dict(
            text='選手の動き（2Dコートビュー）',
            font=dict(color='white', size=18, family='Orbitron')
        )
    )
    
    return fig


def analyze_video_for_trajectories(video_path, max_frames=500, progress_callback=None):
    """
    動画を分析して軌跡データを生成
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
    
    frames_to_analyze = min(total_frames, max_frames)
    
    # 軌跡データ
    trajectories = {i: [] for i in range(1, 7)}
    
    # 簡易的なトラッキング用
    prev_positions = {}
    
    frame_idx = 0
    while cap.isOpened() and frame_idx < frames_to_analyze:
        ret, frame = cap.read()
        if not ret:
            break
        
        if yolo_available:
            results = model(frame, verbose=False)[0]
            
            current_detections = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > 0.5:  # person
                    xyxy = box.xyxy[0].cpu().numpy()
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = xyxy[3]  # 足元
                    current_detections.append((cx, cy))
            
            # 簡易マッチング（位置が近い検出を同じ選手とみなす）
            current_detections.sort(key=lambda p: p[0])  # X座標でソート
            
            for i, (cx, cy) in enumerate(current_detections[:6]):
                player_id = i + 1
                
                # 画像座標をコート座標に変換（簡易版）
                court_x = cx / width * 15
                court_y = cy / height * 11
                
                trajectories[player_id].append((court_x, court_y, frame_idx))
        else:
            # サンプルデータ
            for player_id in range(1, 7):
                base_x = 3 + (player_id - 1) * 2
                base_y = 5 if player_id <= 3 else 7
                
                court_x = base_x + np.sin(frame_idx * 0.1 + player_id) * 2
                court_y = base_y + np.cos(frame_idx * 0.1 + player_id) * 1.5
                
                trajectories[player_id].append((court_x, court_y, frame_idx))
        
        if progress_callback:
            progress_callback((frame_idx + 1) / frames_to_analyze)
        
        frame_idx += 1
    
    cap.release()
    
    return {
        'trajectories': trajectories,
        'total_frames': frame_idx,
        'fps': fps,
        'video_info': {'width': width, 'height': height, 'total_frames': total_frames}
    }


def main():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🏀 3x3 Trajectory Viewer</h1>
        <p style="color: #94a3b8; font-size: 1.1rem;">選手の動きを2Dコートで軌跡として可視化</p>
    </div>
    """, unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.markdown("### 📹 動画を選択")
        
        input_method = st.radio(
            "入力方法",
            ["ファイルパス指定（推奨）", "アップロード（5GBまで）"],
            index=0
        )
        
        video_path_input = None
        uploaded_video = None
        
        if input_method == "ファイルパス指定（推奨）":
            st.markdown("**大容量動画はこちら**")
            video_path_input = st.text_input(
                "動画ファイルのパス",
                placeholder="/Users/xxx/Videos/game.mp4"
            )
            if video_path_input and not os.path.exists(video_path_input):
                st.error("❌ ファイルが見つかりません")
                video_path_input = None
            elif video_path_input:
                file_size = os.path.getsize(video_path_input) / (1024**3)
                st.success(f"✅ {file_size:.2f} GB")
        else:
            st.markdown("**5GB以下の動画**")
            uploaded_video = st.file_uploader(
                "動画ファイルを選択",
                type=['mp4', 'mov', 'avi'],
                help="3x3バスケットボールの動画"
            )
        
        max_frames = st.slider(
            "分析フレーム数",
            min_value=100,
            max_value=1000,
            value=300,
            step=50,
            help="多いほど長い時間の動きを分析"
        )
        
        if uploaded_video or video_path_input:
            analyze_btn = st.button("🚀 軌跡を分析", type="primary", use_container_width=True)
        else:
            analyze_btn = False
            st.info("動画を指定してください")
        
        st.markdown("---")
        
        st.markdown("### 📁 データ読み込み")
        uploaded_json = st.file_uploader(
            "軌跡データ（JSON）",
            type=['json']
        )
        
        st.markdown("---")
        
        st.markdown("### 🎨 表示設定")
        show_animation = st.checkbox("アニメーション表示", value=False)
        
        if show_animation:
            animation_speed = st.slider("速度", 1, 10, 5)
    
    # メインエリア
    data = None
    
    # 動画分析
    if analyze_btn and (uploaded_video or video_path_input):
        # 動画パスを決定
        tmp_path = None
        should_delete = False
        
        if video_path_input:
            tmp_path = video_path_input
        elif uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_video.read())
                tmp_path = tmp.name
                should_delete = True
        
        try:
            st.markdown("### 🔄 分析中...")
            progress = st.progress(0)
            status = st.empty()
            
            def update_progress(p):
                progress.progress(p)
                status.text(f"フレーム分析中... {int(p*100)}%")
            
            data = analyze_video_for_trajectories(tmp_path, max_frames, update_progress)
            
            status.text("✅ 分析完了!")
            st.session_state['trajectory_data'] = data
            
        finally:
            if should_delete and tmp_path:
                os.unlink(tmp_path)
    
    # JSON読み込み
    elif uploaded_json:
        data = json.load(uploaded_json)
        # 軌跡データをタプルに変換
        if 'trajectories' in data:
            data['trajectories'] = {
                int(k): [tuple(p) for p in v] 
                for k, v in data['trajectories'].items()
            }
    
    # セッションから取得
    elif 'trajectory_data' in st.session_state:
        data = st.session_state['trajectory_data']
    
    # データがある場合は表示
    if data:
        trajectories = data['trajectories']
        total_frames = data['total_frames']
        
        st.markdown("---")
        
        # 統計情報
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("分析フレーム数", f"{total_frames:,}")
        with col2:
            st.metric("分析時間", f"{total_frames / data.get('fps', 30):.1f}秒")
        with col3:
            st.metric("検出選手数", f"{sum(1 for t in trajectories.values() if t)}人")
        with col4:
            total_points = sum(len(t) for t in trajectories.values())
            st.metric("軌跡ポイント数", f"{total_points:,}")
        
        st.markdown("---")
        
        # 軌跡表示
        if show_animation:
            st.markdown("### 🎬 アニメーション再生")
            
            # フレームスライダー
            current_frame = st.slider(
                "フレーム",
                0, total_frames - 1, 0,
                key="frame_slider"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                play_btn = st.button("▶️ 再生", use_container_width=True)
            
            # 軌跡図
            fig = create_trajectory_figure(trajectories, current_frame, show_all=False)
            chart = st.plotly_chart(fig, use_container_width=True, key="trajectory_chart")
            
            # 自動再生
            if play_btn:
                placeholder = st.empty()
                for frame in range(current_frame, total_frames, animation_speed):
                    fig = create_trajectory_figure(trajectories, frame, show_all=False)
                    placeholder.plotly_chart(fig, use_container_width=True)
                    time.sleep(0.05)
        
        else:
            st.markdown("### 📍 全軌跡表示")
            
            # 全軌跡を一度に表示
            fig = create_trajectory_figure(trajectories, show_all=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # 選手別の移動距離
        st.markdown("---")
        st.markdown("### 📊 選手別統計")
        
        stats = []
        for player_id, points in trajectories.items():
            if not points:
                continue
            
            # 移動距離計算
            total_distance = 0
            for i in range(1, len(points)):
                dx = points[i][0] - points[i-1][0]
                dy = points[i][1] - points[i-1][1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            team = 'A' if player_id <= 3 else 'B'
            stats.append({
                '選手ID': player_id,
                'チーム': team,
                '軌跡ポイント数': len(points),
                '総移動距離': f"{total_distance:.1f}m",
                '平均速度': f"{total_distance / max(len(points), 1) * 30:.1f}m/s"
            })
        
        if stats:
            st.dataframe(stats, use_container_width=True)
        
        # データダウンロード
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON形式でダウンロード
            export_data = {
                'trajectories': {k: list(v) for k, v in trajectories.items()},
                'total_frames': total_frames,
                'fps': data.get('fps', 30)
            }
            st.download_button(
                "📥 軌跡データをダウンロード (JSON)",
                data=json.dumps(export_data, indent=2),
                file_name="trajectories.json",
                mime="application/json"
            )
        
        with col2:
            # CSV形式
            csv_lines = ["player_id,team,frame,x,y"]
            for player_id, points in trajectories.items():
                team = 'A' if player_id <= 3 else 'B'
                for x, y, frame in points:
                    csv_lines.append(f"{player_id},{team},{frame},{x:.3f},{y:.3f}")
            
            st.download_button(
                "📥 軌跡データをダウンロード (CSV)",
                data="\n".join(csv_lines),
                file_name="trajectories.csv",
                mime="text/csv"
            )
    
    else:
        # サンプル表示
        st.markdown("---")
        st.info("👆 動画をアップロードして分析を開始してください")
        
        st.markdown("### 📍 サンプル表示")
        
        # サンプル軌跡
        sample_trajectories = {}
        for player_id in range(1, 7):
            points = []
            base_x = 3 + (player_id - 1) * 2
            base_y = 4 if player_id <= 3 else 7
            
            for frame in range(100):
                x = base_x + np.sin(frame * 0.1 + player_id) * 2
                y = base_y + np.cos(frame * 0.08 + player_id) * 1.5
                points.append((x, y, frame))
            
            sample_trajectories[player_id] = points
        
        fig = create_trajectory_figure(sample_trajectories, show_all=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("※ これはサンプルデータです。実際の動画を分析すると選手の動きが表示されます。")


if __name__ == "__main__":
    main()

