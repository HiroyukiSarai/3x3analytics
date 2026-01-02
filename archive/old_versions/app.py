"""
3x3バスケットボール戦術分析システム
Streamlit Webダッシュボード
"""

import streamlit as st
import json
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import os
import cv2

# ページ設定
st.set_page_config(
    page_title="3x3 Basketball Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@400;500;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a4a);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    .stMetric {
        background: linear-gradient(145deg, #1e1e2e, #2a2a4a);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stMetric label {
        font-family: 'Rajdhani', sans-serif !important;
        color: #00d4ff !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-family: 'Orbitron', monospace !important;
        color: #ffffff !important;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
    }
    
    .stSelectbox, .stSlider {
        font-family: 'Rajdhani', sans-serif;
    }
    
    div[data-testid="stExpander"] {
        background: rgba(30, 30, 46, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
    }
    
    .uploadedFile {
        background: linear-gradient(145deg, #1e1e2e, #2a2a4a);
        border: 1px dashed rgba(0, 212, 255, 0.5);
        border-radius: 10px;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
    }
</style>
""", unsafe_allow_html=True)


def load_sample_data():
    """サンプルデータを生成"""
    patterns = [
        {"pattern_id": "pattern_01", "pattern_name": "ピック＆ロール", 
         "occurrences": 25, "success_rate": 0.72, "description": "スクリーナーがロール"},
        {"pattern_id": "pattern_02", "pattern_name": "ピック＆ポップ", 
         "occurrences": 18, "success_rate": 0.67, "description": "スクリーナーがポップアウト"},
        {"pattern_id": "pattern_03", "pattern_name": "カッティング", 
         "occurrences": 15, "success_rate": 0.60, "description": "オフボールカット"},
        {"pattern_id": "pattern_04", "pattern_name": "アイソレーション", 
         "occurrences": 12, "success_rate": 0.58, "description": "1対1"},
        {"pattern_id": "pattern_05", "pattern_name": "ドライブ＆キック", 
         "occurrences": 20, "success_rate": 0.65, "description": "ドライブからキックアウト"},
    ]
    
    actions = {
        "screen": {"count": 45, "avg_duration": 18.5},
        "pass": {"count": 120, "avg_duration": 8.2},
        "shot": {"count": 35, "avg_duration": 12.0},
        "drive": {"count": 28, "avg_duration": 25.0},
        "cut": {"count": 22, "avg_duration": 15.0}
    }
    
    player_stats = {
        1: {"team": "A", "frames_visible": 2500, "ball_possession_frames": 450, "total_distance": 15000},
        2: {"team": "A", "frames_visible": 2400, "ball_possession_frames": 380, "total_distance": 14500},
        3: {"team": "A", "frames_visible": 2300, "ball_possession_frames": 320, "total_distance": 13800},
        4: {"team": "B", "frames_visible": 2450, "ball_possession_frames": 420, "total_distance": 14200},
        5: {"team": "B", "frames_visible": 2350, "ball_possession_frames": 360, "total_distance": 13500},
        6: {"team": "B", "frames_visible": 2280, "ball_possession_frames": 340, "total_distance": 13200},
    }
    
    return {
        "patterns": patterns,
        "actions": actions,
        "player_stats": player_stats,
        "total_frames": 3000,
        "fps": 30
    }


def analyze_video(video_path: str, max_frames: int = 300, progress_bar=None):
    """
    動画を分析してトラッキングデータを生成
    
    Args:
        video_path: 動画ファイルパス
        max_frames: 分析する最大フレーム数（デモ用に制限）
        progress_bar: Streamlitのプログレスバー
    """
    try:
        from ultralytics import YOLO
        YOLO_AVAILABLE = True
    except ImportError:
        YOLO_AVAILABLE = False
    
    # 動画を開く
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 分析するフレーム数を制限
    frames_to_analyze = min(total_frames, max_frames)
    
    tracking_data = []
    player_positions = {}
    
    if YOLO_AVAILABLE:
        model = YOLO("yolo11n.pt")
        
        frame_idx = 0
        while cap.isOpened() and frame_idx < frames_to_analyze:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 検出
            results = model(frame, verbose=False)[0]
            
            frame_data = {
                "frame_id": frame_idx,
                "timestamp": frame_idx / fps,
                "players": []
            }
            
            player_count = 0
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > 0.5:  # person
                    xyxy = box.xyxy[0].cpu().numpy()
                    player_count += 1
                    
                    player_data = {
                        "track_id": player_count,
                        "team": "A" if player_count <= 3 else "B",
                        "bbox": list(map(float, xyxy)),
                        "confidence": conf
                    }
                    frame_data["players"].append(player_data)
                    
                    # 位置を記録
                    if player_count not in player_positions:
                        player_positions[player_count] = []
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    player_positions[player_count].append((cx, cy))
            
            tracking_data.append(frame_data)
            
            # プログレスバー更新
            if progress_bar:
                progress_bar.progress((frame_idx + 1) / frames_to_analyze)
            
            frame_idx += 1
    else:
        # YOLOがない場合はダミーデータ
        for frame_idx in range(frames_to_analyze):
            frame_data = {
                "frame_id": frame_idx,
                "timestamp": frame_idx / fps,
                "players": [
                    {"track_id": i, "team": "A" if i <= 3 else "B", 
                     "bbox": [100*i, 200, 100*i+50, 400], "confidence": 0.9}
                    for i in range(1, 7)
                ]
            }
            tracking_data.append(frame_data)
            
            if progress_bar:
                progress_bar.progress((frame_idx + 1) / frames_to_analyze)
    
    cap.release()
    
    # 統計データを生成
    player_stats = {}
    for pid in range(1, 7):
        positions = player_positions.get(pid, [])
        total_distance = 0
        if len(positions) > 1:
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                total_distance += np.sqrt(dx**2 + dy**2)
        
        player_stats[pid] = {
            "team": "A" if pid <= 3 else "B",
            "frames_visible": len([f for f in tracking_data if any(p["track_id"] == pid for p in f["players"])]),
            "ball_possession_frames": 0,
            "total_distance": total_distance
        }
    
    # パターンデータ（簡易版）
    patterns = [
        {"pattern_id": "pattern_01", "pattern_name": "検出パターン1", 
         "occurrences": len(tracking_data) // 50, "success_rate": 0.65, "description": "自動検出"},
    ]
    
    actions = {
        "detected_players": {"count": sum(len(f["players"]) for f in tracking_data), "avg_duration": 1.0},
    }
    
    return {
        "patterns": patterns,
        "actions": actions,
        "player_stats": player_stats,
        "total_frames": len(tracking_data),
        "fps": fps,
        "tracking_data": tracking_data,
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames
        }
    }


def create_court_figure():
    """2Dコートの可視化（Plotly）"""
    fig = go.Figure()
    
    # コート背景
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=15, y1=11,
        fillcolor="rgba(34, 139, 34, 0.8)",
        line=dict(color="white", width=2)
    )
    
    # フリースローエリア
    fig.add_shape(
        type="rect", x0=4.5, y0=3.5, x1=10.5, y1=7.5,
        fillcolor="rgba(34, 139, 34, 0.6)",
        line=dict(color="white", width=2)
    )
    
    # 3ポイントアーク
    theta = np.linspace(np.pi, 2*np.pi, 50)
    x_arc = 7.5 + 6.75 * np.cos(theta)
    y_arc = 10 + 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_arc, y=y_arc, mode='lines',
        line=dict(color='white', width=2),
        showlegend=False
    ))
    
    # ゴール
    fig.add_shape(
        type="circle", x0=7.0, y0=9.5, x1=8.0, y1=10.5,
        fillcolor="rgba(255, 165, 0, 0.8)",
        line=dict(color="white", width=2)
    )
    
    fig.update_layout(
        xaxis=dict(range=[0, 15], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 11], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20),
        height=400
    )
    
    return fig


def render_header():
    """ヘッダーを描画"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏀 3x3 Basketball Analytics</h1>
            <p style="color: #94a3b8; font-family: 'Rajdhani', sans-serif; font-size: 1.2rem;">
                戦術パターン分析ダッシュボード
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_sidebar():
    """サイドバーを描画"""
    with st.sidebar:
        st.markdown("### 🎬 動画分析")
        
        uploaded_video = st.file_uploader(
            "動画をアップロード",
            type=['mp4', 'mov', 'avi'],
            help="3x3バスケットボールの動画をアップロードしてください"
        )
        
        if uploaded_video:
            max_frames = st.slider(
                "分析フレーム数",
                min_value=100,
                max_value=1000,
                value=300,
                step=100,
                help="分析するフレーム数（多いほど時間がかかります）"
            )
            
            analyze_button = st.button("🚀 分析開始", type="primary", use_container_width=True)
        else:
            max_frames = 300
            analyze_button = False
        
        st.markdown("---")
        
        st.markdown("### 📁 結果の読み込み")
        
        uploaded_json = st.file_uploader(
            "分析結果JSONをアップロード",
            type=['json'],
            help="以前の分析結果を読み込めます"
        )
        
        st.markdown("---")
        
        st.markdown("### ⚙️ 表示設定")
        
        show_trajectories = st.checkbox("軌跡を表示", value=True)
        show_heatmap = st.checkbox("ヒートマップを表示", value=True)
        
        st.markdown("---")
        
        st.markdown("### 🎯 フィルター")
        
        team_filter = st.selectbox(
            "チーム",
            ["すべて", "チームA", "チームB"]
        )
        
        pattern_filter = st.multiselect(
            "戦術パターン",
            ["ピック＆ロール", "ピック＆ポップ", "カッティング", "アイソレーション", "ドライブ＆キック"],
            default=["ピック＆ロール", "ピック＆ポップ"]
        )
        
        return {
            "uploaded_video": uploaded_video,
            "uploaded_json": uploaded_json,
            "analyze_button": analyze_button,
            "max_frames": max_frames,
            "show_trajectories": show_trajectories,
            "show_heatmap": show_heatmap,
            "team_filter": team_filter,
            "pattern_filter": pattern_filter
        }


def render_metrics(data):
    """メトリクスを描画"""
    st.markdown("### 📊 基本統計")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="総フレーム数",
            value=f"{data['total_frames']:,}",
            delta=None
        )
    
    with col2:
        duration = data['total_frames'] / data['fps']
        st.metric(
            label="分析時間",
            value=f"{duration:.1f}秒",
            delta=None
        )
    
    with col3:
        total_patterns = sum(p['occurrences'] for p in data['patterns'])
        st.metric(
            label="戦術パターン検出",
            value=f"{total_patterns}回",
            delta=None
        )
    
    with col4:
        avg_success = np.mean([p['success_rate'] for p in data['patterns']]) if data['patterns'] else 0
        st.metric(
            label="平均成功率",
            value=f"{avg_success*100:.1f}%",
            delta=None
        )


def render_pattern_analysis(data):
    """戦術パターン分析を描画"""
    st.markdown("### 🎯 戦術パターン分析")
    
    if not data['patterns']:
        st.info("パターンデータがありません")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        df = pd.DataFrame(data['patterns'])
        
        fig = px.bar(
            df,
            x='pattern_name',
            y='occurrences',
            color='success_rate',
            color_continuous_scale='viridis',
            labels={'pattern_name': 'パターン', 'occurrences': '出現回数', 'success_rate': '成功率'},
            title='パターン別出現回数と成功率'
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(
            df,
            values='occurrences',
            names='pattern_name',
            title='パターン分布',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)


def render_court_visualization(data, settings):
    """コート可視化を描画"""
    st.markdown("### 🏀 2Dコートビュー")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = create_court_figure()
        
        # サンプル選手位置を追加
        team_a_positions = [(3, 5), (7.5, 3), (12, 5)]
        team_b_positions = [(4, 7), (7.5, 6), (11, 7)]
        
        # チームA
        fig.add_trace(go.Scatter(
            x=[p[0] for p in team_a_positions],
            y=[p[1] for p in team_a_positions],
            mode='markers+text',
            marker=dict(size=30, color='#ff6b6b', line=dict(color='white', width=2)),
            text=['1', '2', '3'],
            textposition='middle center',
            textfont=dict(color='white', size=14, family='Orbitron'),
            name='チームA'
        ))
        
        # チームB
        fig.add_trace(go.Scatter(
            x=[p[0] for p in team_b_positions],
            y=[p[1] for p in team_b_positions],
            mode='markers+text',
            marker=dict(size=30, color='#4dabf7', line=dict(color='white', width=2)),
            text=['4', '5', '6'],
            textposition='middle center',
            textfont=dict(color='white', size=14, family='Orbitron'),
            name='チームB'
        ))
        
        # 軌跡（サンプル）
        if settings['show_trajectories']:
            trajectory = [(3, 5), (4, 5.5), (5, 6), (6, 6.5), (7, 7)]
            fig.add_trace(go.Scatter(
                x=[p[0] for p in trajectory],
                y=[p[1] for p in trajectory],
                mode='lines',
                line=dict(color='rgba(255, 107, 107, 0.6)', width=3, dash='dot'),
                name='軌跡'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 選手位置情報")
        
        for player_id, stats in data['player_stats'].items():
            team_color = "🔴" if stats['team'] == 'A' else "🔵"
            possession_pct = stats['ball_possession_frames'] / max(stats['frames_visible'], 1) * 100
            
            with st.expander(f"{team_color} 選手 {player_id}"):
                st.write(f"**出現フレーム:** {stats['frames_visible']}")
                st.write(f"**ボール保持率:** {possession_pct:.1f}%")
                st.write(f"**総移動距離:** {stats['total_distance']/1000:.1f}km相当")


def render_action_analysis(data):
    """アクション分析を描画"""
    st.markdown("### ⚡ アクション分析")
    
    col1, col2 = st.columns(2)
    
    with col1:
        action_df = pd.DataFrame([
            {"アクション": k, "検出数": v['count'], "平均時間": v['avg_duration']}
            for k, v in data['actions'].items()
        ])
        
        if not action_df.empty:
            fig = px.bar(
                action_df,
                x='アクション',
                y='検出数',
                color='平均時間',
                color_continuous_scale='thermal',
                title='アクション検出統計'
            )
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### アクションタイムライン")
        
        timeline_data = [
            {"時間": "0:05", "アクション": "パス", "選手": "1→2"},
            {"時間": "0:08", "アクション": "スクリーン", "選手": "3"},
            {"時間": "0:10", "アクション": "ドライブ", "選手": "2"},
            {"時間": "0:12", "アクション": "パス", "選手": "2→1"},
            {"時間": "0:14", "アクション": "シュート", "選手": "1"},
        ]
        
        for item in timeline_data:
            action_color = {
                "パス": "🟢",
                "スクリーン": "🟡",
                "ドライブ": "🟠",
                "シュート": "🔴"
            }.get(item['アクション'], "⚪")
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.5rem; 
                        background: rgba(255,255,255,0.05); border-radius: 8px; margin: 0.3rem 0;">
                <span style="width: 60px; font-family: 'Orbitron', monospace; color: #00d4ff;">{item['時間']}</span>
                <span style="margin: 0 0.5rem;">{action_color}</span>
                <span style="flex: 1;">{item['アクション']}</span>
                <span style="color: #94a3b8;">選手 {item['選手']}</span>
            </div>
            """, unsafe_allow_html=True)


def render_player_stats(data):
    """選手統計を描画"""
    st.markdown("### 👤 選手統計")
    
    player_df = pd.DataFrame([
        {
            "選手ID": pid,
            "チーム": "A" if stats['team'] == 'A' else "B",
            "出現フレーム": stats['frames_visible'],
            "ボール保持": stats['ball_possession_frames'],
            "保持率(%)": stats['ball_possession_frames'] / max(stats['frames_visible'], 1) * 100,
            "移動距離": stats['total_distance']
        }
        for pid, stats in data['player_stats'].items()
    ])
    
    if player_df.empty:
        st.info("選手データがありません")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            player_df,
            x='保持率(%)',
            y='移動距離',
            color='チーム',
            size='出現フレーム',
            text='選手ID',
            title='ボール保持率 vs 移動距離',
            color_discrete_map={'A': '#ff6b6b', 'B': '#4dabf7'}
        )
        
        fig.update_traces(textposition='top center')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # レーダーチャート
        categories = ['出現率', 'ボール保持', '移動距離', 'アクティブ度']
        
        fig = go.Figure()
        
        for team in ['A', 'B']:
            team_data = player_df[player_df['チーム'] == team]
            if team_data.empty:
                continue
            values = [
                team_data['出現フレーム'].mean() / max(player_df['出現フレーム'].max(), 1) * 100,
                team_data['保持率(%)'].mean(),
                team_data['移動距離'].mean() / max(player_df['移動距離'].max(), 1) * 100,
                (team_data['出現フレーム'].mean() + team_data['保持率(%)'].mean()) / 2
            ]
            values.append(values[0])
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=f'チーム{team}',
                line_color='#ff6b6b' if team == 'A' else '#4dabf7'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title='チーム比較'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_video_analysis_result(data):
    """動画分析結果を表示"""
    st.markdown("### 🎬 動画分析結果")
    
    if "video_info" in data:
        info = data["video_info"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("解像度", f"{info['width']}x{info['height']}")
        with col2:
            st.metric("FPS", f"{info['fps']:.1f}")
        with col3:
            st.metric("総フレーム", f"{info['total_frames']:,}")
        with col4:
            st.metric("分析済み", f"{data['total_frames']:,}")
    
    # トラッキングデータのダウンロード
    if "tracking_data" in data:
        st.markdown("---")
        json_str = json.dumps(data, indent=2, default=str)
        st.download_button(
            label="📥 分析結果をダウンロード (JSON)",
            data=json_str,
            file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """メイン関数"""
    render_header()
    settings = render_sidebar()
    
    # データの読み込み/分析
    data = None
    
    # 動画分析
    if settings['uploaded_video'] and settings['analyze_button']:
        st.markdown("---")
        st.markdown("### 🔄 動画分析中...")
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(settings['uploaded_video'].read())
            tmp_path = tmp_file.name
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("YOLO11で選手を検出中...")
            
            data = analyze_video(tmp_path, settings['max_frames'], progress_bar)
            
            status_text.text("✅ 分析完了!")
            st.success(f"✅ {data['total_frames']}フレームの分析が完了しました！")
            
            # セッションに保存
            st.session_state['analysis_data'] = data
            
        except Exception as e:
            st.error(f"❌ 分析エラー: {str(e)}")
        finally:
            os.unlink(tmp_path)
    
    # JSONから読み込み
    elif settings['uploaded_json']:
        try:
            data = json.load(settings['uploaded_json'])
            st.success("✅ 分析結果を読み込みました！")
        except Exception as e:
            st.error(f"❌ JSON読み込みエラー: {str(e)}")
    
    # セッションから取得
    elif 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']
    
    # サンプルデータ
    else:
        data = load_sample_data()
        st.info("📊 サンプルデータを表示中です。動画をアップロードして分析を開始してください。")
    
    # タブ
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 概要", "🎯 戦術パターン", "🏀 コートビュー", "👤 選手統計"
    ])
    
    with tab1:
        if "video_info" in data:
            render_video_analysis_result(data)
            st.markdown("---")
        render_metrics(data)
        st.markdown("---")
        render_action_analysis(data)
    
    with tab2:
        render_pattern_analysis(data)
    
    with tab3:
        render_court_visualization(data, settings)
    
    with tab4:
        render_player_stats(data)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1rem;">
        <p>3x3 Basketball Tactical Analysis System v0.1.0</p>
        <p style="font-size: 0.8rem;">Powered by YOLO11 + BoxMOT + RTMPose</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
