"""
3x3バスケットボール分析 - シンプル版
クリックで4点指定 → ホモグラフィ変換 → 選手検出 → 追跡 → 2D表示

使用方法:
    streamlit run app_simple.py --server.port 8510
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
from collections import defaultdict
import plotly.graph_objects as go
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image

st.set_page_config(
    page_title="3x3 Analyzer",
    page_icon="🏀",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; }
    h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# 定数
COURT_WIDTH = 15.0  # メートル
COURT_HEIGHT = 11.0
COLORS = ['#ff6b6b', '#ffa94d', '#ffd43b', '#4dabf7', '#69db7c', '#b197fc']
POINT_COLORS = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (0, 255, 255)]
POINT_NAMES = ['① ゴール左(奥左)', '② ゴール右(奥右)', '③ トップ右(手前右)', '④ トップ左(手前左)']


class HomographyTransformer:
    """ホモグラフィ変換クラス"""
    
    def __init__(self):
        self.matrix = None
    
    def set_points(self, image_points):
        """
        4点からホモグラフィ行列を計算
        image_points: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        順序: ゴール左, ゴール右, トップ右, トップ左
        """
        court_points = [
            (0, COURT_HEIGHT),           # ゴール左
            (COURT_WIDTH, COURT_HEIGHT), # ゴール右
            (COURT_WIDTH, 0),            # トップ右
            (0, 0)                       # トップ左
        ]
        
        src = np.float32(image_points)
        dst = np.float32(court_points)
        
        self.matrix, _ = cv2.findHomography(src, dst)
        return self.matrix is not None
    
    def transform(self, px, py):
        """ピクセル座標をコート座標に変換"""
        if self.matrix is None:
            return None, None
        
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)
        
        return float(transformed[0][0][0]), float(transformed[0][0][1])


class SimpleTracker:
    """シンプルなIoUベーストラッカー"""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
    
    def update(self, detections):
        results = []
        matched_tracks = set()
        matched_dets = set()
        
        for di, det in enumerate(detections):
            best_id, best_iou = None, 0.3
            for tid, t in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(det, t['box'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            
            if best_id:
                matched_tracks.add(best_id)
                matched_dets.add(di)
                self.tracks[best_id] = {'box': det, 'age': 0}
                results.append((best_id, *det))
        
        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'box': det, 'age': 0}
                results.append((tid, *det))
        
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
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


def get_first_frame(video_path):
    """動画から最初のフレームを取得"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def draw_points_on_image(frame, points):
    """4点を画像に描画"""
    img = frame.copy()
    
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 12, POINT_COLORS[i], -1)
        cv2.circle(img, (int(x), int(y)), 12, (255, 255, 255), 3)
        cv2.putText(img, str(i+1), (int(x)-6, int(y)+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # 4点あれば線で結ぶ
    if len(points) == 4:
        pts = [(int(x), int(y)) for x, y in points]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i+1) % 4], (255, 255, 255), 2)
    
    return img


def is_point_in_quad(px, py, points):
    """点が四角形内にあるかチェック"""
    if len(points) != 4:
        return False
    
    pts = np.array(points, dtype=np.float32)
    result = cv2.pointPolygonTest(pts, (px, py), False)
    return result >= 0


def create_court_figure(trajectories):
    """2Dコート図を作成"""
    fig = go.Figure()
    
    # コート背景
    fig.add_shape(type="rect", x0=0, y0=0, x1=COURT_WIDTH, y1=COURT_HEIGHT,
                  fillcolor="rgba(34, 139, 34, 0.9)", 
                  line=dict(color="white", width=3))
    
    # フリースローエリア
    fig.add_shape(type="rect", x0=4.5, y0=5.5, x1=10.5, y1=9.5,
                  fillcolor="rgba(34, 139, 34, 0.7)", 
                  line=dict(color="white", width=2))
    
    # ゴール
    fig.add_shape(type="circle", x0=7.0, y0=10.0, x1=8.0, y1=11.0,
                  fillcolor="orange", line=dict(color="white", width=2))
    
    # 3ポイントアーク
    theta = np.linspace(0, np.pi, 50)
    x_arc = 7.5 + 6.75 * np.cos(theta)
    y_arc = 11 - 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_arc, y=y_arc, mode='lines',
        line=dict(color='white', width=2), showlegend=False
    ))
    
    # 軌跡を描画
    for player_id, points in trajectories.items():
        if len(points) < 2:
            continue
        
        color = COLORS[(int(player_id) - 1) % len(COLORS)]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=3),
            name=f'選手 {player_id}',
            opacity=0.8
        ))
        
        # 開始点
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode='markers',
            marker=dict(size=12, color=color, symbol='circle',
                       line=dict(color='white', width=2)),
            showlegend=False
        ))
        
        # 終了点
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode='markers',
            marker=dict(size=10, color=color, symbol='triangle-up',
                       line=dict(color='white', width=2)),
            showlegend=False
        ))
    
    fig.update_layout(
        title="選手の軌跡（2Dコート）",
        xaxis=dict(range=[-0.5, COURT_WIDTH + 0.5], showgrid=False, 
                   zeroline=False, title="横 (m)"),
        yaxis=dict(range=[-0.5, COURT_HEIGHT + 0.5], showgrid=False, 
                   zeroline=False, title="縦 (m)", scaleanchor="x"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        showlegend=True
    )
    
    return fig


def analyze_video(video_path, transformer, court_points, progress_callback=None):
    """動画を分析して軌跡を取得"""
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
    except Exception as e:
        st.error(f"YOLO読み込みエラー: {e}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = SimpleTracker()
    trajectories = defaultdict(list)
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 3 == 0:  # 3フレームごと
            results = model(frame, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > 0.5:  # person
                    xyxy = box.xyxy[0].cpu().numpy()
                    foot_x = (xyxy[0] + xyxy[2]) / 2
                    foot_y = xyxy[3]
                    
                    # コート内かチェック
                    if is_point_in_quad(foot_x, foot_y, court_points):
                        detections.append(tuple(xyxy))
            
            # トラッキング
            tracked = tracker.update(detections)
            
            # 座標変換して記録
            for track_id, bx1, by1, bx2, by2 in tracked[:6]:
                foot_x = (bx1 + bx2) / 2
                foot_y = by2
                
                court_x, court_y = transformer.transform(foot_x, foot_y)
                
                if court_x is not None and 0 <= court_x <= COURT_WIDTH and 0 <= court_y <= COURT_HEIGHT:
                    player_id = ((track_id - 1) % 6) + 1
                    
                    # 外れ値フィルタ
                    if trajectories[player_id]:
                        last = trajectories[player_id][-1]
                        dist = np.sqrt((court_x - last[0])**2 + (court_y - last[1])**2)
                        if dist > 3.0:
                            continue
                    
                    trajectories[player_id].append((court_x, court_y))
        
        if progress_callback:
            progress_callback(frame_idx / total_frames)
        
        frame_idx += 1
    
    cap.release()
    
    # スムージング
    def smooth(pts, window=5):
        if len(pts) < window:
            return pts
        result = []
        for i in range(len(pts)):
            start = max(0, i - window // 2)
            end = min(len(pts), i + window // 2 + 1)
            avg_x = np.mean([p[0] for p in pts[start:end]])
            avg_y = np.mean([p[1] for p in pts[start:end]])
            result.append((float(avg_x), float(avg_y)))
        return result
    
    smoothed = {k: smooth(v) for k, v in trajectories.items() if len(v) >= 3}
    
    return {
        'trajectories': smoothed,
        'duration': total_frames / fps if fps > 0 else 0,
        'fps': fps,
        'total_frames': total_frames
    }


def main():
    st.title("🏀 3x3 Analyzer")
    st.caption("クリックで4点指定 → ホモグラフィ変換 → 2D軌跡表示")
    
    # セッション初期化
    if 'frame' not in st.session_state:
        st.session_state.frame = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'points' not in st.session_state:
        st.session_state.points = []
    if 'click_count' not in st.session_state:
        st.session_state.click_count = 0
    
    # ========== STEP 1: 動画選択 ==========
    st.header("STEP 1: 動画を選択")
    
    video_path = st.text_input("動画ファイルのパス", placeholder="/path/to/video.mp4")
    
    if video_path and os.path.exists(video_path):
        st.success("✅ ファイル確認OK")
        
        if st.button("📷 最初のフレームを取得", use_container_width=True):
            frame = get_first_frame(video_path)
            if frame is not None:
                st.session_state.frame = frame
                st.session_state.video_path = video_path
                st.session_state.result = None
                st.session_state.points = []
                st.session_state.click_count = 0
                st.rerun()
    elif video_path:
        st.error("❌ ファイルが見つかりません")
    
    if st.session_state.frame is None:
        st.info("👆 動画パスを入力して「最初のフレームを取得」をクリック")
        return
    
    # ========== STEP 2: 4点指定（クリック） ==========
    st.header("STEP 2: コートの4隅をクリックで指定")
    
    frame = st.session_state.frame
    h, w = frame.shape[:2]
    
    # 現在の状態を表示
    current_point = len(st.session_state.points)
    
    if current_point < 4:
        next_point_name = POINT_NAMES[current_point]
        st.markdown(f"""
        ### 👆 次にクリック: **{next_point_name}**
        
        | 順番 | 位置 | 状態 |
        |------|------|------|
        | ① ゴール左(奥左) | 画面の奥・左のコート角 | {"✅ 設定済" if current_point > 0 else "⏳ 次"} |
        | ② ゴール右(奥右) | 画面の奥・右のコート角 | {"✅ 設定済" if current_point > 1 else ("⏳ 次" if current_point == 1 else "⏸️ 待機")} |
        | ③ トップ右(手前右) | 画面の手前・右のコート角 | {"✅ 設定済" if current_point > 2 else ("⏳ 次" if current_point == 2 else "⏸️ 待機")} |
        | ④ トップ左(手前左) | 画面の手前・左のコート角 | {"✅ 設定済" if current_point > 3 else ("⏳ 次" if current_point == 3 else "⏸️ 待機")} |
        """)
    else:
        st.success("✅ 4点すべて設定完了！")
    
    # 画像にポイントを描画
    display_frame = draw_points_on_image(frame, st.session_state.points)
    
    # 表示サイズを調整
    display_width = min(900, w)
    scale = display_width / w
    display_height = int(h * scale)
    
    # PIL Imageに変換してリサイズ
    pil_image = Image.fromarray(display_frame)
    pil_image = pil_image.resize((display_width, display_height))
    
    # クリック可能な画像を表示
    st.markdown("**↓ 画像をクリックしてコートの角を指定**")
    
    coords = streamlit_image_coordinates(
        pil_image,
        key=f"click_{st.session_state.click_count}"
    )
    
    # クリックを処理
    if coords is not None and len(st.session_state.points) < 4:
        # 表示座標を元の画像座標に変換
        original_x = coords["x"] / scale
        original_y = coords["y"] / scale
        
        # 新しい点を追加
        st.session_state.points.append((original_x, original_y))
        st.session_state.click_count += 1
        st.rerun()
    
    # ボタン
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ 1つ戻す", use_container_width=True, disabled=len(st.session_state.points) == 0):
            st.session_state.points = st.session_state.points[:-1]
            st.session_state.click_count += 1
            st.rerun()
    with col2:
        if st.button("🔄 全てリセット", use_container_width=True):
            st.session_state.points = []
            st.session_state.click_count += 1
            st.rerun()
    
    # 設定済みの座標を表示
    if st.session_state.points:
        with st.expander("📐 設定済み座標"):
            for i, (x, y) in enumerate(st.session_state.points):
                st.text(f"{POINT_NAMES[i]}: X={x:.0f}, Y={y:.0f}")
    
    # ========== STEP 3: 分析実行 ==========
    st.header("STEP 3: 分析実行")
    
    if len(st.session_state.points) == 4:
        if st.button("🚀 分析開始", type="primary", use_container_width=True):
            transformer = HomographyTransformer()
            if transformer.set_points(st.session_state.points):
                progress = st.progress(0)
                status = st.empty()
                
                def update_progress(p):
                    progress.progress(min(p, 1.0))
                    status.text(f"分析中... {int(p*100)}%")
                
                result = analyze_video(
                    st.session_state.video_path, 
                    transformer, 
                    st.session_state.points,
                    update_progress
                )
                
                if result:
                    st.session_state.result = result
                    status.text("✅ 分析完了！")
                    st.rerun()
            else:
                st.error("ホモグラフィ行列の計算に失敗しました")
    else:
        st.warning(f"⚠️ 4点を指定してください（現在: {len(st.session_state.points)}点）")
    
    # ========== STEP 4: 結果表示 ==========
    if st.session_state.result:
        st.header("STEP 4: 結果")
        
        result = st.session_state.result
        
        # 2Dコート図
        fig = create_court_figure(result['trajectories'])
        st.plotly_chart(fig, use_container_width=True)
        
        # 統計情報
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 動画情報")
            st.text(f"時間: {result['duration']:.1f} 秒")
            st.text(f"FPS: {result['fps']:.1f}")
            st.text(f"フレーム数: {result['total_frames']}")
        
        with col2:
            st.subheader("🏃 選手の移動距離")
            for player_id, pts in result['trajectories'].items():
                if len(pts) > 1:
                    dist = sum(
                        np.sqrt((pts[i][0] - pts[i-1][0])**2 + 
                               (pts[i][1] - pts[i-1][1])**2)
                        for i in range(1, len(pts))
                    )
                    color = '🔴' if player_id <= 3 else '🔵'
                    st.text(f"{color} 選手{player_id}: {dist:.1f}m")
        
        # データダウンロード
        st.subheader("📥 データダウンロード")
        
        export_data = {
            'trajectories': {str(k): [(float(p[0]), float(p[1])) for p in v] 
                            for k, v in result['trajectories'].items()},
            'court_points': [(float(p[0]), float(p[1])) for p in st.session_state.points],
            'duration': float(result['duration']),
            'fps': float(result['fps'])
        }
        
        st.download_button(
            "📥 軌跡データをダウンロード (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="trajectory.json",
            mime="application/json"
        )
        
        if st.button("🔄 新しい動画を分析", use_container_width=True):
            st.session_state.frame = None
            st.session_state.video_path = None
            st.session_state.result = None
            st.session_state.points = []
            st.session_state.click_count = 0
            st.rerun()


if __name__ == "__main__":
    main()
