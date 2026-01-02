"""
3x3バスケットボール分析システム - Roboflow/Supervision版
選手検出 → トラッキング → チーム分類 → ホモグラフィ変換 → 2D表示

使用方法:
    streamlit run analyzer.py --server.port 8510
"""

import streamlit as st
import cv2
import numpy as np
import json
import os
import sys
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
import plotly.graph_objects as go
import torch

# roboflow_sportsをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'roboflow_sports'))

from sports.common.view import ViewTransformer

st.set_page_config(
    page_title="3x3 Analyzer v2",
    page_icon="🏀",
    layout="wide"
)

# スタイル
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# 定数
COURT_WIDTH = 15.0
COURT_HEIGHT = 11.0
TEAM_COLORS = {
    0: {'name': 'チームA', 'color': '#ff6b6b', 'emoji': '🔴'},
    1: {'name': 'チームB', 'color': '#4dabf7', 'emoji': '🔵'}
}
POINT_COLORS = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (0, 255, 255)]
POINT_NAMES = ['① ゴール左(奥左)', '② ゴール右(奥右)', '③ トップ右(手前右)', '④ トップ左(手前左)']


def get_device():
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SimpleTracker:
    """シンプルなIoUベーストラッカー（フォールバック用）"""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
    
    def update(self, detections: List[Tuple]) -> List[Tuple]:
        results = []
        matched_tracks = set()
        matched_dets = set()
        
        for di, det in enumerate(detections):
            best_id, best_iou = None, self.iou_threshold
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
                if self.tracks[tid]['age'] > self.max_age:
                    del self.tracks[tid]
        
        return results
    
    def _iou(self, a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / area if area > 0 else 0


class ByteTrackWrapper:
    """ByteTrackトラッカーのラッパー"""
    
    def __init__(self):
        self.tracker = None
        self.initialized = False
    
    def initialize(self, frame_shape):
        """トラッカーを初期化"""
        if self.initialized:
            return
        
        try:
            from boxmot import BYTETracker
            self.tracker = BYTETracker()
            self.initialized = True
        except Exception as e:
            print(f"ByteTrack初期化エラー: {e}")
            self.tracker = None
    
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        detections: [x1, y1, x2, y2, conf, cls] の配列
        戻り値: [x1, y1, x2, y2, track_id, conf, cls] の配列
        """
        if self.tracker is None or len(detections) == 0:
            return np.array([])
        
        try:
            # ByteTrackerのupdate
            tracks = self.tracker.update(detections, frame)
            return tracks
        except Exception as e:
            print(f"ByteTrack更新エラー: {e}")
            return np.array([])


class TeamClassifierWrapper:
    """チーム分類器のラッパー（MPS対応）"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.classifier = None
        self.is_fitted = False
        
    def initialize(self):
        """モデルを初期化"""
        if self.classifier is not None:
            return
        
        try:
            from sports.common.team import TeamClassifier
            # MPSの場合はCPUにフォールバック（互換性のため）
            actual_device = 'cpu' if self.device == 'mps' else self.device
            self.classifier = TeamClassifier(device=actual_device, batch_size=16)
        except Exception as e:
            st.error(f"TeamClassifier初期化エラー: {e}")
            self.classifier = None
    
    def fit_and_predict(self, crops: List[np.ndarray]) -> Optional[np.ndarray]:
        """学習と予測を同時に実行"""
        if len(crops) < 2:
            return None
        
        self.initialize()
        if self.classifier is None:
            return None
        
        try:
            self.classifier.fit(crops)
            self.is_fitted = True
            return self.classifier.predict(crops)
        except Exception as e:
            st.warning(f"チーム分類エラー: {e}")
            return None


class JerseyNumberRecognizer:
    """背番号認識器（TrOCRベース - より安定）"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.model = None
        self.processor = None
        self.initialized = False
    
    def initialize(self):
        """モデルを初期化（遅延ロード）"""
        if self.initialized:
            return True
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            model_id = "microsoft/trocr-base-printed"
            
            with st.spinner("🔢 TrOCRモデルをロード中..."):
                self.processor = TrOCRProcessor.from_pretrained(model_id)
                self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
                
                # デバイス設定（MPSはCPUにフォールバック）
                actual_device = 'cpu' if self.device == 'mps' else self.device
                self.model = self.model.to(actual_device)
                self.model.eval()
            
            self.initialized = True
            return True
        except Exception as e:
            st.warning(f"TrOCR初期化エラー: {e}")
            return False
    
    def recognize(self, crop: np.ndarray) -> Optional[str]:
        """選手画像から背番号を認識"""
        if not self.initialized:
            if not self.initialize():
                return None
        
        try:
            from PIL import Image
            
            # 画像が小さすぎる場合はスキップ
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                return None
            
            # OpenCV BGR -> PIL RGB
            if len(crop.shape) == 3 and crop.shape[2] == 3:
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = crop
            
            pil_image = Image.fromarray(crop_rgb)
            
            # 背番号部分（上半身中央）を切り出し
            w, h = pil_image.size
            # 上半身の中央部分を切り出し
            left = int(w * 0.2)
            right = int(w * 0.8)
            top = int(h * 0.1)
            bottom = int(h * 0.5)
            
            jersey_region = pil_image.crop((left, top, right, bottom))
            
            # グレースケールに変換してコントラスト強調
            jersey_region = jersey_region.convert('L').convert('RGB')
            
            # 入力準備
            pixel_values = self.processor(
                images=jersey_region, 
                return_tensors="pt"
            ).pixel_values
            
            # 推論
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_new_tokens=5)
            
            # デコード
            result = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 数字を抽出（1-99の範囲）
            import re
            numbers = re.findall(r'\b([1-9]|[1-9][0-9])\b', result)
            if numbers:
                return numbers[0]
            
            return None
        except Exception as e:
            return None
    
    def recognize_batch(self, crops: List[np.ndarray], progress_callback=None) -> Dict[int, str]:
        """複数の選手画像から背番号を一括認識"""
        results = {}
        
        for i, crop in enumerate(crops):
            if crop.size > 0:
                number = self.recognize(crop)
                if number:
                    results[i] = number
            
            if progress_callback:
                progress_callback((i + 1) / len(crops))
        
        return results


def get_first_frame(video_path: str) -> Optional[np.ndarray]:
    """動画から最初のフレームを取得"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def draw_points_on_image(frame: np.ndarray, points: List[Tuple]) -> np.ndarray:
    """4点を画像に描画"""
    img = frame.copy()
    
    for i, (x, y) in enumerate(points):
        cv2.circle(img, (int(x), int(y)), 15, POINT_COLORS[i], -1)
        cv2.circle(img, (int(x), int(y)), 15, (255, 255, 255), 3)
        cv2.putText(img, str(i+1), (int(x)-8, int(y)+6), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    if len(points) == 4:
        pts = [(int(x), int(y)) for x, y in points]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i+1) % 4], (255, 255, 255), 2)
    
    return img


def is_point_in_quad(px: float, py: float, points: List[Tuple]) -> bool:
    """点が四角形内にあるかチェック"""
    if len(points) != 4:
        return False
    pts = np.array(points, dtype=np.float32)
    return cv2.pointPolygonTest(pts, (px, py), False) >= 0


def create_court_figure(trajectories: Dict, team_assignments: Dict = None) -> go.Figure:
    """2Dコート図を作成（チーム色対応）"""
    fig = go.Figure()
    
    # コート背景
    fig.add_shape(type="rect", x0=0, y0=0, x1=COURT_WIDTH, y1=COURT_HEIGHT,
                  fillcolor="rgba(34, 139, 34, 0.9)", 
                  line=dict(color="white", width=3))
    
    # フリースローエリア
    fig.add_shape(type="rect", x0=4.5, y0=6, x1=10.5, y1=10,
                  fillcolor="rgba(34, 139, 34, 0.7)", 
                  line=dict(color="white", width=2))
    
    # ゴール
    fig.add_shape(type="circle", x0=6.75, y0=10.0, x1=8.25, y1=11.5,
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
        
        # チーム色を取得
        team = team_assignments.get(player_id, 0) if team_assignments else 0
        team_info = TEAM_COLORS.get(team, TEAM_COLORS[0])
        color = team_info['color']
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=3),
            name=f'{team_info["emoji"]} 選手{player_id}',
            opacity=0.8
        ))
        
        # 開始点
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode='markers',
            marker=dict(size=14, color=color, symbol='circle',
                       line=dict(color='white', width=2)),
            showlegend=False
        ))
        
        # 終了点
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode='markers',
            marker=dict(size=12, color=color, symbol='triangle-up',
                       line=dict(color='white', width=2)),
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text="🏀 選手の軌跡（チーム色分け）", font=dict(size=20, color='white')),
        xaxis=dict(range=[-1, COURT_WIDTH + 1], showgrid=False, 
                   zeroline=False, title="横 (m)", color='white'),
        yaxis=dict(range=[-1, COURT_HEIGHT + 1], showgrid=False, 
                   zeroline=False, title="縦 (m)", scaleanchor="x", color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    return fig


def analyze_video_with_teams(video_path: str, court_points: List[Tuple], 
                             enable_team_classification: bool = True,
                             use_bytetrack: bool = True,
                             enable_jersey_recognition: bool = False,
                             progress_callback=None) -> Optional[Dict]:
    """動画を分析して軌跡とチーム分類を取得"""
    
    # YOLO読み込み
    try:
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt")
    except Exception as e:
        st.error(f"YOLO読み込みエラー: {e}")
        return None
    
    # ViewTransformer設定
    source_points = np.array(court_points, dtype=np.float32)
    target_points = np.array([
        [0, COURT_HEIGHT],
        [COURT_WIDTH, COURT_HEIGHT],
        [COURT_WIDTH, 0],
        [0, 0]
    ], dtype=np.float32)
    
    try:
        transformer = ViewTransformer(source_points, target_points)
    except Exception as e:
        st.error(f"ホモグラフィ計算エラー: {e}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # トラッカー初期化
    bytetrack = None
    simple_tracker = None
    
    if use_bytetrack:
        bytetrack = ByteTrackWrapper()
    else:
        simple_tracker = SimpleTracker()
    
    trajectories = defaultdict(list)
    player_crops = defaultdict(list)
    
    frame_idx = 0
    sample_interval = max(1, total_frames // 30)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 3 == 0:
            # ByteTrack初期化（最初のフレームで）
            if use_bytetrack and not bytetrack.initialized:
                bytetrack.initialize(frame.shape)
            
            results = model(frame, verbose=False)[0]
            
            # 検出結果を収集
            detections_for_track = []
            detection_crops = []
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf > 0.5:
                    xyxy = box.xyxy[0].cpu().numpy()
                    foot_x = (xyxy[0] + xyxy[2]) / 2
                    foot_y = xyxy[3]
                    
                    if is_point_in_quad(foot_x, foot_y, court_points):
                        # ByteTrack用: [x1, y1, x2, y2, conf, cls]
                        detections_for_track.append([*xyxy, conf, cls])
                        
                        # 選手画像を切り出し
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            detection_crops.append(crop)
            
            # トラッキング
            tracked_results = []
            
            if use_bytetrack and bytetrack.initialized and len(detections_for_track) > 0:
                dets_array = np.array(detections_for_track)
                tracks = bytetrack.update(dets_array, frame)
                
                if len(tracks) > 0:
                    for track in tracks:
                        # tracks: [x1, y1, x2, y2, track_id, conf, cls]
                        track_id = int(track[4])
                        tracked_results.append((track_id, track[0], track[1], track[2], track[3]))
            else:
                # フォールバック: SimpleTracker
                if simple_tracker is None:
                    simple_tracker = SimpleTracker()
                simple_dets = [tuple(d[:4]) for d in detections_for_track]
                tracked_results = simple_tracker.update(simple_dets)
            
            # 座標変換と記録
            for i, result in enumerate(tracked_results[:6]):
                track_id = result[0]
                bx1, by1, bx2, by2 = result[1], result[2], result[3], result[4]
                
                foot_x = (bx1 + bx2) / 2
                foot_y = by2
                
                point = np.array([[foot_x, foot_y]], dtype=np.float32)
                court_pos = transformer.transform_points(point)[0]
                court_x, court_y = float(court_pos[0]), float(court_pos[1])
                
                if 0 <= court_x <= COURT_WIDTH and 0 <= court_y <= COURT_HEIGHT:
                    player_id = ((track_id - 1) % 6) + 1
                    
                    # 外れ値フィルタ
                    if trajectories[player_id]:
                        last = trajectories[player_id][-1]
                        dist = np.sqrt((court_x - last[0])**2 + (court_y - last[1])**2)
                        if dist > 3.0:
                            continue
                    
                    trajectories[player_id].append((court_x, court_y))
                    
                    # チーム分類用のサンプル収集
                    if enable_team_classification and frame_idx % sample_interval == 0:
                        if i < len(detection_crops):
                            player_crops[player_id].append(detection_crops[i])
        
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
    
    # チーム分類
    team_assignments = {}
    if enable_team_classification and player_crops:
        with st.spinner("🔴🔵 チーム分類中..."):
            try:
                # 各選手から代表的な画像を選択
                all_crops = []
                player_indices = []
                
                for pid, crops in player_crops.items():
                    if crops:
                        # 中央の画像を選択
                        selected_crop = crops[len(crops) // 2]
                        all_crops.append(selected_crop)
                        player_indices.append(pid)
                
                if len(all_crops) >= 2:
                    classifier = TeamClassifierWrapper(device=get_device())
                    teams = classifier.fit_and_predict(all_crops)
                    
                    if teams is not None:
                        for i, pid in enumerate(player_indices):
                            team_assignments[pid] = int(teams[i])
            except Exception as e:
                st.warning(f"チーム分類スキップ: {e}")
    
    # 背番号認識
    jersey_numbers = {}
    if enable_jersey_recognition and player_crops:
        with st.spinner("🔢 背番号認識中..."):
            try:
                recognizer = JerseyNumberRecognizer(device=get_device())
                
                for pid, crops in player_crops.items():
                    if crops:
                        # 複数の画像から認識を試みる
                        for crop in crops[:3]:  # 最大3枚試行
                            number = recognizer.recognize(crop)
                            if number:
                                jersey_numbers[pid] = number
                                break
                
                if jersey_numbers:
                    st.success(f"✅ {len(jersey_numbers)}人の背番号を認識")
            except Exception as e:
                st.warning(f"背番号認識スキップ: {e}")
    
    return {
        'trajectories': smoothed,
        'team_assignments': team_assignments,
        'jersey_numbers': jersey_numbers,
        'duration': total_frames / fps if fps > 0 else 0,
        'fps': fps,
        'total_frames': total_frames
    }


def main():
    st.title("🏀 3x3 Analyzer v2")
    st.caption("選手検出 → 追跡 → チーム分類 → 2D軌跡表示")
    
    # セッション初期化
    for key in ['frame', 'video_path', 'result', 'points', 'click_count']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'points' else []
            if key == 'click_count':
                st.session_state[key] = 0
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        st.subheader("🎯 トラッキング")
        use_bytetrack = st.checkbox("ByteTrack使用", value=True, 
                                    help="高精度なマルチオブジェクトトラッカー（遮蔽に強い）")
        
        st.subheader("🔴🔵 チーム分析")
        enable_team = st.checkbox("チーム分類を有効化", value=True, 
                                  help="SigLIPでユニフォーム色からチームを自動分類")
        
        st.subheader("🔢 背番号認識")
        enable_jersey = st.checkbox("背番号認識を有効化", value=False, 
                                    help="SmolVLM2で背番号を読み取り（処理時間増加）")
        
        device = get_device()
        st.info(f"🖥️ デバイス: **{device.upper()}**")
        
        tracker_name = "ByteTrack" if use_bytetrack else "IoUトラッカー"
        jersey_status = "✅ ON" if enable_jersey else "❌ OFF"
        st.markdown(f"""
        ### 技術スタック
        - 🔍 YOLO11n（検出）
        - 🎯 **{tracker_name}**（追跡）
        - 🔴🔵 SigLIP + K-means（チーム分類）
        - 🔢 SmolVLM2（背番号）{jersey_status}
        - 📐 ViewTransformer（座標変換）
        """)
    
    # ========== STEP 1 ==========
    st.header("STEP 1: 動画を選択")
    
    video_path = st.text_input("動画ファイルのパス", placeholder="/path/to/video.mp4")
    
    if video_path and os.path.exists(video_path):
        st.success("✅ ファイル確認OK")
        
        if st.button("📷 フレームを取得", use_container_width=True):
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
        st.info("👆 動画パスを入力して「フレームを取得」をクリック")
        return
    
    # ========== STEP 2 ==========
    st.header("STEP 2: コートの4隅を指定")
    
    frame = st.session_state.frame
    h, w = frame.shape[:2]
    current_point = len(st.session_state.points)
    
    if current_point < 4:
        st.markdown(f"### 👆 次にクリック: **{POINT_NAMES[current_point]}**")
    else:
        st.success("✅ 4点すべて設定完了！")
    
    display_frame = draw_points_on_image(frame, st.session_state.points)
    
    from streamlit_image_coordinates import streamlit_image_coordinates
    from PIL import Image
    
    display_width = min(900, w)
    scale = display_width / w
    display_height = int(h * scale)
    
    pil_image = Image.fromarray(display_frame)
    pil_image = pil_image.resize((display_width, display_height))
    
    coords = streamlit_image_coordinates(
        pil_image,
        key=f"click_{st.session_state.click_count}"
    )
    
    if coords is not None and len(st.session_state.points) < 4:
        original_x = coords["x"] / scale
        original_y = coords["y"] / scale
        st.session_state.points.append((original_x, original_y))
        st.session_state.click_count += 1
        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("↩️ 1つ戻す", use_container_width=True, 
                     disabled=len(st.session_state.points) == 0):
            st.session_state.points = st.session_state.points[:-1]
            st.session_state.click_count += 1
            st.rerun()
    with col2:
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.points = []
            st.session_state.click_count += 1
            st.rerun()
    
    # ========== STEP 3 ==========
    st.header("STEP 3: 分析実行")
    
    if len(st.session_state.points) == 4:
        if st.button("🚀 分析開始", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            def update_progress(p):
                progress.progress(min(p, 1.0))
                status.text(f"分析中... {int(p*100)}%")
            
            result = analyze_video_with_teams(
                st.session_state.video_path,
                st.session_state.points,
                enable_team_classification=enable_team,
                use_bytetrack=use_bytetrack,
                enable_jersey_recognition=enable_jersey,
                progress_callback=update_progress
            )
            
            if result:
                st.session_state.result = result
                status.text("✅ 分析完了！")
                st.rerun()
    else:
        st.warning(f"⚠️ 4点を指定してください（現在: {len(st.session_state.points)}点）")
    
    # ========== STEP 4 ==========
    if st.session_state.result:
        st.header("STEP 4: 結果")
        
        result = st.session_state.result
        
        # 2Dコート図
        fig = create_court_figure(result['trajectories'], result.get('team_assignments'))
        st.plotly_chart(fig, use_container_width=True)
        
        # チーム分類結果
        if result.get('team_assignments'):
            st.subheader("🔴🔵 チーム分類結果")
            team_a = [pid for pid, team in result['team_assignments'].items() if team == 0]
            team_b = [pid for pid, team in result['team_assignments'].items() if team == 1]
            
            jersey_numbers = result.get('jersey_numbers', {})
            
            def format_player(pid):
                """選手IDと背番号をフォーマット"""
                if pid in jersey_numbers:
                    return f"選手{pid} (#{jersey_numbers[pid]})"
                return f"選手{pid}"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### 🔴 チームA")
                players_a = [format_player(pid) for pid in sorted(team_a)]
                st.write(", ".join(players_a))
            with col2:
                st.markdown(f"### 🔵 チームB")
                players_b = [format_player(pid) for pid in sorted(team_b)]
                st.write(", ".join(players_b))
        
        # 背番号認識結果
        if result.get('jersey_numbers'):
            st.subheader("🔢 背番号認識結果")
            jersey_data = []
            for pid, number in sorted(result['jersey_numbers'].items()):
                team = result.get('team_assignments', {}).get(pid, -1)
                team_emoji = "🔴" if team == 0 else "🔵" if team == 1 else "⚪"
                jersey_data.append({"チーム": team_emoji, "選手ID": pid, "背番号": f"#{number}"})
            
            st.table(jersey_data)
        
        # 統計
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 動画情報")
            st.metric("時間", f"{result['duration']:.1f} 秒")
            st.metric("FPS", f"{result['fps']:.1f}")
        
        with col2:
            st.subheader("🏃 選手の移動距離")
            team_assignments = result.get('team_assignments', {})
            for player_id, pts in sorted(result['trajectories'].items()):
                if len(pts) > 1:
                    dist = sum(
                        np.sqrt((pts[i][0] - pts[i-1][0])**2 + 
                               (pts[i][1] - pts[i-1][1])**2)
                        for i in range(1, len(pts))
                    )
                    team = team_assignments.get(player_id, 0)
                    emoji = TEAM_COLORS[team]['emoji']
                    st.metric(f"{emoji} 選手{player_id}", f"{dist:.1f} m")
        
        # ダウンロード
        st.subheader("📥 データダウンロード")
        
        export_data = {
            'trajectories': {str(k): [(float(p[0]), float(p[1])) for p in v] 
                            for k, v in result['trajectories'].items()},
            'team_assignments': {str(k): v for k, v in result.get('team_assignments', {}).items()},
            'court_points': [(float(p[0]), float(p[1])) for p in st.session_state.points],
            'duration': float(result['duration']),
            'fps': float(result['fps'])
        }
        
        st.download_button(
            "📥 軌跡データ (JSON)",
            data=json.dumps(export_data, indent=2, ensure_ascii=False),
            file_name="trajectory_with_teams.json",
            mime="application/json"
        )
        
        if st.button("🔄 新しい動画を分析", use_container_width=True):
            for key in ['frame', 'video_path', 'result', 'points']:
                st.session_state[key] = None if key != 'points' else []
            st.session_state.click_count = 0
            st.rerun()


if __name__ == "__main__":
    main()
