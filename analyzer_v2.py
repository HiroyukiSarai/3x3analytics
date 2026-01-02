"""
3x3バスケットボール分析システム v2 - Roboflow完全統合版
- RF-DETR / バスケ専用モデルで検出
- SAM2 / ByteTrackでトラッキング
- コートキーポイント自動検出（オプション）
- チーム分類（SigLIP）
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
    .stAlert { background-color: rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

# 定数
COURT_WIDTH = 15.0
COURT_HEIGHT = 11.0
TEAM_COLORS = {
    0: {'name': 'チームA', 'color': '#ff6b6b', 'emoji': '🔴'},
    1: {'name': 'チームB', 'color': '#4dabf7', 'emoji': '🔵'}
}


class RoboflowDetector:
    """Roboflow Inferenceを使用したバスケ専用検出器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.player_model = None
        self.court_model = None
        self.initialized = False
        self.use_roboflow = False
    
    def initialize(self):
        """モデルを初期化"""
        if self.initialized:
            return True
        
        # まずAPIキーがある場合はRoboflowを試す
        if self.api_key:
            clean_key = self.api_key.strip()
            st.info(f"🔑 APIキー: {clean_key[:4]}...{clean_key[-4:]}")
            
            # HTTP API方式を使用（最も信頼性が高い）
            self.api_key = clean_key
            self.use_roboflow = True
            self.use_http_api = True
            self.initialized = True
            st.success("✅ Roboflow HTTP API モードで接続")
            return True
        
        # フォールバック: YOLO
        try:
            from ultralytics import YOLO
            self.player_model = YOLO("yolo11n.pt")
            self.use_roboflow = False
            self.initialized = True
            st.info("ℹ️ YOLO11nを使用中")
            return True
        except Exception as e:
            st.error(f"モデル初期化エラー: {e}")
            return False
    
    def detect_players(self, frame: np.ndarray, confidence: float = 0.5) -> List[Dict]:
        """選手を検出"""
        if not self.initialized:
            if not self.initialize():
                return []
        
        detections = []
        
        try:
            if self.use_roboflow and hasattr(self, 'use_http_api') and self.use_http_api:
                # HTTP API方式
                import base64
                import requests
                
                # フレームをbase64エンコード
                _, buffer = cv2.imencode('.jpg', frame)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Roboflow Hosted API呼び出し（バージョン25 = 最新）
                url = f"https://detect.roboflow.com/basketball-players-fy4c2/25?api_key={self.api_key}&confidence={int(confidence*100)}"
                response = requests.post(url, data=img_base64, headers={'Content-Type': 'application/x-www-form-urlencoded'})
                
                if response.status_code == 200:
                    results = response.json()
                    preds = results.get('predictions', [])
                    
                    # デバッグ: 最初の数フレームで検出数を表示
                    if not hasattr(self, '_debug_count'):
                        self._debug_count = 0
                    if self._debug_count < 3:
                        st.write(f"🔍 API応答: {len(preds)}件検出, クラス: {[p.get('class') for p in preds[:5]]}")
                        self._debug_count += 1
                    
                    for pred in preds:
                        x, y = pred['x'], pred['y']
                        w, h = pred['width'], pred['height']
                        cls_name = pred.get('class', 'player')
                        conf = pred.get('confidence', 0.5)
                        
                        # すべてのクラスを受け入れる（フィルタを緩める）
                        detections.append({
                            'bbox': [x - w/2, y - h/2, x + w/2, y + h/2],
                            'confidence': conf,
                            'class': cls_name
                        })
                else:
                    # エラー時のデバッグ
                    if not hasattr(self, '_error_shown'):
                        st.error(f"❌ API エラー: {response.status_code} - {response.text[:200]}")
                        self._error_shown = True
            elif self.use_roboflow and self.player_model is not None:
                # SDK方式
                results = self.player_model.infer(frame, confidence=confidence)
                for pred in results[0].predictions:
                    if pred.class_name.lower() in ['player', 'person']:
                        x, y, w, h = pred.x, pred.y, pred.width, pred.height
                        detections.append({
                            'bbox': [x - w/2, y - h/2, x + w/2, y + h/2],
                            'confidence': pred.confidence,
                            'class': pred.class_name
                        })
            else:
                # YOLO
                if self.player_model is not None:
                    results = self.player_model(frame, verbose=False)[0]
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls == 0 and conf > confidence:  # person class
                            xyxy = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': xyxy.tolist(),
                                'confidence': conf,
                                'class': 'person'
                            })
        except Exception as e:
            pass  # エラーは無視して空リストを返す
        
        return detections


class CourtKeypointDetector:
    """コートキーポイント自動検出（Roboflow Keypoint Detection使用）"""
    
    # 学習済みモデル情報
    MODEL_ID = "3x3-court-detection/2"
    API_URL = "https://serverless.roboflow.com"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.initialized = False
        self.client = None
    
    def initialize(self):
        if self.api_key:
            try:
                from inference_sdk import InferenceHTTPClient
                self.client = InferenceHTTPClient(
                    api_url=self.API_URL,
                    api_key=self.api_key
                )
                self.initialized = True
                return True
            except ImportError:
                st.warning("inference-sdk未インストール。pip install inference-sdk を実行してください")
                return False
        return False
    
    def detect_court(self, frame: np.ndarray) -> Optional[Dict]:
        """コートのキーポイントを検出"""
        if not self.api_key:
            return None
        
        # inference-sdkを使用
        if self.client is None:
            if not self.initialize():
                # フォールバック: HTTP API直接呼び出し
                return self._detect_court_http(frame)
        
        try:
            # 一時ファイルに保存してinferenceを実行
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                result = self.client.infer(tmp.name, model_id=self.MODEL_ID)
                os.unlink(tmp.name)
            
            return self._parse_keypoint_result(result)
        except Exception as e:
            st.warning(f"SDK検出エラー: {e}、HTTP APIにフォールバック")
            return self._detect_court_http(frame)
    
    def _detect_court_http(self, frame: np.ndarray) -> Optional[Dict]:
        """HTTP API直接呼び出しによるフォールバック"""
        try:
            import base64
            import requests
            
            # フレームをbase64エンコード
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Roboflow Keypoint Detection API
            url = f"https://detect.roboflow.com/{self.MODEL_ID}?api_key={self.api_key}&confidence=30"
            response = requests.post(url, data=img_base64, headers={'Content-Type': 'application/x-www-form-urlencoded'})
            
            if response.status_code == 200:
                return self._parse_keypoint_result(response.json())
            else:
                st.error(f"API エラー: {response.status_code}")
        except Exception as e:
            st.warning(f"コート検出エラー: {e}")
        
        return None
    
    def _parse_keypoint_result(self, result: Dict) -> Optional[Dict]:
        """キーポイント検出結果をパース"""
        keypoints = {}
        
        # キーポイント検出モデルのレスポンス形式に対応
        predictions = result.get('predictions', [])
        
        for pred in predictions:
            # キーポイントが含まれる場合
            if 'keypoints' in pred:
                for kp in pred['keypoints']:
                    kp_name = kp.get('class_name', kp.get('class', f"kp_{len(keypoints)}"))
                    x, y = kp.get('x', 0), kp.get('y', 0)
                    confidence = kp.get('confidence', 0)
                    if confidence > 0.3:  # 信頼度フィルタ
                        keypoints[kp_name] = (x, y)
            else:
                # 通常の検出結果形式
                class_name = pred.get('class', '')
                x, y = pred.get('x', 0), pred.get('y', 0)
                keypoints[class_name] = (x, y)
        
        return keypoints if keypoints else None
    
    def get_court_corners(self, keypoints: Dict) -> Optional[List[Tuple]]:
        """キーポイントから4角を抽出"""
        if not keypoints:
            return None
        
        # 3x3コートの主要なキーポイントを探す
        # 期待されるキーポイント名（モデルによって異なる可能性あり）
        corner_names = [
            # ゴール側
            ['baseline-left', 'baseline_left', 'corner-baseline-left', '0'],
            ['baseline-right', 'baseline_right', 'corner-baseline-right', '1'],
            # トップ側（3ポイントライン付近）
            ['sideline-right', 'sideline_right', 'corner-sideline-right', '2'],
            ['sideline-left', 'sideline_left', 'corner-sideline-left', '3'],
        ]
        
        corners = []
        for name_options in corner_names:
            for name in name_options:
                if name in keypoints:
                    corners.append(keypoints[name])
                    break
        
        if len(corners) == 4:
            return corners
        
        # 見つからない場合は、検出されたすべてのポイントを返す
        all_points = list(keypoints.values())
        if len(all_points) >= 4:
            # Y座標でソートして上2つと下2つを取得
            sorted_by_y = sorted(all_points, key=lambda p: p[1])
            top_points = sorted(sorted_by_y[:2], key=lambda p: p[0])  # 上の2点をX順
            bottom_points = sorted(sorted_by_y[-2:], key=lambda p: p[0])  # 下の2点をX順
            return [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]
        
        return None


class ImprovedTracker:
    """改善されたトラッカー（ByteTrack + フィルタリング）"""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_age = 30
        self.iou_threshold = 0.3
    
    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[Dict]:
        """検出結果を更新してトラックIDを付与"""
        results = []
        matched_tracks = set()
        matched_dets = set()
        
        det_bboxes = [d['bbox'] for d in detections]
        
        # マッチング
        for di, det in enumerate(detections):
            best_id, best_iou = None, self.iou_threshold
            for tid, t in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(det['bbox'], t['bbox'])
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            
            if best_id:
                matched_tracks.add(best_id)
                matched_dets.add(di)
                self.tracks[best_id] = {'bbox': det['bbox'], 'age': 0}
                results.append({**det, 'track_id': best_id})
        
        # 新規トラック
        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bbox': det['bbox'], 'age': 0}
                results.append({**det, 'track_id': tid})
        
        # 古いトラックを削除
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


class TeamClassifierWrapper:
    """チーム分類器"""
    
    def __init__(self):
        self.classifier = None
    
    def classify(self, crops: List[np.ndarray]) -> Optional[List[int]]:
        if len(crops) < 2:
            return None
        
        try:
            from sports.common.team import TeamClassifier
            if self.classifier is None:
                self.classifier = TeamClassifier(device='cpu', batch_size=16)
            
            self.classifier.fit(crops)
            return self.classifier.predict(crops).tolist()
        except Exception as e:
            st.warning(f"チーム分類エラー: {e}")
            return None


def get_first_frame(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def is_point_in_quad(x, y, quad):
    """点が四角形内にあるか判定"""
    if len(quad) != 4:
        return True
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    signs = []
    for i in range(4):
        o = quad[i]
        a = quad[(i + 1) % 4]
        signs.append(cross(o, a, (x, y)) > 0)
    
    return all(signs) or not any(signs)


def analyze_video(video_path: str, court_points: List[Tuple], 
                  detector: RoboflowDetector,
                  enable_team: bool = True,
                  progress_callback=None) -> Optional[Dict]:
    """動画を分析"""
    
    # ViewTransformer設定
    # クリック順序: ①ゴール左 ②ゴール右 ③トップ右 ④トップ左
    # ゴール = Y=0（コート下側）、トップ = Y=11（コート上側）
    source_points = np.array(court_points, dtype=np.float32)
    target_points = np.array([
        [0, 0],                      # ① ゴール左 → 左下
        [COURT_WIDTH, 0],            # ② ゴール右 → 右下
        [COURT_WIDTH, COURT_HEIGHT], # ③ トップ右 → 右上
        [0, COURT_HEIGHT]            # ④ トップ左 → 左上
    ], dtype=np.float32)
    
    try:
        transformer = ViewTransformer(source_points, target_points)
    except Exception as e:
        st.error(f"ホモグラフィ計算エラー: {e}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tracker = ImprovedTracker()
    trajectories = defaultdict(list)
    player_crops = defaultdict(list)
    
    frame_idx = 0
    sample_interval = max(1, total_frames // 30)
    
    # デバッグ用カウンター
    debug_total_detections = 0
    debug_court_detections = 0
    debug_frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 3 == 0:
            # 検出
            detections = detector.detect_players(frame, confidence=0.5)
            debug_total_detections += len(detections)
            debug_frame_count += 1
            
            # 最初のフレームでデバッグ情報を表示
            if frame_idx == 0 and detections:
                st.info(f"🔍 フレーム0: {len(detections)}人検出、コート範囲: X={min(p[0] for p in court_points):.0f}-{max(p[0] for p in court_points):.0f}, Y={min(p[1] for p in court_points):.0f}-{max(p[1] for p in court_points):.0f}")
                for i, det in enumerate(detections[:3]):
                    bbox = det['bbox']
                    foot_x = (bbox[0] + bbox[2]) / 2
                    foot_y = bbox[3]
                    in_court = is_point_in_quad(foot_x, foot_y, court_points)
                    st.write(f"  選手{i+1}: 足位置=({foot_x:.0f}, {foot_y:.0f}), コート内={in_court}")
            
            # コート内のみフィルタ
            court_detections = []
            for det in detections:
                bbox = det['bbox']
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]
                if is_point_in_quad(foot_x, foot_y, court_points):
                    court_detections.append(det)
            
            debug_court_detections += len(court_detections)
            
            # トラッキング
            tracked = tracker.update(court_detections, frame)
            
            # 座標変換と記録
            for det in tracked[:6]:
                bbox = det['bbox']
                track_id = det['track_id']
                
                foot_x = (bbox[0] + bbox[2]) / 2
                foot_y = bbox[3]
                
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
                    
                    # チーム分類用サンプル
                    if enable_team and frame_idx % sample_interval == 0:
                        x1, y1, x2, y2 = map(int, bbox)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            player_crops[player_id].append(crop)
        
        if progress_callback:
            progress_callback(frame_idx / total_frames)
        
        frame_idx += 1
    
    cap.release()
    
    # デバッグサマリー
    st.info(f"""
    📊 **検出サマリー**
    - 処理フレーム数: {debug_frame_count}
    - 総検出数: {debug_total_detections}（平均: {debug_total_detections/max(1,debug_frame_count):.1f}/フレーム）
    - コート内検出数: {debug_court_detections}（平均: {debug_court_detections/max(1,debug_frame_count):.1f}/フレーム）
    - 追跡選手数: {len(trajectories)}
    """)
    
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
    if enable_team and player_crops:
        classifier = TeamClassifierWrapper()
        all_crops = []
        player_indices = []
        
        for pid, crops in player_crops.items():
            if crops:
                all_crops.append(crops[len(crops) // 2])
                player_indices.append(pid)
        
        if len(all_crops) >= 2:
            teams = classifier.classify(all_crops)
            if teams:
                for i, pid in enumerate(player_indices):
                    team_assignments[pid] = teams[i]
    
    return {
        'trajectories': smoothed,
        'team_assignments': team_assignments,
        'duration': total_frames / fps if fps > 0 else 0,
        'fps': fps,
        'total_frames': total_frames
    }


def create_court_figure(trajectories: Dict, team_assignments: Dict = None):
    """2Dコート図を作成"""
    fig = go.Figure()
    
    # コート描画
    fig.add_shape(type="rect", x0=0, y0=0, x1=COURT_WIDTH, y1=COURT_HEIGHT,
                  line=dict(color="white", width=2), fillcolor="rgba(50,50,50,0.5)")
    
    # 3ポイントライン
    theta = np.linspace(-np.pi/2, np.pi/2, 50)
    arc_x = COURT_WIDTH/2 + 6.75 * np.cos(theta)
    arc_y = 6.75 * np.sin(theta)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines',
                             line=dict(color='white', width=1, dash='dash'),
                             showlegend=False))
    
    # ゴール
    fig.add_trace(go.Scatter(x=[COURT_WIDTH/2], y=[0], mode='markers',
                             marker=dict(size=20, color='orange', symbol='circle'),
                             name='ゴール'))
    
    # 軌跡
    colors = ['#ff6b6b', '#4dabf7', '#51cf66', '#fcc419', '#cc5de8', '#20c997']
    
    for pid, points in trajectories.items():
        if len(points) < 2:
            continue
        
        team = team_assignments.get(pid, 0) if team_assignments else 0
        color = TEAM_COLORS[team]['color']
        emoji = TEAM_COLORS[team]['emoji']
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            name=f'{emoji} 選手{pid}'
        ))
        
        # 終点
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode='markers+text',
            marker=dict(size=15, color=color),
            text=[str(pid)], textposition='middle center',
            textfont=dict(color='white', size=10),
            showlegend=False
        ))
    
    fig.update_layout(
        title="🏀 2D軌跡表示",
        xaxis=dict(range=[-1, COURT_WIDTH+1], title="X (m)", scaleanchor="y"),
        yaxis=dict(range=[-1, COURT_HEIGHT+1], title="Y (m)"),
        height=600,
        template="plotly_dark",
        legend=dict(x=1.02, y=1)
    )
    
    return fig


def main():
    st.title("🏀 3x3 Analyzer v2")
    st.caption("Roboflow統合版 - バスケ専用モデルで高精度分析")
    
    # セッション初期化
    for key in ['frame', 'video_path', 'result', 'points', 'detector', 'last_click']:
        if key not in st.session_state:
            if key == 'points':
                st.session_state[key] = []
            elif key == 'last_click':
                st.session_state[key] = None
            else:
                st.session_state[key] = None
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        st.subheader("🔑 Roboflow API")
        api_key = st.text_input("APIキー（オプション）", type="password",
                                help="universe.roboflow.comで無料取得")
        
        if api_key:
            st.success("✅ APIキー設定済み")
        else:
            st.info("💡 APIキーなしでも動作（YOLO使用）")
        
        st.subheader("🎯 検出設定")
        confidence = st.slider("検出閾値", 0.3, 0.9, 0.5)
        
        st.subheader("🔴🔵 チーム分析")
        enable_team = st.checkbox("チーム分類を有効化", value=True)
        
        model_name = "Roboflow バスケ専用" if api_key else "YOLO11n"
        st.markdown(f"""
        ### 技術スタック
        - 🔍 **{model_name}**（検出）
        - 🎯 IoUトラッカー（追跡）
        - 🔴🔵 SigLIP + K-means（チーム分類）
        - 📐 ViewTransformer（座標変換）
        """)
    
    # 動画入力
    st.header("📹 動画入力")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("動画をアップロード", type=['mp4', 'mov', 'avi'])
    with col2:
        video_path_input = st.text_input("または動画パスを入力")
    
    video_path = None
    if uploaded:
        temp_path = f"/tmp/{uploaded.name}"
        with open(temp_path, 'wb') as f:
            f.write(uploaded.read())
        video_path = temp_path
    elif video_path_input and os.path.exists(video_path_input):
        video_path = video_path_input
    
    if video_path and video_path != st.session_state.video_path:
        st.session_state.video_path = video_path
        st.session_state.frame = get_first_frame(video_path)
        st.session_state.points = []
        st.session_state.result = None
    
    # コート座標設定
    if st.session_state.frame is not None:
        st.header("📐 コート座標設定")
        
        # 自動検出ボタン
        col_auto, col_manual = st.columns(2)
        with col_auto:
            if st.button("🤖 コート自動検出", type="primary", help="Roboflow AIでコートを自動検出"):
                if api_key:
                    court_detector = CourtKeypointDetector(api_key)
                    with st.spinner("コートを検出中..."):
                        keypoints = court_detector.detect_court(st.session_state.frame)
                        if keypoints:
                            st.success(f"✅ {len(keypoints)}個のキーポイントを検出")
                            st.json(keypoints)
                            
                            corners = court_detector.get_court_corners(keypoints)
                            if corners:
                                st.session_state.points = list(corners)
                                st.success("✅ コート4角を自動設定しました！")
                                st.rerun()
                            else:
                                st.warning("4角の抽出に失敗。手動で設定してください。")
                        else:
                            st.warning("コート検出に失敗。手動で設定してください。")
                else:
                    st.warning("⚠️ APIキーを入力してください")
        
        with col_manual:
            st.markdown("**または手動で設定：**")
        
        st.markdown("""
        **手動設定の場合、4角を以下の順番でクリック:**
        
        1️⃣ **ゴール左** = ゴール下の左角（画面奥・左側）  
        2️⃣ **ゴール右** = ゴール下の右角（画面奥・右側）  
        3️⃣ **トップ右** = トップの右角（画面手前・右側）  
        4️⃣ **トップ左** = トップの左角（画面手前・左側）
        """)
        
        from streamlit_image_coordinates import streamlit_image_coordinates
        
        display_frame = st.session_state.frame.copy()
        h, w = display_frame.shape[:2]
        scale = 800 / w
        display_frame = cv2.resize(display_frame, (800, int(h * scale)))
        
        # ポイント描画
        for i, (px, py) in enumerate(st.session_state.points):
            sx, sy = int(px * scale), int(py * scale)
            color = [(255,0,0), (255,165,0), (0,255,0), (0,255,255)][i]
            cv2.circle(display_frame, (sx, sy), 10, color, -1)
            cv2.putText(display_frame, str(i+1), (sx-5, sy+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # 四角形描画
        if len(st.session_state.points) == 4:
            pts = np.array([[int(p[0]*scale), int(p[1]*scale)] 
                           for p in st.session_state.points], np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)
        
        display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        coords = streamlit_image_coordinates(display_rgb, key="court_click")
        
        if coords and len(st.session_state.points) < 4:
            # 重複クリック防止
            current_click = (coords['x'], coords['y'])
            if st.session_state.last_click != current_click:
                st.session_state.last_click = current_click
                orig_x = coords['x'] / scale
                orig_y = coords['y'] / scale
                st.session_state.points.append((orig_x, orig_y))
                st.rerun()
        
        # ポイント表示
        col1, col2 = st.columns([3, 1])
        with col2:
            st.write("**設定済みポイント:**")
            labels = ['① ゴール左', '② ゴール右', '③ トップ右', '④ トップ左']
            for i, (px, py) in enumerate(st.session_state.points):
                st.write(f"{labels[i]}: ({px:.0f}, {py:.0f})")
            
            if st.button("🔄 リセット"):
                st.session_state.points = []
                st.session_state.last_click = None
                st.rerun()
    
    # 分析実行
    st.header("🎬 分析")
    
    if len(st.session_state.points) == 4:
        if st.button("🚀 分析開始", type="primary"):
            # 検出器初期化
            detector = RoboflowDetector(api_key=api_key if api_key else None)
            detector.initialize()
            
            progress = st.progress(0)
            status = st.empty()
            
            def update_progress(p):
                progress.progress(min(p, 1.0))
                status.text(f"分析中... {int(p*100)}%")
            
            result = analyze_video(
                st.session_state.video_path,
                st.session_state.points,
                detector,
                enable_team=enable_team,
                progress_callback=update_progress
            )
            
            if result:
                st.session_state.result = result
                status.text("✅ 分析完了！")
                st.rerun()
    else:
        st.warning(f"⚠️ 4点を指定してください（現在: {len(st.session_state.points)}点）")
    
    # 結果表示
    if st.session_state.result:
        result = st.session_state.result
        
        st.header("📊 分析結果")
        
        # 2D表示
        fig = create_court_figure(result['trajectories'], result.get('team_assignments'))
        st.plotly_chart(fig, use_container_width=True)
        
        # チーム分類
        if result.get('team_assignments'):
            st.subheader("🔴🔵 チーム分類結果")
            team_a = [pid for pid, team in result['team_assignments'].items() if team == 0]
            team_b = [pid for pid, team in result['team_assignments'].items() if team == 1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔴 チームA")
                st.write(f"選手: {', '.join(map(str, sorted(team_a)))}")
            with col2:
                st.markdown("### 🔵 チームB")
                st.write(f"選手: {', '.join(map(str, sorted(team_b)))}")
        
        # 統計
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 動画情報")
            st.write(f"- 時間: {result['duration']:.1f}秒")
            st.write(f"- FPS: {result['fps']:.1f}")
            st.write(f"- 検出選手数: {len(result['trajectories'])}人")
        
        with col2:
            st.subheader("📈 移動距離")
            for pid, pts in sorted(result['trajectories'].items()):
                if len(pts) >= 2:
                    dist = sum(np.sqrt((pts[i+1][0]-pts[i][0])**2 + 
                              (pts[i+1][1]-pts[i][1])**2) 
                              for i in range(len(pts)-1))
                    team = result.get('team_assignments', {}).get(pid, 0)
                    emoji = TEAM_COLORS[team]['emoji']
                    st.write(f"{emoji} 選手{pid}: {dist:.1f}m")
        
        # エクスポート
        st.subheader("💾 データエクスポート")
        export_data = {
            'trajectories': {str(k): v for k, v in result['trajectories'].items()},
            'team_assignments': {str(k): v for k, v in result.get('team_assignments', {}).items()},
            'court_points': st.session_state.points,
            'duration': result['duration'],
            'fps': result['fps']
        }
        
        st.download_button(
            "📥 JSONダウンロード",
            data=json.dumps(export_data, indent=2),
            file_name="trajectory_with_teams.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()

