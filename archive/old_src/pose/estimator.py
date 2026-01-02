"""
骨格推定モジュール
RTMPose/YOLO-poseを使用した選手の骨格推定
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import math


@dataclass
class Skeleton:
    """
    骨格情報を格納するデータクラス
    
    COCOフォーマット (17キーポイント):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    
    # キーポイント座標 [x, y, confidence]
    keypoints: np.ndarray = field(default_factory=lambda: np.zeros((17, 3)))
    track_id: Optional[int] = None
    bbox: Optional[np.ndarray] = None
    
    # キーポイント名
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # 骨格の接続（描画用）
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 顔
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上半身
        (5, 11), (6, 12), (11, 12),  # 胴体
        (11, 13), (13, 15), (12, 14), (14, 16)  # 下半身
    ]
    
    def get_keypoint(self, name: str) -> Optional[Tuple[float, float, float]]:
        """キーポイント名から座標を取得"""
        if name in self.KEYPOINT_NAMES:
            idx = self.KEYPOINT_NAMES.index(name)
            return tuple(self.keypoints[idx])
        return None
    
    @property
    def body_orientation(self) -> float:
        """
        体の向きを計算（度）
        肩の向きから推定
        
        Returns:
            向き角度（0-360度、0=右向き、90=前向き）
        """
        left_shoulder = self.keypoints[5]
        right_shoulder = self.keypoints[6]
        
        if left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
            return -1  # 信頼度が低い
        
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        # 0-360度に正規化
        if angle < 0:
            angle += 360
        
        return angle
    
    @property
    def is_facing_camera(self) -> bool:
        """カメラに向いているかどうか"""
        # 鼻と両肩の位置関係で判定
        nose = self.keypoints[0]
        left_shoulder = self.keypoints[5]
        right_shoulder = self.keypoints[6]
        
        if nose[2] < 0.3 or left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3:
            return False
        
        # 鼻が両肩の間にあるか
        min_x = min(left_shoulder[0], right_shoulder[0])
        max_x = max(left_shoulder[0], right_shoulder[0])
        
        return min_x <= nose[0] <= max_x
    
    @property
    def center_of_mass(self) -> Tuple[float, float]:
        """重心位置を推定"""
        # 主要な関節点から重心を推定
        major_points = [5, 6, 11, 12]  # 両肩、両腰
        
        valid_points = []
        for idx in major_points:
            if self.keypoints[idx][2] > 0.3:
                valid_points.append(self.keypoints[idx][:2])
        
        if not valid_points:
            return (0, 0)
        
        points = np.array(valid_points)
        return tuple(points.mean(axis=0))
    
    def is_shooting_pose(self) -> bool:
        """
        シュートポーズの検出（簡易版）
        手が顔より上にあるかで判定
        """
        left_wrist = self.keypoints[9]
        right_wrist = self.keypoints[10]
        nose = self.keypoints[0]
        
        if nose[2] < 0.3:
            return False
        
        # いずれかの手首が鼻より上
        left_up = left_wrist[2] > 0.3 and left_wrist[1] < nose[1]
        right_up = right_wrist[2] > 0.3 and right_wrist[1] < nose[1]
        
        return left_up or right_up


class PoseEstimator:
    """
    骨格推定器
    YOLO-pose または RTMPose を使用
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.5
    ):
        """
        Args:
            model_path: 骨格推定モデルのパス
            device: 推論デバイス
            conf_threshold: 検出信頼度閾値
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.use_yolo = True
        except ImportError:
            print("Warning: ultralytics not installed")
            self.model = None
            self.use_yolo = False
        
        self.device = device
        self.conf_threshold = conf_threshold
    
    def estimate(self, frame: np.ndarray) -> List[Skeleton]:
        """
        フレームから全選手の骨格を推定
        
        Args:
            frame: BGR画像
            
        Returns:
            List[Skeleton]: 検出された骨格のリスト
        """
        if self.model is None:
            return []
        
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        skeletons = []
        
        if results.keypoints is not None:
            for i, kpts in enumerate(results.keypoints.data):
                keypoints = kpts.cpu().numpy()
                
                # バウンディングボックス
                bbox = None
                if results.boxes is not None and i < len(results.boxes):
                    bbox = results.boxes[i].xyxy[0].cpu().numpy()
                
                skeleton = Skeleton(
                    keypoints=keypoints,
                    bbox=bbox
                )
                skeletons.append(skeleton)
        
        return skeletons
    
    def estimate_for_bbox(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        track_id: Optional[int] = None
    ) -> Optional[Skeleton]:
        """
        指定されたバウンディングボックス内の骨格を推定
        
        Args:
            frame: BGR画像
            bbox: [x1, y1, x2, y2]
            track_id: トラックID
            
        Returns:
            Skeleton or None
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        skeletons = self.estimate(crop)
        
        if not skeletons:
            return None
        
        # 最も大きい骨格を選択
        best = max(skeletons, key=lambda s: s.bbox[2] * s.bbox[3] if s.bbox is not None else 0)
        
        # 座標を元のフレームに変換
        best.keypoints[:, 0] += x1
        best.keypoints[:, 1] += y1
        best.bbox = bbox
        best.track_id = track_id
        
        return best


class TeamClassifier:
    """
    チーム分類器
    ジャージの色からチームA/Bを分類
    """
    
    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: int = 2
    ):
        """
        Args:
            method: 分類手法 ('kmeans', 'histogram', 'clip')
            n_clusters: クラスタ数
        """
        self.method = method
        self.n_clusters = n_clusters
        self.team_colors = None  # 学習後のチームカラー
        self.is_trained = False
    
    def train(self, player_crops: List[np.ndarray]) -> bool:
        """
        選手画像からチームカラーを学習
        
        Args:
            player_crops: 選手の切り抜き画像リスト
            
        Returns:
            成功したかどうか
        """
        if len(player_crops) < 2:
            return False
        
        # 各選手の代表色を抽出
        colors = []
        for crop in player_crops:
            color = self._extract_dominant_color(crop)
            colors.append(color)
        
        colors = np.array(colors)
        
        # K-meansでクラスタリング
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans.fit(colors)
        
        self.team_colors = kmeans.cluster_centers_
        self.kmeans = kmeans
        self.is_trained = True
        
        return True
    
    def classify(self, player_crop: np.ndarray) -> str:
        """
        選手画像からチームを分類
        
        Args:
            player_crop: 選手の切り抜き画像
            
        Returns:
            "A" or "B"
        """
        if not self.is_trained:
            return "A"  # デフォルト
        
        color = self._extract_dominant_color(player_crop)
        cluster = self.kmeans.predict([color])[0]
        
        return "A" if cluster == 0 else "B"
    
    def _extract_dominant_color(self, image: np.ndarray) -> np.ndarray:
        """
        画像の代表色を抽出（上半身中央部分）
        
        Args:
            image: BGR画像
            
        Returns:
            代表色 [B, G, R]
        """
        h, w = image.shape[:2]
        
        # 上半身中央部分を抽出（ジャージ部分）
        y_start = int(h * 0.2)
        y_end = int(h * 0.5)
        x_start = int(w * 0.2)
        x_end = int(w * 0.8)
        
        roi = image[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            return np.array([0, 0, 0])
        
        # HSVに変換して彩度の高いピクセルをフィルタリング
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 彩度が一定以上のピクセルを抽出
        saturation_mask = hsv[:, :, 1] > 50
        
        if np.sum(saturation_mask) < 10:
            # 彩度の高いピクセルが少ない場合は全体の平均
            return np.mean(roi.reshape(-1, 3), axis=0)
        
        # マスク適用して平均色を計算
        masked_pixels = roi[saturation_mask]
        return np.mean(masked_pixels, axis=0)
    
    def filter_non_players(
        self,
        detections: List,
        frame: np.ndarray
    ) -> List:
        """
        審判・観客を除外
        
        Args:
            detections: 検出結果リスト
            frame: BGR画像
            
        Returns:
            フィルタリング後の検出リスト
        """
        if not self.is_trained:
            return detections
        
        filtered = []
        for det in detections:
            # バウンディングボックス領域を切り出し
            x1, y1, x2, y2 = map(int, det.bbox)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            color = self._extract_dominant_color(crop)
            
            # チームカラーとの距離を計算
            distances = [np.linalg.norm(color - tc) for tc in self.team_colors]
            min_distance = min(distances)
            
            # 閾値以内ならプレイヤーとして採用
            if min_distance < 100:  # 閾値は調整可能
                filtered.append(det)
        
        return filtered


