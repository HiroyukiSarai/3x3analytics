"""
物体検出モジュール
YOLO11を使用した選手・ボール検出
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


@dataclass
class Detection:
    """検出結果を格納するデータクラス"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    
    @property
    def center(self) -> Tuple[float, float]:
        """バウンディングボックスの中心座標"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """足元座標（バウンディングボックスの下端中央）"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3]
        )
    
    @property
    def area(self) -> float:
        """バウンディングボックスの面積"""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


class BasketballDetector:
    """3x3バスケットボール用物体検出器"""
    
    # COCOデータセットのクラスID
    PERSON_CLASS = 0
    SPORTS_BALL_CLASS = 32
    
    def __init__(
        self, 
        model_path: str = "yolo11n.pt",
        device: str = "cuda:0",
        conf_threshold: float = 0.5,
        ball_conf_threshold: float = 0.3
    ):
        """
        Args:
            model_path: YOLOモデルのパス（事前学習済みまたはファインチューニング済み）
            device: 推論デバイス ('cuda:0', 'cuda:1', 'cpu')
            conf_threshold: 選手検出の信頼度閾値
            ball_conf_threshold: ボール検出の信頼度閾値
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required. Run: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.ball_conf_threshold = ball_conf_threshold
        
    def detect(self, frame: np.ndarray) -> Dict[str, any]:
        """
        フレームから選手とボールを検出
        
        Args:
            frame: BGR画像 (H, W, 3)
            
        Returns:
            {
                'players': List[Detection],
                'ball': Optional[Detection],
                'raw_detections': np.ndarray  # トラッカー用 [N, 5]
            }
        """
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        players = []
        ball = None
        raw_detections = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if cls == self.PERSON_CLASS and conf > self.conf_threshold:
                detection = Detection(
                    bbox=xyxy,
                    confidence=conf,
                    class_id=cls,
                    class_name="player"
                )
                players.append(detection)
                raw_detections.append([*xyxy, conf])
                
            elif cls == self.SPORTS_BALL_CLASS and conf > self.ball_conf_threshold:
                # 最も信頼度の高いボールを採用
                if ball is None or conf > ball.confidence:
                    ball = Detection(
                        bbox=xyxy,
                        confidence=conf,
                        class_id=cls,
                        class_name="ball"
                    )
        
        return {
            'players': players,
            'ball': ball,
            'raw_detections': np.array(raw_detections) if raw_detections else np.empty((0, 5))
        }
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[Dict[str, any]]:
        """バッチ処理で複数フレームを検出"""
        return [self.detect(frame) for frame in frames]


class BallDetector:
    """
    ボール専用検出器
    
    通常のYOLOでは小さく高速なボールの検出が困難なため、
    専用モデル（TrackNet等）を使用する場合のインターフェース
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0"
    ):
        """
        Args:
            model_path: ボール専用モデルのパス（Noneの場合はYOLO使用）
            device: 推論デバイス
        """
        self.device = device
        
        if model_path and Path(model_path).exists():
            # カスタムボール検出モデル
            self.model = YOLO(model_path)
            self.use_custom = True
        else:
            # 汎用YOLOを使用
            self.model = YOLO("yolo11n.pt")
            self.use_custom = False
    
    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """ボールを検出"""
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        best_ball = None
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # カスタムモデルの場合はクラス0がボール
            target_cls = 0 if self.use_custom else 32
            
            if cls == target_cls and conf > 0.3:
                if best_ball is None or conf > best_ball.confidence:
                    best_ball = Detection(
                        bbox=box.xyxy[0].cpu().numpy(),
                        confidence=conf,
                        class_id=cls,
                        class_name="ball"
                    )
        
        return best_ball
    
    def track_ball(
        self, 
        frames: List[np.ndarray],
        interpolate: bool = True
    ) -> List[Optional[Detection]]:
        """
        複数フレームでボールを追跡
        検出できなかったフレームは補間可能
        
        Args:
            frames: フレームのリスト
            interpolate: 欠損フレームを補間するか
            
        Returns:
            各フレームのボール検出結果リスト
        """
        detections = [self.detect(frame) for frame in frames]
        
        if interpolate:
            detections = self._interpolate_missing(detections)
        
        return detections
    
    def _interpolate_missing(
        self, 
        detections: List[Optional[Detection]]
    ) -> List[Optional[Detection]]:
        """欠損フレームを線形補間"""
        result = detections.copy()
        
        # 検出されたフレームのインデックスと位置を取得
        detected_indices = []
        detected_positions = []
        
        for i, det in enumerate(detections):
            if det is not None:
                detected_indices.append(i)
                detected_positions.append(det.center)
        
        if len(detected_indices) < 2:
            return result
        
        # 欠損フレームを補間
        for i in range(len(result)):
            if result[i] is None:
                # 前後の検出フレームを探す
                prev_idx = None
                next_idx = None
                
                for j in detected_indices:
                    if j < i:
                        prev_idx = j
                    elif j > i and next_idx is None:
                        next_idx = j
                        break
                
                if prev_idx is not None and next_idx is not None:
                    # 線形補間
                    t = (i - prev_idx) / (next_idx - prev_idx)
                    prev_pos = detected_positions[detected_indices.index(prev_idx)]
                    next_pos = detected_positions[detected_indices.index(next_idx)]
                    
                    interp_x = prev_pos[0] + t * (next_pos[0] - prev_pos[0])
                    interp_y = prev_pos[1] + t * (next_pos[1] - prev_pos[1])
                    
                    # 補間されたボール（サイズは前フレームから継承）
                    prev_det = detections[prev_idx]
                    w = prev_det.bbox[2] - prev_det.bbox[0]
                    h = prev_det.bbox[3] - prev_det.bbox[1]
                    
                    result[i] = Detection(
                        bbox=np.array([
                            interp_x - w/2, interp_y - h/2,
                            interp_x + w/2, interp_y + h/2
                        ]),
                        confidence=0.0,  # 補間されたことを示す
                        class_id=32,
                        class_name="ball_interpolated"
                    )
        
        return result


