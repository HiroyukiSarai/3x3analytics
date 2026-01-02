"""
マルチオブジェクトトラッキングモジュール
BoxMOT（ByteTrack/BoT-SORT）を使用した選手追跡
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from boxmot import ByteTrack, BoTSORT, DeepOCSORT
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("Warning: boxmot not installed. Run: pip install boxmot")


@dataclass
class Track:
    """トラック情報を格納するデータクラス"""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int = 0
    team: Optional[str] = None  # "A" or "B"
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def center(self) -> Tuple[float, float]:
        """トラックの中心座標"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    @property
    def bottom_center(self) -> Tuple[float, float]:
        """足元座標"""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3]
        )
    
    def update_trajectory(self, max_length: int = 90):
        """軌跡を更新（デフォルト: 3秒@30fps）"""
        self.trajectory.append(self.center)
        if len(self.trajectory) > max_length:
            self.trajectory.pop(0)


class BasketballTracker:
    """3x3バスケットボール用マルチオブジェクトトラッカー"""
    
    TRACKER_TYPES = ["bytetrack", "botsort", "deepocsort"]
    
    def __init__(
        self,
        tracker_type: str = "bytetrack",
        device: str = "cuda:0",
        reid_weights: Optional[str] = None,
        max_trajectory_length: int = 90
    ):
        """
        Args:
            tracker_type: トラッカーの種類 ('bytetrack', 'botsort', 'deepocsort')
            device: 推論デバイス
            reid_weights: Re-IDモデルの重み（BoT-SORT/DeepOCSORT用）
            max_trajectory_length: 軌跡の最大長（フレーム数）
        """
        if not BOXMOT_AVAILABLE:
            raise ImportError("boxmot is required. Run: pip install boxmot")
        
        self.tracker_type = tracker_type.lower()
        self.device = device
        self.max_trajectory_length = max_trajectory_length
        
        # トラッカー初期化
        self.tracker = self._create_tracker(reid_weights)
        
        # トラッキング履歴
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.active_tracks: Dict[int, Track] = {}
        
        # 統計情報
        self.frame_count = 0
        self.total_tracks = 0
    
    def _create_tracker(self, reid_weights: Optional[str] = None):
        """トラッカーインスタンスを作成"""
        if self.tracker_type == "bytetrack":
            return ByteTrack()
        
        elif self.tracker_type == "botsort":
            weights = Path(reid_weights) if reid_weights else Path("osnet_x0_25_msmt17.pt")
            return BoTSORT(
                model_weights=weights,
                device=self.device,
                fp16=False
            )
        
        elif self.tracker_type == "deepocsort":
            weights = Path(reid_weights) if reid_weights else Path("osnet_x0_25_msmt17.pt")
            return DeepOCSORT(
                model_weights=weights,
                device=self.device,
                fp16=False
            )
        
        else:
            raise ValueError(
                f"Unknown tracker type: {self.tracker_type}. "
                f"Available: {self.TRACKER_TYPES}"
            )
    
    def update(
        self, 
        detections: np.ndarray, 
        frame: np.ndarray
    ) -> List[Track]:
        """
        検出結果を更新してトラッキング
        
        Args:
            detections: 検出結果 [[x1, y1, x2, y2, conf], ...] (N, 5)
            frame: BGR画像（Re-ID用）
            
        Returns:
            アクティブなトラックのリスト
        """
        self.frame_count += 1
        
        if len(detections) == 0:
            return list(self.active_tracks.values())
        
        # BoxMOT形式に変換 (x1, y1, x2, y2, conf, cls)
        dets = np.zeros((len(detections), 6))
        dets[:, :5] = detections
        dets[:, 5] = 0  # cls = person
        
        # トラッカー更新
        tracks_array = self.tracker.update(dets, frame)
        
        # トラック情報を更新
        current_track_ids = set()
        tracks = []
        
        for track_data in tracks_array:
            track_id = int(track_data[4])
            current_track_ids.add(track_id)
            
            bbox = track_data[:4]
            conf = float(track_data[5]) if len(track_data) > 5 else 1.0
            
            # 既存トラックを更新または新規作成
            if track_id in self.active_tracks:
                track = self.active_tracks[track_id]
                track.bbox = bbox
                track.confidence = conf
            else:
                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=0
                )
                self.active_tracks[track_id] = track
                self.total_tracks += 1
            
            # 軌跡を更新
            track.update_trajectory(self.max_trajectory_length)
            self.track_history[track_id].append(track.center)
            
            # 履歴の長さを制限
            if len(self.track_history[track_id]) > self.max_trajectory_length:
                self.track_history[track_id].pop(0)
            
            tracks.append(track)
        
        # 消失したトラックを処理（一定フレーム後に削除することも可能）
        # ここでは保持し続ける
        
        return tracks
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """指定IDの軌跡を取得"""
        return self.track_history.get(track_id, [])
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """全トラックの軌跡を取得"""
        return dict(self.track_history)
    
    def reset(self):
        """トラッカーをリセット"""
        self.tracker = self._create_tracker()
        self.track_history.clear()
        self.active_tracks.clear()
        self.frame_count = 0
        self.total_tracks = 0
    
    def get_stats(self) -> Dict[str, any]:
        """トラッキング統計を取得"""
        return {
            "frame_count": self.frame_count,
            "total_tracks": self.total_tracks,
            "active_tracks": len(self.active_tracks),
            "tracker_type": self.tracker_type
        }


class TeamAwareTracker(BasketballTracker):
    """チーム情報を考慮したトラッカー"""
    
    def __init__(
        self,
        team_classifier=None,
        **kwargs
    ):
        """
        Args:
            team_classifier: チーム分類器（Phase 3で実装）
            **kwargs: BasketballTrackerの引数
        """
        super().__init__(**kwargs)
        self.team_classifier = team_classifier
    
    def update_with_teams(
        self,
        detections: np.ndarray,
        frame: np.ndarray
    ) -> List[Track]:
        """チーム情報付きでトラッキング更新"""
        tracks = self.update(detections, frame)
        
        if self.team_classifier is not None:
            for track in tracks:
                # バウンディングボックス領域を切り出し
                x1, y1, x2, y2 = map(int, track.bbox)
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    team = self.team_classifier.classify(crop)
                    track.team = team
        
        return tracks


