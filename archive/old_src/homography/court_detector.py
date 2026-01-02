"""
コート検出モジュール
3x3バスケットボールコートのキーポイントを検出
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class CourtKeypoints:
    """
    3x3バスケットボールコートのキーポイント
    
    コート配置（ハーフコート）:
    
        0 -------- 1 -------- 2
        |                     |
        |    3 ------- 4      |
        |    |  5   6  |      |
        |    7 ------- 8      |
        |          9          |
        10 ------- 11 ------- 12
        
    0-2: ハーフコートライン（トップ）
    3-8: フリースローエリア
    5, 6: フリースローライン上のポイント
    9: ゴール位置
    10-12: ベースライン
    """
    
    # キーポイント座標 [x, y, visibility]
    points: np.ndarray = field(default_factory=lambda: np.zeros((13, 3)))
    
    # 3x3コートの実寸（メートル）
    COURT_WIDTH = 15.0  # 横幅
    COURT_LENGTH = 11.0  # 縦（ハーフコート）
    FREE_THROW_LINE = 5.8  # フリースローライン距離
    THREE_POINT_DISTANCE = 6.75  # 3ポイントライン距離
    
    # 2D座標での標準コート（ピクセル）
    STANDARD_COURT_2D = {
        0: (0, 0),
        1: (750, 0),
        2: (1500, 0),
        3: (450, 400),
        4: (1050, 400),
        5: (600, 580),
        6: (900, 580),
        7: (450, 760),
        8: (1050, 760),
        9: (750, 900),
        10: (0, 1100),
        11: (750, 1100),
        12: (1500, 1100)
    }
    
    def get_visible_points(self) -> List[Tuple[int, Tuple[float, float]]]:
        """可視のキーポイントをリストで取得"""
        visible = []
        for i, point in enumerate(self.points):
            if point[2] > 0.5:  # visibility > 0.5
                visible.append((i, (point[0], point[1])))
        return visible
    
    def to_array(self) -> np.ndarray:
        """[x, y]のみの配列に変換"""
        return self.points[:, :2]
    
    def get_standard_2d_points(self, indices: List[int]) -> np.ndarray:
        """指定インデックスの標準2Dコート座標を取得"""
        points = []
        for idx in indices:
            if idx in self.STANDARD_COURT_2D:
                points.append(self.STANDARD_COURT_2D[idx])
        return np.array(points, dtype=np.float32)


class CourtDetector:
    """
    コートキーポイント検出器
    
    検出方法:
    1. 事前学習モデルによる自動検出
    2. 手動キーポイント指定（フォールバック）
    3. ライン検出ベースの自動推定
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0"
    ):
        """
        Args:
            model_path: コートキーポイント検出モデルのパス
            device: 推論デバイス
        """
        self.device = device
        self.model = None
        
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"Warning: Failed to load court model: {e}")
    
    def detect(self, frame: np.ndarray) -> CourtKeypoints:
        """
        コートキーポイントを検出
        
        Args:
            frame: BGR画像
            
        Returns:
            CourtKeypoints: 検出されたキーポイント
        """
        if self.model is not None:
            return self._detect_with_model(frame)
        else:
            return self._detect_with_lines(frame)
    
    def _detect_with_model(self, frame: np.ndarray) -> CourtKeypoints:
        """モデルベースのキーポイント検出"""
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        keypoints = CourtKeypoints()
        
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            kpts = results.keypoints.data[0].cpu().numpy()
            keypoints.points[:len(kpts)] = kpts
        
        return keypoints
    
    def _detect_with_lines(self, frame: np.ndarray) -> CourtKeypoints:
        """
        ライン検出ベースのキーポイント推定
        Hough変換でコートラインを検出し、交点を計算
        """
        keypoints = CourtKeypoints()
        
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough変換でライン検出
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            return keypoints
        
        # ラインをフィルタリング（水平/垂直ラインのみ）
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            if angle < 15 or angle > 165:  # 水平ライン
                horizontal_lines.append(line[0])
            elif 75 < angle < 105:  # 垂直ライン
                vertical_lines.append(line[0])
        
        # 交点を計算してキーポイントを推定
        intersections = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = self._line_intersection(
                    (h_line[0], h_line[1]), (h_line[2], h_line[3]),
                    (v_line[0], v_line[1]), (v_line[2], v_line[3])
                )
                if intersection is not None:
                    intersections.append(intersection)
        
        # 交点をクラスタリングして代表点を選択
        if intersections:
            intersections = np.array(intersections)
            # 簡易的なクラスタリング（K-means等で改善可能）
            for i, point in enumerate(intersections[:13]):
                keypoints.points[i] = [point[0], point[1], 1.0]
        
        return keypoints
    
    def _line_intersection(
        self,
        p1: Tuple[float, float], p2: Tuple[float, float],
        p3: Tuple[float, float], p4: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """2つの線分の交点を計算"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None  # 平行
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def set_manual_keypoints(
        self,
        points: List[Tuple[float, float]],
        indices: Optional[List[int]] = None
    ) -> CourtKeypoints:
        """
        手動でキーポイントを設定
        
        Args:
            points: キーポイント座標のリスト [(x1, y1), (x2, y2), ...]
            indices: 各ポイントに対応するキーポイントインデックス
            
        Returns:
            CourtKeypoints: 設定されたキーポイント
        """
        keypoints = CourtKeypoints()
        
        if indices is None:
            indices = list(range(len(points)))
        
        for i, (idx, point) in enumerate(zip(indices, points)):
            if 0 <= idx < 13:
                keypoints.points[idx] = [point[0], point[1], 1.0]
        
        return keypoints


