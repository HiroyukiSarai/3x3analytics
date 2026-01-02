"""
ホモグラフィ変換モジュール
カメラ視点から2Dコート座標への変換
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .court_detector import CourtKeypoints


@dataclass
class TransformResult:
    """変換結果"""
    court_position: Tuple[float, float]  # 2Dコート座標
    original_position: Tuple[float, float]  # 元のピクセル座標
    is_valid: bool = True  # コート内かどうか


class HomographyTransformer:
    """
    ホモグラフィ変換器
    カメラ視点から2D真上視点（コート座標）への変換
    """
    
    # 2Dコートの標準サイズ（ピクセル）
    COURT_2D_WIDTH = 1500
    COURT_2D_HEIGHT = 1100
    
    def __init__(
        self,
        court_keypoints: Optional[CourtKeypoints] = None,
        homography_matrix: Optional[np.ndarray] = None
    ):
        """
        Args:
            court_keypoints: コートキーポイント（ホモグラフィ計算用）
            homography_matrix: 事前計算されたホモグラフィ行列
        """
        self.court_keypoints = court_keypoints
        self.H = homography_matrix
        self.H_inv = None
        
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)
    
    def compute_homography(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> bool:
        """
        ホモグラフィ行列を計算
        
        Args:
            src_points: カメラ視点のポイント (N, 2)
            dst_points: 2Dコート上のポイント (N, 2)
            
        Returns:
            成功したかどうか
        """
        if len(src_points) < 4 or len(dst_points) < 4:
            print("Error: At least 4 point pairs required")
            return False
        
        # RANSAC使用でホモグラフィ計算
        self.H, mask = cv2.findHomography(
            src_points.astype(np.float32),
            dst_points.astype(np.float32),
            cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if self.H is None:
            print("Error: Failed to compute homography")
            return False
        
        self.H_inv = np.linalg.inv(self.H)
        return True
    
    def compute_from_keypoints(
        self,
        detected_keypoints: CourtKeypoints,
        keypoint_indices: Optional[List[int]] = None
    ) -> bool:
        """
        検出されたキーポイントからホモグラフィを計算
        
        Args:
            detected_keypoints: 検出されたコートキーポイント
            keypoint_indices: 使用するキーポイントのインデックス
            
        Returns:
            成功したかどうか
        """
        self.court_keypoints = detected_keypoints
        
        # 可視のキーポイントを取得
        visible = detected_keypoints.get_visible_points()
        
        if len(visible) < 4:
            print(f"Error: Only {len(visible)} visible keypoints, need at least 4")
            return False
        
        # 使用するインデックスをフィルタリング
        if keypoint_indices:
            visible = [(i, p) for i, p in visible if i in keypoint_indices]
        
        if len(visible) < 4:
            print("Error: Not enough matching keypoints")
            return False
        
        # ソースポイント（カメラ視点）
        indices = [v[0] for v in visible]
        src_points = np.array([v[1] for v in visible], dtype=np.float32)
        
        # デスティネーションポイント（2Dコート）
        dst_points = detected_keypoints.get_standard_2d_points(indices)
        
        return self.compute_homography(src_points, dst_points)
    
    def transform_point(
        self,
        point: Tuple[float, float]
    ) -> TransformResult:
        """
        1点をカメラ座標から2Dコート座標に変換
        
        Args:
            point: (x, y) カメラ視点のピクセル座標
            
        Returns:
            TransformResult: 変換結果
        """
        if self.H is None:
            return TransformResult(
                court_position=(0, 0),
                original_position=point,
                is_valid=False
            )
        
        # 同次座標に変換
        src = np.array([[point[0], point[1], 1.0]], dtype=np.float32).T
        
        # ホモグラフィ変換
        dst = self.H @ src
        dst = dst / dst[2, 0]  # 正規化
        
        court_x, court_y = dst[0, 0], dst[1, 0]
        
        # コート内判定
        is_valid = (
            0 <= court_x <= self.COURT_2D_WIDTH and
            0 <= court_y <= self.COURT_2D_HEIGHT
        )
        
        return TransformResult(
            court_position=(court_x, court_y),
            original_position=point,
            is_valid=is_valid
        )
    
    def transform_points(
        self,
        points: np.ndarray
    ) -> np.ndarray:
        """
        複数点を一括変換
        
        Args:
            points: (N, 2) カメラ視点の座標
            
        Returns:
            (N, 2) 2Dコート座標
        """
        if self.H is None or len(points) == 0:
            return points
        
        # 同次座標に変換
        ones = np.ones((len(points), 1))
        src = np.hstack([points, ones])
        
        # ホモグラフィ変換
        dst = (self.H @ src.T).T
        dst = dst[:, :2] / dst[:, 2:3]  # 正規化
        
        return dst
    
    def inverse_transform_point(
        self,
        court_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        2Dコート座標からカメラ座標に逆変換
        
        Args:
            court_point: (x, y) 2Dコート座標
            
        Returns:
            (x, y) カメラ視点のピクセル座標
        """
        if self.H_inv is None:
            return court_point
        
        src = np.array([[court_point[0], court_point[1], 1.0]], dtype=np.float32).T
        dst = self.H_inv @ src
        dst = dst / dst[2, 0]
        
        return (dst[0, 0], dst[1, 0])
    
    def warp_frame(
        self,
        frame: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        フレーム全体を2Dコート視点に変換
        
        Args:
            frame: BGR画像
            output_size: 出力サイズ (width, height)
            
        Returns:
            変換後の画像
        """
        if self.H is None:
            return frame
        
        if output_size is None:
            output_size = (self.COURT_2D_WIDTH, self.COURT_2D_HEIGHT)
        
        warped = cv2.warpPerspective(frame, self.H, output_size)
        return warped
    
    def save(self, filepath: str):
        """ホモグラフィ行列を保存"""
        if self.H is not None:
            np.save(filepath, self.H)
    
    def load(self, filepath: str) -> bool:
        """ホモグラフィ行列を読み込み"""
        try:
            self.H = np.load(filepath)
            self.H_inv = np.linalg.inv(self.H)
            return True
        except Exception as e:
            print(f"Error loading homography: {e}")
            return False


class CourtVisualizer:
    """2Dコート可視化"""
    
    # コートの色
    COURT_COLOR = (34, 139, 34)  # 緑
    LINE_COLOR = (255, 255, 255)  # 白
    TEAM_A_COLOR = (255, 0, 0)  # 青（BGR）
    TEAM_B_COLOR = (0, 0, 255)  # 赤
    BALL_COLOR = (0, 165, 255)  # オレンジ
    
    def __init__(
        self,
        width: int = 1500,
        height: int = 1100
    ):
        """
        Args:
            width: 2Dコート画像の幅
            height: 2Dコート画像の高さ
        """
        self.width = width
        self.height = height
        self.court_image = self._create_court()
    
    def _create_court(self) -> np.ndarray:
        """3x3コートの2D画像を生成"""
        court = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        court[:] = self.COURT_COLOR
        
        # コート外枠
        cv2.rectangle(court, (0, 0), (self.width-1, self.height-1), 
                     self.LINE_COLOR, 3)
        
        # ハーフコートライン（トップ）
        cv2.line(court, (0, 0), (self.width, 0), self.LINE_COLOR, 3)
        
        # ベースライン（ボトム）
        cv2.line(court, (0, self.height-1), (self.width, self.height-1), 
                self.LINE_COLOR, 3)
        
        # フリースローエリア
        ft_left = 450
        ft_right = 1050
        ft_top = 400
        ft_bottom = 760
        cv2.rectangle(court, (ft_left, ft_top), (ft_right, ft_bottom), 
                     self.LINE_COLOR, 2)
        
        # フリースローライン
        cv2.line(court, (ft_left, 580), (ft_right, 580), self.LINE_COLOR, 2)
        
        # 3ポイントアーク（簡略化）
        center = (self.width // 2, self.height - 100)
        cv2.ellipse(court, center, (550, 550), 0, 180, 360, self.LINE_COLOR, 2)
        
        # ゴール位置
        goal_pos = (self.width // 2, self.height - 100)
        cv2.circle(court, goal_pos, 30, self.LINE_COLOR, 2)
        
        return court
    
    def draw_players(
        self,
        court_positions: List[Tuple[float, float, str]],  # [(x, y, team), ...]
        draw_trails: bool = True,
        trails: Optional[Dict[int, List[Tuple[float, float]]]] = None
    ) -> np.ndarray:
        """
        2Dコート上に選手を描画
        
        Args:
            court_positions: [(x, y, team), ...] リスト
            draw_trails: 軌跡を描画するか
            trails: {track_id: [(x, y), ...]} 軌跡データ
            
        Returns:
            描画された2Dコート画像
        """
        court = self.court_image.copy()
        
        # 軌跡を描画
        if draw_trails and trails:
            for track_id, trail in trails.items():
                if len(trail) > 1:
                    points = np.array(trail, dtype=np.int32)
                    # チームによる色分けは別途対応
                    cv2.polylines(court, [points], False, (200, 200, 200), 2)
        
        # 選手を描画
        for i, (x, y, team) in enumerate(court_positions):
            color = self.TEAM_A_COLOR if team == "A" else self.TEAM_B_COLOR
            pos = (int(x), int(y))
            
            # 選手マーカー
            cv2.circle(court, pos, 20, color, -1)
            cv2.circle(court, pos, 20, (255, 255, 255), 2)
            
            # ID表示
            cv2.putText(court, str(i+1), (pos[0]-8, pos[1]+8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return court
    
    def draw_ball(
        self,
        court: np.ndarray,
        position: Tuple[float, float]
    ) -> np.ndarray:
        """ボールを描画"""
        pos = (int(position[0]), int(position[1]))
        cv2.circle(court, pos, 12, self.BALL_COLOR, -1)
        cv2.circle(court, pos, 12, (255, 255, 255), 2)
        return court


