"""
選手ルート分析モジュール
得点シーンの選手移動ルートを抽出・正規化
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class PlayerRoute:
    """選手の移動ルートデータ"""
    track_id: int
    team: str  # "A" or "B"
    positions: List[Tuple[float, float]]  # 2Dコート座標
    timestamps: List[float]  # タイムスタンプ（秒）
    has_ball: List[bool] = field(default_factory=list)  # ボール保持状態
    
    @property
    def start_position(self) -> Tuple[float, float]:
        """開始位置"""
        return self.positions[0] if self.positions else (0, 0)
    
    @property
    def end_position(self) -> Tuple[float, float]:
        """終了位置"""
        return self.positions[-1] if self.positions else (0, 0)
    
    @property
    def total_distance(self) -> float:
        """総移動距離"""
        if len(self.positions) < 2:
            return 0.0
        
        distance = 0.0
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i-1][0]
            dy = self.positions[i][1] - self.positions[i-1][1]
            distance += np.sqrt(dx**2 + dy**2)
        
        return distance
    
    @property
    def average_speed(self) -> float:
        """平均速度（ピクセル/秒）"""
        if len(self.timestamps) < 2:
            return 0.0
        
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
        
        return self.total_distance / duration
    
    def normalize(
        self,
        reference_point: Tuple[float, float] = (750, 550)
    ) -> 'PlayerRoute':
        """
        開始位置を基準点に正規化
        
        Args:
            reference_point: 正規化の基準点（コート中央など）
            
        Returns:
            正規化されたPlayerRoute
        """
        if not self.positions:
            return self
        
        # 開始位置からのオフセット
        start = self.start_position
        offset_x = reference_point[0] - start[0]
        offset_y = reference_point[1] - start[1]
        
        normalized_positions = [
            (x + offset_x, y + offset_y) for x, y in self.positions
        ]
        
        return PlayerRoute(
            track_id=self.track_id,
            team=self.team,
            positions=normalized_positions,
            timestamps=self.timestamps.copy(),
            has_ball=self.has_ball.copy()
        )
    
    def to_array(self) -> np.ndarray:
        """numpy配列に変換"""
        return np.array(self.positions)
    
    def resample(self, n_points: int = 50) -> 'PlayerRoute':
        """
        一定間隔でリサンプリング
        
        Args:
            n_points: リサンプリング後のポイント数
            
        Returns:
            リサンプリングされたPlayerRoute
        """
        if len(self.positions) < 2:
            return self
        
        positions = np.array(self.positions)
        
        # 累積距離を計算
        distances = np.zeros(len(positions))
        for i in range(1, len(positions)):
            distances[i] = distances[i-1] + np.linalg.norm(
                positions[i] - positions[i-1]
            )
        
        if distances[-1] == 0:
            return self
        
        # 等間隔でサンプリング
        target_distances = np.linspace(0, distances[-1], n_points)
        
        resampled_positions = []
        for target_d in target_distances:
            # 補間
            idx = np.searchsorted(distances, target_d)
            if idx == 0:
                resampled_positions.append(tuple(positions[0]))
            elif idx >= len(positions):
                resampled_positions.append(tuple(positions[-1]))
            else:
                # 線形補間
                t = (target_d - distances[idx-1]) / (distances[idx] - distances[idx-1])
                p = positions[idx-1] + t * (positions[idx] - positions[idx-1])
                resampled_positions.append(tuple(p))
        
        # タイムスタンプも補間
        if self.timestamps:
            resampled_timestamps = np.linspace(
                self.timestamps[0], self.timestamps[-1], n_points
            ).tolist()
        else:
            resampled_timestamps = list(range(n_points))
        
        return PlayerRoute(
            track_id=self.track_id,
            team=self.team,
            positions=resampled_positions,
            timestamps=resampled_timestamps,
            has_ball=[]  # ボール保持状態は補間しない
        )


class RouteAnalyzer:
    """選手ルート分析器"""
    
    def __init__(
        self,
        fps: float = 30.0,
        pre_score_duration: float = 10.0
    ):
        """
        Args:
            fps: 動画のFPS
            pre_score_duration: 得点前の分析区間（秒）
        """
        self.fps = fps
        self.pre_score_duration = pre_score_duration
    
    def extract_routes_from_tracking(
        self,
        tracking_data: List[Dict],
        start_frame: int,
        end_frame: int,
        team_filter: Optional[str] = None
    ) -> Dict[int, PlayerRoute]:
        """
        トラッキングデータからルートを抽出
        
        Args:
            tracking_data: フレームごとのトラッキングデータ
            start_frame: 開始フレーム
            end_frame: 終了フレーム
            team_filter: チームでフィルタリング（"A" or "B"）
            
        Returns:
            {track_id: PlayerRoute}
        """
        routes = defaultdict(lambda: {
            'positions': [],
            'timestamps': [],
            'has_ball': [],
            'team': None
        })
        
        for frame_data in tracking_data:
            frame_id = frame_data.get('frame_id', 0)
            
            if frame_id < start_frame or frame_id > end_frame:
                continue
            
            timestamp = frame_id / self.fps
            
            for player in frame_data.get('players', []):
                track_id = player.get('track_id')
                team = player.get('team', 'A')
                
                if team_filter and team != team_filter:
                    continue
                
                # 2Dコート座標（ホモグラフィ変換後）
                court_pos = player.get('court_position')
                if court_pos is None:
                    # bbox から足元座標を計算
                    bbox = player.get('bbox', [0, 0, 0, 0])
                    court_pos = ((bbox[0] + bbox[2]) / 2, bbox[3])
                
                routes[track_id]['positions'].append(tuple(court_pos))
                routes[track_id]['timestamps'].append(timestamp)
                routes[track_id]['has_ball'].append(player.get('has_ball', False))
                routes[track_id]['team'] = team
        
        # PlayerRouteオブジェクトに変換
        result = {}
        for track_id, data in routes.items():
            if len(data['positions']) > 0:
                result[track_id] = PlayerRoute(
                    track_id=track_id,
                    team=data['team'] or 'A',
                    positions=data['positions'],
                    timestamps=data['timestamps'],
                    has_ball=data['has_ball']
                )
        
        return result
    
    def extract_scoring_play_routes(
        self,
        tracking_data: List[Dict],
        scoring_frames: List[int]
    ) -> List[Dict[int, PlayerRoute]]:
        """
        得点シーンのルートを抽出
        
        Args:
            tracking_data: 全フレームのトラッキングデータ
            scoring_frames: 得点が入ったフレームのリスト
            
        Returns:
            各得点シーンのルートデータリスト
        """
        pre_frames = int(self.pre_score_duration * self.fps)
        
        all_plays = []
        for score_frame in scoring_frames:
            start_frame = max(0, score_frame - pre_frames)
            routes = self.extract_routes_from_tracking(
                tracking_data, start_frame, score_frame
            )
            all_plays.append(routes)
        
        return all_plays
    
    def calculate_spacing(
        self,
        routes: Dict[int, PlayerRoute],
        frame_index: int = -1
    ) -> Dict[str, float]:
        """
        選手間の距離（スペーシング）を計算
        
        Args:
            routes: 選手ルートデータ
            frame_index: 計算するフレーム（-1で最終フレーム）
            
        Returns:
            スペーシング統計
        """
        positions = []
        for route in routes.values():
            if route.positions:
                pos = route.positions[frame_index] if abs(frame_index) <= len(route.positions) else route.positions[-1]
                positions.append(pos)
        
        if len(positions) < 2:
            return {'avg_distance': 0, 'min_distance': 0, 'max_distance': 0}
        
        distances = []
        positions = np.array(positions)
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        return {
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances)
        }
    
    def calculate_route_similarity(
        self,
        route1: PlayerRoute,
        route2: PlayerRoute,
        method: str = 'dtw'
    ) -> float:
        """
        2つのルートの類似度を計算
        
        Args:
            route1, route2: 比較するルート
            method: 類似度計算手法 ('dtw', 'frechet', 'hausdorff')
            
        Returns:
            類似度（距離、小さいほど類似）
        """
        # リサンプリング
        r1 = route1.resample(50).to_array()
        r2 = route2.resample(50).to_array()
        
        if method == 'dtw':
            return self._dtw_distance(r1, r2)
        elif method == 'frechet':
            return self._frechet_distance(r1, r2)
        else:  # hausdorff
            return self._hausdorff_distance(r1, r2)
    
    def _dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Dynamic Time Warping距離"""
        try:
            from dtaidistance import dtw
            # 2次元座標を1次元に変換
            s1 = np.sqrt(seq1[:, 0]**2 + seq1[:, 1]**2)
            s2 = np.sqrt(seq2[:, 0]**2 + seq2[:, 1]**2)
            return dtw.distance(s1, s2)
        except ImportError:
            # フォールバック: ユークリッド距離の合計
            return np.sum(np.linalg.norm(seq1 - seq2, axis=1))
    
    def _frechet_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Fréchet距離（簡易版）"""
        n, m = len(seq1), len(seq2)
        
        # 距離行列
        dist_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dist_matrix[i, j] = np.linalg.norm(seq1[i] - seq2[j])
        
        # 動的計画法
        dp = np.zeros((n, m))
        dp[0, 0] = dist_matrix[0, 0]
        
        for i in range(1, n):
            dp[i, 0] = max(dp[i-1, 0], dist_matrix[i, 0])
        for j in range(1, m):
            dp[0, j] = max(dp[0, j-1], dist_matrix[0, j])
        
        for i in range(1, n):
            for j in range(1, m):
                dp[i, j] = max(
                    min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]),
                    dist_matrix[i, j]
                )
        
        return dp[n-1, m-1]
    
    def _hausdorff_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Hausdorff距離"""
        # seq1の各点からseq2への最小距離の最大値
        d1 = np.max([np.min(np.linalg.norm(seq2 - p, axis=1)) for p in seq1])
        # seq2の各点からseq1への最小距離の最大値
        d2 = np.max([np.min(np.linalg.norm(seq1 - p, axis=1)) for p in seq2])
        
        return max(d1, d2)


