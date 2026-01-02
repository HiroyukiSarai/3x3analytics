"""
戦術パターン分析モジュール
ルートのクラスタリングと戦術パターン分類
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .route_analyzer import PlayerRoute, RouteAnalyzer


@dataclass
class TacticalPattern:
    """
    戦術パターンデータ
    """
    pattern_id: str
    pattern_name: str
    description: str = ""
    occurrences: int = 0
    success_count: int = 0
    routes: List[Dict[int, PlayerRoute]] = field(default_factory=list)
    centroid_routes: Optional[Dict[int, np.ndarray]] = None
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.occurrences == 0:
            return 0.0
        return self.success_count / self.occurrences
    
    def to_dict(self) -> Dict:
        """辞書に変換（JSON保存用）"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'description': self.description,
            'occurrences': self.occurrences,
            'success_count': self.success_count,
            'success_rate': self.success_rate,
            'avg_duration_sec': self._avg_duration()
        }
    
    def _avg_duration(self) -> float:
        """平均所要時間"""
        durations = []
        for play_routes in self.routes:
            for route in play_routes.values():
                if route.timestamps:
                    durations.append(route.timestamps[-1] - route.timestamps[0])
                    break
        return np.mean(durations) if durations else 0.0


class PatternAnalyzer:
    """
    戦術パターン分析器
    DTW + K-meansによるルートクラスタリング
    """
    
    # 定義済み戦術パターン
    KNOWN_PATTERNS = {
        'pick_and_roll': 'ピック＆ロール',
        'pick_and_pop': 'ピック＆ポップ',
        'cutting': 'カッティング',
        'drive_and_kick': 'ドライブ＆キックアウト',
        'isolation': 'アイソレーション',
        'off_ball_screen': 'オフボールスクリーン',
        'give_and_go': 'ギブ＆ゴー',
        'hand_off': 'ハンドオフ'
    }
    
    def __init__(
        self,
        n_clusters: int = 8,
        route_analyzer: Optional[RouteAnalyzer] = None
    ):
        """
        Args:
            n_clusters: クラスタ数（戦術パターン数）
            route_analyzer: ルート分析器
        """
        self.n_clusters = n_clusters
        self.route_analyzer = route_analyzer or RouteAnalyzer()
        self.patterns: Dict[str, TacticalPattern] = {}
        self.cluster_model = None
        self.is_trained = False
    
    def analyze_plays(
        self,
        plays: List[Dict[int, PlayerRoute]],
        success_flags: Optional[List[bool]] = None
    ) -> List[TacticalPattern]:
        """
        複数のプレーを分析してパターンを抽出
        
        Args:
            plays: プレーのリスト（各プレーは{track_id: PlayerRoute}）
            success_flags: 各プレーの成功/失敗フラグ
            
        Returns:
            抽出されたパターンのリスト
        """
        if len(plays) < self.n_clusters:
            print(f"Warning: Not enough plays ({len(plays)}) for {self.n_clusters} clusters")
            self.n_clusters = max(2, len(plays) // 2)
        
        # ルートを特徴ベクトルに変換
        features = self._extract_features(plays)
        
        if len(features) == 0:
            return []
        
        # クラスタリング
        cluster_labels = self._cluster_routes(features)
        
        # パターンを生成
        patterns = self._create_patterns(plays, cluster_labels, success_flags)
        
        self.patterns = {p.pattern_id: p for p in patterns}
        self.is_trained = True
        
        return patterns
    
    def _extract_features(
        self,
        plays: List[Dict[int, PlayerRoute]]
    ) -> np.ndarray:
        """
        プレーから特徴ベクトルを抽出
        
        各プレーの特徴:
        - 各選手のリサンプリングされたルート（正規化済み）
        - 選手間の相対位置
        - 移動距離・速度
        """
        features_list = []
        
        for play in plays:
            if not play:
                continue
            
            play_features = []
            
            # 攻撃側選手のルートを抽出（最大3人）
            attacking_routes = [r for r in play.values() if r.team == 'A'][:3]
            
            for route in attacking_routes:
                # 正規化してリサンプリング
                normalized = route.normalize().resample(20)
                route_array = normalized.to_array().flatten()
                play_features.extend(route_array)
                
                # 追加特徴
                play_features.append(normalized.total_distance)
                play_features.append(normalized.average_speed)
            
            # パディング（3人未満の場合）
            expected_length = 3 * (20 * 2 + 2)  # 3選手 × (20点×2座標 + 距離 + 速度)
            while len(play_features) < expected_length:
                play_features.append(0)
            
            features_list.append(play_features[:expected_length])
        
        return np.array(features_list) if features_list else np.array([])
    
    def _cluster_routes(self, features: np.ndarray) -> np.ndarray:
        """
        特徴ベクトルをクラスタリング
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 標準化
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # K-means
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(features_scaled)
            
            self.cluster_model = kmeans
            self.scaler = scaler
            
            return labels
            
        except ImportError:
            print("Warning: sklearn not available, using random clustering")
            return np.random.randint(0, self.n_clusters, len(features))
    
    def _create_patterns(
        self,
        plays: List[Dict[int, PlayerRoute]],
        cluster_labels: np.ndarray,
        success_flags: Optional[List[bool]] = None
    ) -> List[TacticalPattern]:
        """
        クラスタからパターンを生成
        """
        patterns = []
        
        for cluster_id in range(self.n_clusters):
            # このクラスタに属するプレーを抽出
            cluster_plays = [
                plays[i] for i in range(len(plays))
                if cluster_labels[i] == cluster_id
            ]
            
            if not cluster_plays:
                continue
            
            # 成功数をカウント
            success_count = 0
            if success_flags:
                for i, label in enumerate(cluster_labels):
                    if label == cluster_id and success_flags[i]:
                        success_count += 1
            
            # パターン名を推定（または仮名）
            pattern_name = self._estimate_pattern_name(cluster_plays)
            
            pattern = TacticalPattern(
                pattern_id=f"pattern_{cluster_id:02d}",
                pattern_name=pattern_name,
                description=self._generate_description(cluster_plays),
                occurrences=len(cluster_plays),
                success_count=success_count,
                routes=cluster_plays
            )
            
            patterns.append(pattern)
        
        # 出現回数でソート
        patterns.sort(key=lambda p: p.occurrences, reverse=True)
        
        return patterns
    
    def _estimate_pattern_name(
        self,
        plays: List[Dict[int, PlayerRoute]]
    ) -> str:
        """
        プレーの特徴からパターン名を推定
        """
        if not plays:
            return "Unknown"
        
        # 特徴を分析
        avg_distances = []
        has_screen = False
        has_drive = False
        
        for play in plays:
            routes = list(play.values())
            
            for route in routes:
                avg_distances.append(route.total_distance)
                
                # スクリーン判定（ある選手が停止している）
                if route.average_speed < 50:  # 閾値
                    has_screen = True
                
                # ドライブ判定（ゴール方向への高速移動）
                if route.total_distance > 500 and route.average_speed > 200:
                    has_drive = True
        
        avg_distance = np.mean(avg_distances) if avg_distances else 0
        
        # パターン推定
        if has_screen and has_drive:
            return "ピック＆ロール"
        elif has_screen and not has_drive:
            return "ピック＆ポップ"
        elif avg_distance > 600:
            return "カッティング"
        elif avg_distance < 200:
            return "アイソレーション"
        else:
            return "その他"
    
    def _generate_description(
        self,
        plays: List[Dict[int, PlayerRoute]]
    ) -> str:
        """
        パターンの説明文を生成
        """
        if not plays:
            return ""
        
        # 統計情報を計算
        avg_players = np.mean([len(p) for p in plays])
        
        distances = []
        for play in plays:
            for route in play.values():
                distances.append(route.total_distance)
        
        avg_distance = np.mean(distances) if distances else 0
        
        return f"平均{avg_players:.1f}人関与、平均移動距離{avg_distance:.0f}px"
    
    def classify_play(
        self,
        play: Dict[int, PlayerRoute]
    ) -> Tuple[str, float]:
        """
        新しいプレーをパターンに分類
        
        Args:
            play: プレーのルートデータ
            
        Returns:
            (パターンID, 信頼度)
        """
        if not self.is_trained or self.cluster_model is None:
            return ("unknown", 0.0)
        
        # 特徴抽出
        features = self._extract_features([play])
        
        if len(features) == 0:
            return ("unknown", 0.0)
        
        # スケーリング
        features_scaled = self.scaler.transform(features)
        
        # 予測
        cluster_id = self.cluster_model.predict(features_scaled)[0]
        
        # 信頼度（クラスタ中心との距離から計算）
        center = self.cluster_model.cluster_centers_[cluster_id]
        distance = np.linalg.norm(features_scaled[0] - center)
        confidence = 1.0 / (1.0 + distance)
        
        pattern_id = f"pattern_{cluster_id:02d}"
        
        return (pattern_id, confidence)
    
    def get_pattern_statistics(self) -> Dict[str, Dict]:
        """
        全パターンの統計情報を取得
        """
        stats = {}
        for pattern_id, pattern in self.patterns.items():
            stats[pattern_id] = pattern.to_dict()
        
        return stats
    
    def save_patterns(self, filepath: str):
        """パターンをJSONで保存"""
        data = {
            'n_clusters': self.n_clusters,
            'patterns': [p.to_dict() for p in self.patterns.values()]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_patterns(self, filepath: str):
        """パターンをJSONから読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.n_clusters = data.get('n_clusters', 8)
        
        self.patterns = {}
        for p_data in data.get('patterns', []):
            pattern = TacticalPattern(
                pattern_id=p_data['pattern_id'],
                pattern_name=p_data['pattern_name'],
                description=p_data.get('description', ''),
                occurrences=p_data.get('occurrences', 0),
                success_count=p_data.get('success_count', 0)
            )
            self.patterns[pattern.pattern_id] = pattern


