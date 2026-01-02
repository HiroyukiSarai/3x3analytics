"""
アクション検出モジュール
スクリーン、パス、シュート等のアクションを検出
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .route_analyzer import PlayerRoute


class ActionType(Enum):
    """アクションの種類"""
    SCREEN = "screen"
    PASS = "pass"
    DRIBBLE = "dribble"
    SHOT = "shot"
    REBOUND = "rebound"
    DRIVE = "drive"
    CUT = "cut"
    HANDOFF = "handoff"


@dataclass
class GameAction:
    """ゲームアクションデータ"""
    action_type: ActionType
    frame_start: int
    frame_end: int
    player_ids: List[int]  # 関与した選手ID
    position: Tuple[float, float]  # 発生位置（2Dコート座標）
    confidence: float = 1.0
    description: str = ""
    
    @property
    def duration_frames(self) -> int:
        """アクションの継続フレーム数"""
        return self.frame_end - self.frame_start
    
    def to_dict(self) -> Dict:
        """辞書に変換"""
        return {
            'action_type': self.action_type.value,
            'frame_start': self.frame_start,
            'frame_end': self.frame_end,
            'player_ids': self.player_ids,
            'position': self.position,
            'confidence': self.confidence,
            'description': self.description
        }


class ActionDetector:
    """
    アクション検出器
    ルールベース + 機械学習によるアクション検出
    """
    
    # スクリーン検出パラメータ
    SCREEN_MIN_PROXIMITY = 100  # 選手間の最小距離（ピクセル）
    SCREEN_MAX_SPEED = 30  # スクリーナーの最大速度
    SCREEN_MIN_DURATION = 15  # 最小継続フレーム数
    
    # パス検出パラメータ
    PASS_MAX_BALL_TRAVEL_TIME = 20  # フレーム
    
    # シュート検出パラメータ
    GOAL_POSITION = (750, 1000)  # ゴール位置
    SHOT_ZONE_RADIUS = 300  # シュートゾーン半径
    
    def __init__(
        self,
        fps: float = 30.0
    ):
        """
        Args:
            fps: 動画のFPS
        """
        self.fps = fps
        self.detected_actions: List[GameAction] = []
    
    def detect_actions(
        self,
        tracking_data: List[Dict],
        routes: Optional[Dict[int, PlayerRoute]] = None
    ) -> List[GameAction]:
        """
        トラッキングデータからアクションを検出
        
        Args:
            tracking_data: フレームごとのトラッキングデータ
            routes: 選手ルートデータ
            
        Returns:
            検出されたアクションのリスト
        """
        actions = []
        
        # スクリーン検出
        screens = self._detect_screens(tracking_data)
        actions.extend(screens)
        
        # パス検出
        passes = self._detect_passes(tracking_data)
        actions.extend(passes)
        
        # シュート検出
        shots = self._detect_shots(tracking_data)
        actions.extend(shots)
        
        # ドライブ検出
        if routes:
            drives = self._detect_drives(routes)
            actions.extend(drives)
            
            # カット検出
            cuts = self._detect_cuts(routes)
            actions.extend(cuts)
        
        # 時系列でソート
        actions.sort(key=lambda a: a.frame_start)
        
        self.detected_actions = actions
        return actions
    
    def _detect_screens(
        self,
        tracking_data: List[Dict]
    ) -> List[GameAction]:
        """
        スクリーンを検出
        条件: 2人の選手が近接 + 一方が静止
        """
        screens = []
        screen_candidates = {}  # {(player1, player2): start_frame}
        
        for frame_data in tracking_data:
            frame_id = frame_data.get('frame_id', 0)
            players = frame_data.get('players', [])
            
            if len(players) < 2:
                continue
            
            # 全選手ペアをチェック
            for i, p1 in enumerate(players):
                for p2 in players[i+1:]:
                    # 異なるチームのペアをチェック
                    if p1.get('team') == p2.get('team'):
                        continue
                    
                    pos1 = self._get_position(p1)
                    pos2 = self._get_position(p2)
                    
                    if pos1 is None or pos2 is None:
                        continue
                    
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    
                    # 近接判定
                    if distance < self.SCREEN_MIN_PROXIMITY:
                        pair_key = (p1['track_id'], p2['track_id'])
                        
                        if pair_key not in screen_candidates:
                            screen_candidates[pair_key] = {
                                'start_frame': frame_id,
                                'position': pos1
                            }
                    else:
                        # 離れた場合、スクリーン終了判定
                        pair_key = (p1['track_id'], p2['track_id'])
                        if pair_key in screen_candidates:
                            candidate = screen_candidates[pair_key]
                            duration = frame_id - candidate['start_frame']
                            
                            if duration >= self.SCREEN_MIN_DURATION:
                                action = GameAction(
                                    action_type=ActionType.SCREEN,
                                    frame_start=candidate['start_frame'],
                                    frame_end=frame_id,
                                    player_ids=list(pair_key),
                                    position=candidate['position'],
                                    confidence=0.8,
                                    description="スクリーンプレー"
                                )
                                screens.append(action)
                            
                            del screen_candidates[pair_key]
        
        return screens
    
    def _detect_passes(
        self,
        tracking_data: List[Dict]
    ) -> List[GameAction]:
        """
        パスを検出
        条件: ボール保持者が変わる
        """
        passes = []
        prev_ball_holder = None
        prev_frame = 0
        prev_position = None
        
        for frame_data in tracking_data:
            frame_id = frame_data.get('frame_id', 0)
            
            # ボール保持者を特定
            current_holder = None
            current_position = None
            
            for player in frame_data.get('players', []):
                if player.get('has_ball', False):
                    current_holder = player['track_id']
                    current_position = self._get_position(player)
                    break
            
            # ボール保持者が変わった
            if current_holder is not None and prev_ball_holder is not None:
                if current_holder != prev_ball_holder:
                    # 同じチーム間のパス
                    frame_diff = frame_id - prev_frame
                    
                    if frame_diff <= self.PASS_MAX_BALL_TRAVEL_TIME:
                        action = GameAction(
                            action_type=ActionType.PASS,
                            frame_start=prev_frame,
                            frame_end=frame_id,
                            player_ids=[prev_ball_holder, current_holder],
                            position=prev_position or (0, 0),
                            confidence=0.9,
                            description=f"パス: {prev_ball_holder} → {current_holder}"
                        )
                        passes.append(action)
            
            if current_holder is not None:
                prev_ball_holder = current_holder
                prev_frame = frame_id
                prev_position = current_position
        
        return passes
    
    def _detect_shots(
        self,
        tracking_data: List[Dict]
    ) -> List[GameAction]:
        """
        シュートを検出
        条件: ボールがゴール付近 + ボール速度の変化
        """
        shots = []
        
        for i, frame_data in enumerate(tracking_data):
            frame_id = frame_data.get('frame_id', 0)
            ball = frame_data.get('ball')
            
            if ball is None:
                continue
            
            ball_pos = ball.get('bbox', [0, 0, 0, 0])
            ball_center = ((ball_pos[0] + ball_pos[2]) / 2, 
                          (ball_pos[1] + ball_pos[3]) / 2)
            
            # ゴール付近判定
            distance_to_goal = np.linalg.norm(
                np.array(ball_center) - np.array(self.GOAL_POSITION)
            )
            
            if distance_to_goal < self.SHOT_ZONE_RADIUS:
                # 前後のフレームでボールの動きをチェック
                if i > 0 and i < len(tracking_data) - 1:
                    prev_ball = tracking_data[i-1].get('ball')
                    
                    if prev_ball:
                        prev_pos = prev_ball.get('bbox', [0, 0, 0, 0])
                        prev_center = ((prev_pos[0] + prev_pos[2]) / 2,
                                      (prev_pos[1] + prev_pos[3]) / 2)
                        
                        # ボールがゴール方向に移動
                        if ball_center[1] > prev_center[1]:
                            # ボール保持者を探す
                            shooter_id = None
                            for player in frame_data.get('players', []):
                                if player.get('has_ball', False):
                                    shooter_id = player['track_id']
                                    break
                            
                            action = GameAction(
                                action_type=ActionType.SHOT,
                                frame_start=frame_id - 10,
                                frame_end=frame_id,
                                player_ids=[shooter_id] if shooter_id else [],
                                position=ball_center,
                                confidence=0.7,
                                description="シュート試行"
                            )
                            shots.append(action)
        
        # 重複を除去（近いフレームのシュートを統合）
        return self._merge_nearby_actions(shots, min_gap=30)
    
    def _detect_drives(
        self,
        routes: Dict[int, PlayerRoute]
    ) -> List[GameAction]:
        """
        ドライブを検出
        条件: ゴール方向への高速移動
        """
        drives = []
        
        for track_id, route in routes.items():
            if route.team != 'A':  # 攻撃側のみ
                continue
            
            positions = route.to_array()
            
            if len(positions) < 10:
                continue
            
            # 移動ベクトルを計算
            for i in range(len(positions) - 10):
                segment = positions[i:i+10]
                
                # セグメントの移動方向と速度
                direction = segment[-1] - segment[0]
                distance = np.linalg.norm(direction)
                
                # ゴール方向（Y正方向）への移動
                if direction[1] > 0 and distance > 150:
                    # ゴールへの接近
                    start_dist = np.linalg.norm(segment[0] - np.array(self.GOAL_POSITION))
                    end_dist = np.linalg.norm(segment[-1] - np.array(self.GOAL_POSITION))
                    
                    if end_dist < start_dist:
                        frame_start = int(route.timestamps[i] * self.fps)
                        frame_end = int(route.timestamps[i+9] * self.fps)
                        
                        action = GameAction(
                            action_type=ActionType.DRIVE,
                            frame_start=frame_start,
                            frame_end=frame_end,
                            player_ids=[track_id],
                            position=tuple(segment[0]),
                            confidence=0.8,
                            description=f"ドライブ: {track_id}"
                        )
                        drives.append(action)
        
        return self._merge_nearby_actions(drives, min_gap=20)
    
    def _detect_cuts(
        self,
        routes: Dict[int, PlayerRoute]
    ) -> List[GameAction]:
        """
        カッティングを検出
        条件: 急な方向転換 + ゴール方向への移動
        """
        cuts = []
        
        for track_id, route in routes.items():
            positions = route.to_array()
            
            if len(positions) < 20:
                continue
            
            # 移動ベクトルの角度変化を計算
            for i in range(5, len(positions) - 5):
                prev_vec = positions[i] - positions[i-5]
                next_vec = positions[i+5] - positions[i]
                
                # ベクトルの角度を計算
                if np.linalg.norm(prev_vec) < 1e-6 or np.linalg.norm(next_vec) < 1e-6:
                    continue
                
                cos_angle = np.dot(prev_vec, next_vec) / (
                    np.linalg.norm(prev_vec) * np.linalg.norm(next_vec)
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                # 急な方向転換（60度以上）
                if angle > 60 and np.linalg.norm(next_vec) > 50:
                    frame = int(route.timestamps[i] * self.fps)
                    
                    action = GameAction(
                        action_type=ActionType.CUT,
                        frame_start=frame - 5,
                        frame_end=frame + 5,
                        player_ids=[track_id],
                        position=tuple(positions[i]),
                        confidence=0.7,
                        description=f"カット: {track_id}"
                    )
                    cuts.append(action)
        
        return self._merge_nearby_actions(cuts, min_gap=15)
    
    def _get_position(self, player: Dict) -> Optional[Tuple[float, float]]:
        """選手の位置を取得"""
        court_pos = player.get('court_position')
        if court_pos:
            return tuple(court_pos)
        
        bbox = player.get('bbox')
        if bbox:
            return ((bbox[0] + bbox[2]) / 2, bbox[3])
        
        return None
    
    def _merge_nearby_actions(
        self,
        actions: List[GameAction],
        min_gap: int = 30
    ) -> List[GameAction]:
        """近いフレームのアクションを統合"""
        if not actions:
            return []
        
        actions.sort(key=lambda a: a.frame_start)
        merged = [actions[0]]
        
        for action in actions[1:]:
            if action.frame_start - merged[-1].frame_end < min_gap:
                # 統合
                merged[-1].frame_end = action.frame_end
                merged[-1].player_ids = list(set(
                    merged[-1].player_ids + action.player_ids
                ))
            else:
                merged.append(action)
        
        return merged
    
    def get_action_summary(self) -> Dict[str, int]:
        """アクション種類別のサマリー"""
        summary = {}
        for action_type in ActionType:
            count = sum(1 for a in self.detected_actions 
                       if a.action_type == action_type)
            summary[action_type.value] = count
        
        return summary


