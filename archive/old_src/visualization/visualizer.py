"""
可視化モジュール
トラッキング結果・戦術パターンの可視化
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import colorsys


@dataclass
class ColorScheme:
    """カラースキーム"""
    team_a: Tuple[int, int, int] = (255, 100, 50)   # オレンジ（BGR）
    team_b: Tuple[int, int, int] = (50, 100, 255)   # 青
    ball: Tuple[int, int, int] = (0, 165, 255)       # オレンジ
    court: Tuple[int, int, int] = (34, 139, 34)      # 緑
    line: Tuple[int, int, int] = (255, 255, 255)     # 白
    text: Tuple[int, int, int] = (255, 255, 255)     # 白
    trajectory: Tuple[int, int, int] = (200, 200, 200)  # グレー


class Visualizer:
    """トラッキング結果の可視化"""
    
    def __init__(
        self,
        colors: Optional[ColorScheme] = None,
        show_trajectory: bool = True,
        trajectory_length: int = 30
    ):
        """
        Args:
            colors: カラースキーム
            show_trajectory: 軌跡を表示するか
            trajectory_length: 軌跡の長さ（フレーム数）
        """
        self.colors = colors or ColorScheme()
        self.show_trajectory = show_trajectory
        self.trajectory_length = trajectory_length
    
    def draw_tracking_result(
        self,
        frame: np.ndarray,
        tracks: List,
        ball: Optional[Dict] = None,
        track_history: Optional[Dict] = None
    ) -> np.ndarray:
        """
        トラッキング結果をフレームに描画
        
        Args:
            frame: 元画像
            tracks: トラック情報のリスト
            ball: ボール情報
            track_history: {track_id: [(x, y), ...]}
            
        Returns:
            描画後の画像
        """
        annotated = frame.copy()
        
        # 軌跡を描画
        if self.show_trajectory and track_history:
            annotated = self._draw_trajectories(annotated, tracks, track_history)
        
        # 選手を描画
        for track in tracks:
            annotated = self._draw_player(annotated, track)
        
        # ボールを描画
        if ball is not None:
            annotated = self._draw_ball(annotated, ball)
        
        return annotated
    
    def _draw_player(
        self,
        frame: np.ndarray,
        track
    ) -> np.ndarray:
        """選手を描画"""
        bbox = track.bbox if hasattr(track, 'bbox') else track[:4]
        x1, y1, x2, y2 = map(int, bbox)
        
        track_id = track.track_id if hasattr(track, 'track_id') else int(track[4])
        team = track.team if hasattr(track, 'team') else 'A'
        
        # チーム別の色
        color = self.colors.team_a if team == 'A' else self.colors.team_b
        
        # バウンディングボックス
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # ID表示
        label = f"ID:{track_id}"
        if team:
            label = f"{team}-{track_id}"
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return frame
    
    def _draw_ball(
        self,
        frame: np.ndarray,
        ball: Dict
    ) -> np.ndarray:
        """ボールを描画"""
        bbox = ball.get('bbox', ball.get('position', [0, 0, 0, 0]))
        
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = int(max(x2 - x1, y2 - y1) / 2)
        else:
            cx, cy = int(bbox[0]), int(bbox[1])
            radius = 10
        
        cv2.circle(frame, (cx, cy), radius + 3, self.colors.ball, -1)
        cv2.circle(frame, (cx, cy), radius + 3, (255, 255, 255), 2)
        
        return frame
    
    def _draw_trajectories(
        self,
        frame: np.ndarray,
        tracks: List,
        track_history: Dict
    ) -> np.ndarray:
        """軌跡を描画"""
        for track in tracks:
            track_id = track.track_id if hasattr(track, 'track_id') else int(track[4])
            team = track.team if hasattr(track, 'team') else 'A'
            
            trajectory = track_history.get(track_id, [])
            
            if len(trajectory) < 2:
                continue
            
            # 最新のN点のみ使用
            trajectory = trajectory[-self.trajectory_length:]
            
            # チーム別の色（薄め）
            if team == 'A':
                base_color = self.colors.team_a
            else:
                base_color = self.colors.team_b
            
            # グラデーション描画
            points = np.array(trajectory, dtype=np.int32)
            for i in range(1, len(points)):
                # 古いほど薄く
                alpha = i / len(points)
                color = tuple(int(c * alpha) for c in base_color)
                cv2.line(frame, tuple(points[i-1]), tuple(points[i]), color, 2)
        
        return frame
    
    def create_2d_court(
        self,
        width: int = 1500,
        height: int = 1100
    ) -> np.ndarray:
        """
        2Dコート画像を生成
        """
        court = np.zeros((height, width, 3), dtype=np.uint8)
        court[:] = self.colors.court
        
        line_color = self.colors.line
        
        # コート外枠
        cv2.rectangle(court, (10, 10), (width-10, height-10), line_color, 3)
        
        # ハーフコートライン（トップ）
        cv2.line(court, (10, 10), (width-10, 10), line_color, 3)
        
        # フリースローエリア
        ft_left = width // 2 - 250
        ft_right = width // 2 + 250
        ft_top = 350
        ft_bottom = 750
        cv2.rectangle(court, (ft_left, ft_top), (ft_right, ft_bottom), line_color, 2)
        
        # フリースローライン
        cv2.line(court, (ft_left, 550), (ft_right, 550), line_color, 2)
        
        # 3ポイントアーク
        center = (width // 2, height - 100)
        cv2.ellipse(court, center, (550, 550), 0, 180, 360, line_color, 2)
        
        # ゴール
        goal_pos = (width // 2, height - 100)
        cv2.circle(court, goal_pos, 45, line_color, 2)
        cv2.circle(court, goal_pos, 5, line_color, -1)
        
        # ペイントエリア
        cv2.rectangle(court, (ft_left + 50, ft_bottom), 
                     (ft_right - 50, height - 50), line_color, 2)
        
        return court
    
    def draw_players_on_court(
        self,
        court: np.ndarray,
        player_positions: List[Tuple[float, float, str]],
        ball_position: Optional[Tuple[float, float]] = None,
        trajectories: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        player_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        2Dコート上に選手を描画
        
        Args:
            court: コート画像
            player_positions: [(x, y, team), ...] リスト
            ball_position: ボール位置
            trajectories: {player_id: [(x, y), ...]} 軌跡
            player_ids: 選手IDリスト
        """
        result = court.copy()
        
        # 軌跡を描画
        if trajectories:
            for player_id, traj in trajectories.items():
                if len(traj) > 1:
                    points = np.array(traj, dtype=np.int32)
                    cv2.polylines(result, [points], False, 
                                 self.colors.trajectory, 2)
        
        # 選手を描画
        for i, (x, y, team) in enumerate(player_positions):
            color = self.colors.team_a if team == 'A' else self.colors.team_b
            pos = (int(x), int(y))
            
            # 選手マーカー
            cv2.circle(result, pos, 25, color, -1)
            cv2.circle(result, pos, 25, (255, 255, 255), 2)
            
            # ID表示
            if player_ids and i < len(player_ids):
                player_id = str(player_ids[i])
            else:
                player_id = str(i + 1)
            
            text_size = cv2.getTextSize(player_id, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = pos[0] - text_size[0] // 2
            text_y = pos[1] + text_size[1] // 2
            cv2.putText(result, player_id, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ボールを描画
        if ball_position:
            ball_pos = (int(ball_position[0]), int(ball_position[1]))
            cv2.circle(result, ball_pos, 15, self.colors.ball, -1)
            cv2.circle(result, ball_pos, 15, (255, 255, 255), 2)
        
        return result
    
    def draw_pattern_routes(
        self,
        court: np.ndarray,
        routes: Dict,
        show_arrows: bool = True
    ) -> np.ndarray:
        """
        戦術パターンのルートを描画
        
        Args:
            court: コート画像
            routes: {player_id: PlayerRoute} ルートデータ
            show_arrows: 矢印を表示するか
        """
        result = court.copy()
        
        colors = self._generate_colors(len(routes))
        
        for (player_id, route), color in zip(routes.items(), colors):
            positions = route.positions if hasattr(route, 'positions') else route
            
            if len(positions) < 2:
                continue
            
            points = np.array(positions, dtype=np.int32)
            
            # ルート描画
            cv2.polylines(result, [points], False, color, 3)
            
            # 開始点
            cv2.circle(result, tuple(points[0]), 10, color, -1)
            cv2.circle(result, tuple(points[0]), 10, (255, 255, 255), 2)
            
            # 終了点（矢印）
            if show_arrows and len(points) >= 2:
                self._draw_arrow(result, points[-2], points[-1], color)
        
        return result
    
    def _draw_arrow(
        self,
        frame: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        color: Tuple[int, int, int],
        arrow_size: int = 15
    ):
        """矢印を描画"""
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        
        # 矢印の先端
        p1 = (
            int(end[0] - arrow_size * np.cos(angle - np.pi / 6)),
            int(end[1] - arrow_size * np.sin(angle - np.pi / 6))
        )
        p2 = (
            int(end[0] - arrow_size * np.cos(angle + np.pi / 6)),
            int(end[1] - arrow_size * np.sin(angle + np.pi / 6))
        )
        
        cv2.fillPoly(frame, [np.array([tuple(end), p1, p2])], color)
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """N個の異なる色を生成"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors


class VideoWriter:
    """動画書き出し"""
    
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = 'mp4v'
    ):
        """
        Args:
            output_path: 出力ファイルパス
            fps: フレームレート
            frame_size: (width, height)
            codec: コーデック
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
        self.frame_count = 0
    
    def write(self, frame: np.ndarray):
        """フレームを書き込み"""
        if self.writer is None:
            if self.frame_size is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
            
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                self.frame_size
            )
        
        # リサイズが必要な場合
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        
        self.writer.write(frame)
        self.frame_count += 1
    
    def release(self):
        """リソースを解放"""
        if self.writer is not None:
            self.writer.release()
            print(f"Video saved: {self.output_path} ({self.frame_count} frames)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class AnimationGenerator:
    """戦術パターンのアニメーション生成"""
    
    def __init__(
        self,
        visualizer: Optional[Visualizer] = None,
        fps: float = 30.0
    ):
        """
        Args:
            visualizer: Visualizerインスタンス
            fps: 出力FPS
        """
        self.visualizer = visualizer or Visualizer()
        self.fps = fps
    
    def generate_pattern_animation(
        self,
        routes: Dict,
        output_path: str,
        duration: float = 5.0
    ):
        """
        戦術パターンのアニメーションを生成
        
        Args:
            routes: {player_id: PlayerRoute}
            output_path: 出力パス
            duration: アニメーション時間（秒）
        """
        court = self.visualizer.create_2d_court()
        n_frames = int(duration * self.fps)
        
        with VideoWriter(output_path, self.fps, 
                        (court.shape[1], court.shape[0])) as writer:
            
            for frame_idx in range(n_frames):
                progress = frame_idx / n_frames
                
                # 現在位置を計算
                current_positions = []
                current_trajectories = {}
                
                for player_id, route in routes.items():
                    positions = route.positions if hasattr(route, 'positions') else route
                    team = route.team if hasattr(route, 'team') else 'A'
                    
                    if not positions:
                        continue
                    
                    # 現在のフレームに対応する位置
                    idx = int(progress * (len(positions) - 1))
                    x, y = positions[idx]
                    current_positions.append((x, y, team))
                    
                    # 軌跡（これまでの位置）
                    current_trajectories[player_id] = positions[:idx + 1]
                
                # 描画
                frame = self.visualizer.draw_players_on_court(
                    court.copy(),
                    current_positions,
                    trajectories=current_trajectories
                )
                
                # フレーム番号表示
                cv2.putText(frame, f"Frame: {frame_idx}/{n_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                writer.write(frame)
    
    def generate_play_gif(
        self,
        routes: Dict,
        output_path: str,
        duration: float = 3.0
    ):
        """
        GIFアニメーションを生成
        
        Args:
            routes: {player_id: PlayerRoute}
            output_path: 出力パス（.gif）
            duration: 時間（秒）
        """
        try:
            import imageio
        except ImportError:
            print("Warning: imageio not installed. Run: pip install imageio")
            return
        
        court = self.visualizer.create_2d_court()
        n_frames = int(duration * 10)  # GIFは10fpsで
        
        frames = []
        
        for frame_idx in range(n_frames):
            progress = frame_idx / n_frames
            
            current_positions = []
            current_trajectories = {}
            
            for player_id, route in routes.items():
                positions = route.positions if hasattr(route, 'positions') else route
                team = route.team if hasattr(route, 'team') else 'A'
                
                if not positions:
                    continue
                
                idx = int(progress * (len(positions) - 1))
                x, y = positions[idx]
                current_positions.append((x, y, team))
                current_trajectories[player_id] = positions[:idx + 1]
            
            frame = self.visualizer.draw_players_on_court(
                court.copy(),
                current_positions,
                trajectories=current_trajectories
            )
            
            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        imageio.mimsave(output_path, frames, duration=0.1)
        print(f"GIF saved: {output_path}")


