"""
3x3バスケットボール戦術分析システム
Phase 1: 物体検出 + トラッキング サンプルコード

使用方法:
    python phase1_detection_tracking.py --source video.mp4 --output outputs/
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

# Ultralytics YOLO
from ultralytics import YOLO

# BoxMOT トラッカー
try:
    from boxmot import ByteTrack, BoTSORT
    BOXMOT_AVAILABLE = True
except ImportError:
    BOXMOT_AVAILABLE = False
    print("Warning: boxmot not installed. Run: pip install boxmot")


class BasketballDetector:
    """3x3バスケ用の物体検出器"""
    
    def __init__(self, model_path: str = "yolo11n.pt", device: str = "cuda:0"):
        """
        Args:
            model_path: YOLOモデルのパス
            device: 推論デバイス ('cuda:0' or 'cpu')
        """
        self.model = YOLO(model_path)
        self.device = device
        
        # 検出対象クラス（COCO）
        self.PERSON_CLASS = 0
        self.SPORTS_BALL_CLASS = 32
        
    def detect(self, frame: np.ndarray) -> dict:
        """
        フレームから選手とボールを検出
        
        Args:
            frame: BGR画像 (H, W, 3)
            
        Returns:
            {
                'players': [[x1, y1, x2, y2, conf], ...],
                'ball': [x1, y1, x2, y2, conf] or None
            }
        """
        results = self.model(frame, device=self.device, verbose=False)[0]
        
        players = []
        ball = None
        
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            if cls == self.PERSON_CLASS and conf > 0.5:
                players.append([*xyxy, conf])
            elif cls == self.SPORTS_BALL_CLASS and conf > 0.3:
                if ball is None or conf > ball[4]:
                    ball = [*xyxy, conf]
        
        return {
            'players': np.array(players) if players else np.empty((0, 5)),
            'ball': ball
        }


class BasketballTracker:
    """3x3バスケ用のマルチオブジェクトトラッカー"""
    
    def __init__(self, tracker_type: str = "bytetrack", device: str = "cuda:0"):
        """
        Args:
            tracker_type: 'bytetrack' or 'botsort'
            device: 推論デバイス
        """
        if not BOXMOT_AVAILABLE:
            raise ImportError("boxmot is required. Run: pip install boxmot")
        
        if tracker_type == "bytetrack":
            self.tracker = ByteTrack()
        elif tracker_type == "botsort":
            self.tracker = BoTSORT(
                model_weights=Path("osnet_x0_25_msmt17.pt"),
                device=device,
                fp16=False
            )
        else:
            raise ValueError(f"Unknown tracker: {tracker_type}")
        
        # トラッキング履歴
        self.track_history = defaultdict(list)
        
    def update(self, detections: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        検出結果を更新してトラッキング
        
        Args:
            detections: [[x1, y1, x2, y2, conf], ...] (N, 5)
            frame: BGR画像
            
        Returns:
            tracks: [[x1, y1, x2, y2, track_id, conf, cls, idx], ...] (M, 8)
        """
        if len(detections) == 0:
            return np.empty((0, 8))
        
        # BoxMOT形式に変換 (x1, y1, x2, y2, conf, cls)
        dets = np.zeros((len(detections), 6))
        dets[:, :5] = detections
        dets[:, 5] = 0  # cls = person
        
        tracks = self.tracker.update(dets, frame)
        
        # トラッキング履歴を更新
        for track in tracks:
            track_id = int(track[4])
            cx = (track[0] + track[2]) / 2
            cy = (track[1] + track[3]) / 2
            self.track_history[track_id].append((cx, cy))
            
            # 履歴の長さを制限（最新90フレーム = 3秒@30fps）
            if len(self.track_history[track_id]) > 90:
                self.track_history[track_id].pop(0)
        
        return tracks
    
    def get_trajectory(self, track_id: int) -> list:
        """指定IDの軌跡を取得"""
        return self.track_history.get(track_id, [])


class Visualizer:
    """検出・トラッキング結果の可視化"""
    
    # チームカラー（仮）
    TEAM_A_COLOR = (0, 255, 0)   # 緑
    TEAM_B_COLOR = (0, 0, 255)   # 赤
    BALL_COLOR = (255, 165, 0)   # オレンジ
    
    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: np.ndarray, 
                    track_history: dict) -> np.ndarray:
        """トラッキング結果を描画"""
        annotated = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            
            # バウンディングボックス
            color = Visualizer.TEAM_A_COLOR  # TODO: チーム分類
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # ID表示
            label = f"ID:{track_id}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 軌跡描画
            trajectory = track_history.get(track_id, [])
            if len(trajectory) > 1:
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(annotated, [points], False, color, 2)
        
        return annotated
    
    @staticmethod
    def draw_ball(frame: np.ndarray, ball: list) -> np.ndarray:
        """ボールを描画"""
        if ball is None:
            return frame
        
        annotated = frame.copy()
        x1, y1, x2, y2, conf = ball
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        radius = int(max(x2 - x1, y2 - y1) / 2)
        
        cv2.circle(annotated, (cx, cy), radius, Visualizer.BALL_COLOR, 2)
        cv2.putText(annotated, f"Ball:{conf:.2f}", (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, Visualizer.BALL_COLOR, 2)
        
        return annotated


def process_video(source: str, output_dir: str, 
                  tracker_type: str = "bytetrack",
                  show: bool = False):
    """
    動画を処理してトラッキング結果を出力
    
    Args:
        source: 入力動画パス
        output_dir: 出力ディレクトリ
        tracker_type: トラッカー種類
        show: リアルタイム表示するか
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初期化
    detector = BasketballDetector()
    tracker = BasketballTracker(tracker_type=tracker_type)
    
    # 動画読み込み
    cap = cv2.VideoCapture(source)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 出力動画
    output_video = str(output_path / "tracked_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # トラッキングデータ保存用
    tracking_data = []
    
    print(f"Processing: {source}")
    print(f"  FPS: {fps}, Size: {width}x{height}, Frames: {total_frames}")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 検出
        detections = detector.detect(frame)
        
        # トラッキング
        tracks = tracker.update(detections['players'], frame)
        
        # 可視化
        annotated = Visualizer.draw_tracks(frame, tracks, tracker.track_history)
        annotated = Visualizer.draw_ball(annotated, detections['ball'])
        
        # フレーム番号表示
        cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存
        writer.write(annotated)
        
        # トラッキングデータ記録
        frame_data = {
            'frame_id': frame_idx,
            'timestamp': frame_idx / fps,
            'players': []
        }
        for track in tracks:
            frame_data['players'].append({
                'track_id': int(track[4]),
                'bbox': track[:4].tolist(),
                'confidence': float(track[5])
            })
        if detections['ball'] is not None:
            frame_data['ball'] = {
                'bbox': detections['ball'][:4],
                'confidence': detections['ball'][4]
            }
        tracking_data.append(frame_data)
        
        # 表示
        if show:
            cv2.imshow("Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 進捗表示
        if frame_idx % 100 == 0:
            print(f"  Processing frame {frame_idx}/{total_frames}")
        
        frame_idx += 1
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # JSONで保存
    import json
    json_path = output_path / "tracking_data.json"
    with open(json_path, 'w') as f:
        json.dump(tracking_data, f, indent=2)
    
    print(f"\nOutput saved:")
    print(f"  Video: {output_video}")
    print(f"  Data: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="3x3 Basketball Detection & Tracking")
    parser.add_argument("--source", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--tracker", type=str, default="bytetrack", 
                       choices=["bytetrack", "botsort"], help="Tracker type")
    parser.add_argument("--show", action="store_true", help="Show realtime")
    
    args = parser.parse_args()
    
    process_video(
        source=args.source,
        output_dir=args.output,
        tracker_type=args.tracker,
        show=args.show
    )


if __name__ == "__main__":
    main()
