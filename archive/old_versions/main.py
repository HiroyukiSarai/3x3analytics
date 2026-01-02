"""
3x3バスケットボール戦術分析システム
メインパイプライン

使用方法:
    python main.py --source video.mp4 --output outputs/
    python main.py --source video.mp4 --output outputs/ --analyze-patterns
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm

# 内部モジュールのインポート
from src.detection.detector import BasketballDetector, Detection
from src.tracking.tracker import BasketballTracker, Track
from src.homography.court_detector import CourtDetector, CourtKeypoints
from src.homography.transformer import HomographyTransformer, CourtVisualizer
from src.pose.estimator import PoseEstimator, TeamClassifier, Skeleton
from src.analysis.route_analyzer import RouteAnalyzer, PlayerRoute
from src.analysis.pattern_analyzer import PatternAnalyzer, TacticalPattern
from src.analysis.action_detector import ActionDetector, GameAction
from src.visualization.visualizer import Visualizer, VideoWriter
from src.visualization.report_generator import ReportGenerator


class Basketball3x3Pipeline:
    """
    3x3バスケットボール分析パイプライン
    全Phaseを統合した完全な分析パイプライン
    """
    
    def __init__(
        self,
        detector_model: str = "yolo11n.pt",
        pose_model: str = "yolo11n-pose.pt",
        tracker_type: str = "bytetrack",
        device: str = "cuda:0",
        output_dir: str = "outputs"
    ):
        """
        Args:
            detector_model: 物体検出モデルパス
            pose_model: 骨格推定モデルパス
            tracker_type: トラッカー種類
            device: 推論デバイス
            output_dir: 出力ディレクトリ
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("3x3 Basketball Tactical Analysis System")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Output: {output_dir}")
        print()
        
        # Phase 1: 検出・トラッキング
        print("[Phase 1] 物体検出・トラッキングモジュール初期化...")
        self.detector = BasketballDetector(
            model_path=detector_model,
            device=device
        )
        self.tracker = BasketballTracker(
            tracker_type=tracker_type,
            device=device
        )
        
        # Phase 2: コート検出・座標変換
        print("[Phase 2] コート検出・座標変換モジュール初期化...")
        self.court_detector = CourtDetector(device=device)
        self.transformer = HomographyTransformer()
        self.court_visualizer = CourtVisualizer()
        
        # Phase 3: 骨格推定・チーム分類
        print("[Phase 3] 骨格推定・チーム分類モジュール初期化...")
        try:
            self.pose_estimator = PoseEstimator(
                model_path=pose_model,
                device=device
            )
        except Exception as e:
            print(f"  Warning: Pose estimator not available: {e}")
            self.pose_estimator = None
        
        self.team_classifier = TeamClassifier()
        
        # Phase 4: 戦術分析
        print("[Phase 4] 戦術パターン分析モジュール初期化...")
        self.route_analyzer = RouteAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()
        self.action_detector = ActionDetector()
        
        # Phase 5: 可視化・レポート
        print("[Phase 5] 可視化・レポートモジュール初期化...")
        self.visualizer = Visualizer()
        self.report_generator = ReportGenerator(
            output_dir=str(self.output_dir / "reports")
        )
        
        # データ保存用
        self.tracking_data: List[Dict] = []
        self.scoring_frames: List[int] = []
        
        print()
        print("初期化完了!")
        print("=" * 60)
    
    def process_video(
        self,
        source: str,
        analyze_patterns: bool = True,
        generate_report: bool = True,
        show_preview: bool = False,
        save_video: bool = True
    ) -> Dict:
        """
        動画を処理して完全な分析を実行
        
        Args:
            source: 入力動画パス
            analyze_patterns: 戦術パターン分析を実行するか
            generate_report: レポートを生成するか
            show_preview: プレビューを表示するか
            save_video: 動画を保存するか
            
        Returns:
            分析結果
        """
        print(f"\n処理開始: {source}")
        
        # 動画情報取得
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  FPS: {fps:.1f}")
        print(f"  解像度: {width}x{height}")
        print(f"  総フレーム数: {total_frames}")
        
        # 出力動画
        video_writer = None
        if save_video:
            output_video_path = self.output_dir / "tracked_output.mp4"
            video_writer = VideoWriter(
                str(output_video_path),
                fps=fps,
                frame_size=(width, height)
            )
        
        # 2Dコート画像
        court_img = self.visualizer.create_2d_court()
        
        # チーム分類の学習用サンプル収集
        team_samples = []
        
        # フレーム処理
        self.tracking_data = []
        frame_idx = 0
        
        print("\n[Processing] フレーム処理中...")
        
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Phase 1: 検出
                detections = self.detector.detect(frame)
                
                # Phase 1: トラッキング
                tracks = self.tracker.update(
                    detections['raw_detections'],
                    frame
                )
                
                # Phase 3: チーム分類サンプル収集（最初の100フレーム）
                if frame_idx < 100:
                    for track in tracks:
                        bbox = track.bbox if hasattr(track, 'bbox') else track[:4]
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            team_samples.append(crop)
                
                # フレームデータ記録
                frame_data = self._create_frame_data(
                    frame_idx, fps, tracks, detections['ball']
                )
                self.tracking_data.append(frame_data)
                
                # 可視化
                annotated = self.visualizer.draw_tracking_result(
                    frame, tracks, detections, self.tracker.track_history
                )
                
                # フレーム情報表示
                cv2.putText(annotated, f"Frame: {frame_idx}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(annotated, f"Players: {len(tracks)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 動画保存
                if video_writer:
                    video_writer.write(annotated)
                
                # プレビュー
                if show_preview:
                    cv2.imshow("3x3 Basketball Analysis", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # チーム分類の学習
        if len(team_samples) > 6:
            print("\n[Team Classification] チーム分類モデル学習中...")
            self.team_classifier.train(team_samples[:50])
        
        # トラッキングデータ保存
        tracking_json_path = self.output_dir / "tracking_data.json"
        with open(tracking_json_path, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        print(f"\nトラッキングデータ保存: {tracking_json_path}")
        
        # 戦術パターン分析
        patterns = []
        actions = []
        
        if analyze_patterns:
            print("\n[Pattern Analysis] 戦術パターン分析中...")
            patterns, actions = self._analyze_patterns()
        
        # レポート生成
        if generate_report:
            print("\n[Report Generation] レポート生成中...")
            self.report_generator.generate_full_report(
                self.tracking_data,
                patterns,
                actions,
                game_info={
                    "source": str(source),
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "total_frames": total_frames
                }
            )
        
        # 結果サマリー
        results = {
            "total_frames": len(self.tracking_data),
            "total_tracks": self.tracker.total_tracks,
            "patterns_detected": len(patterns),
            "actions_detected": len(actions),
            "output_dir": str(self.output_dir)
        }
        
        print("\n" + "=" * 60)
        print("処理完了!")
        print(f"  総フレーム数: {results['total_frames']}")
        print(f"  トラッキング数: {results['total_tracks']}")
        print(f"  戦術パターン: {results['patterns_detected']}種類")
        print(f"  アクション検出: {results['actions_detected']}回")
        print(f"  出力先: {results['output_dir']}")
        print("=" * 60)
        
        return results
    
    def _create_frame_data(
        self,
        frame_idx: int,
        fps: float,
        tracks: List,
        ball: Optional[Detection]
    ) -> Dict:
        """フレームデータを作成"""
        frame_data = {
            "frame_id": frame_idx,
            "timestamp": frame_idx / fps,
            "players": []
        }
        
        for track in tracks:
            bbox = track.bbox if hasattr(track, 'bbox') else track[:4]
            track_id = track.track_id if hasattr(track, 'track_id') else int(track[4])
            team = track.team if hasattr(track, 'team') else 'A'
            
            player_data = {
                "track_id": track_id,
                "team": team,
                "bbox": list(map(float, bbox)),
                "confidence": float(track.confidence) if hasattr(track, 'confidence') else 1.0
            }
            
            # 足元座標（2Dコート座標変換用）
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]
            player_data["foot_position"] = [foot_x, foot_y]
            
            # ホモグラフィ変換（設定済みの場合）
            if self.transformer.H is not None:
                result = self.transformer.transform_point((foot_x, foot_y))
                if result.is_valid:
                    player_data["court_position"] = list(result.court_position)
            
            frame_data["players"].append(player_data)
        
        # ボール情報
        if ball is not None:
            frame_data["ball"] = {
                "bbox": list(map(float, ball.bbox)),
                "confidence": ball.confidence
            }
        
        return frame_data
    
    def _analyze_patterns(self) -> tuple:
        """戦術パターン分析を実行"""
        # ルート抽出
        routes = self.route_analyzer.extract_routes_from_tracking(
            self.tracking_data,
            start_frame=0,
            end_frame=len(self.tracking_data) - 1
        )
        
        # プレーを分割（10秒ごと）
        fps = 30.0
        segment_frames = int(10 * fps)
        plays = []
        
        for start in range(0, len(self.tracking_data), segment_frames):
            end = min(start + segment_frames, len(self.tracking_data))
            segment_routes = self.route_analyzer.extract_routes_from_tracking(
                self.tracking_data, start, end
            )
            if segment_routes:
                plays.append(segment_routes)
        
        # パターン分析
        patterns = []
        if len(plays) >= 3:
            patterns = self.pattern_analyzer.analyze_plays(plays)
            
            # パターン保存
            pattern_path = self.output_dir / "patterns.json"
            self.pattern_analyzer.save_patterns(str(pattern_path))
            print(f"  パターン保存: {pattern_path}")
        
        # アクション検出
        actions = self.action_detector.detect_actions(self.tracking_data, routes)
        
        # アクション統計
        action_summary = self.action_detector.get_action_summary()
        print(f"  アクション検出: {action_summary}")
        
        return patterns, actions
    
    def setup_homography(
        self,
        frame: np.ndarray,
        manual_points: Optional[List] = None
    ) -> bool:
        """
        ホモグラフィ変換をセットアップ
        
        Args:
            frame: 基準フレーム
            manual_points: 手動キーポイント（指定しない場合は自動検出）
            
        Returns:
            成功したかどうか
        """
        if manual_points:
            keypoints = self.court_detector.set_manual_keypoints(manual_points)
        else:
            keypoints = self.court_detector.detect(frame)
        
        success = self.transformer.compute_from_keypoints(keypoints)
        
        if success:
            # ホモグラフィ行列を保存
            h_path = self.output_dir / "homography.npy"
            self.transformer.save(str(h_path))
            print(f"ホモグラフィ行列保存: {h_path}")
        
        return success
    
    def load_homography(self, filepath: str) -> bool:
        """保存されたホモグラフィ行列を読み込み"""
        return self.transformer.load(filepath)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="3x3 Basketball Tactical Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="入力動画パス"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="出力ディレクトリ"
    )
    
    parser.add_argument(
        "--detector-model",
        type=str,
        default="yolo11n.pt",
        help="物体検出モデルパス"
    )
    
    parser.add_argument(
        "--pose-model",
        type=str,
        default="yolo11n-pose.pt",
        help="骨格推定モデルパス"
    )
    
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort", "deepocsort"],
        help="トラッカー種類"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="推論デバイス"
    )
    
    parser.add_argument(
        "--analyze-patterns",
        action="store_true",
        help="戦術パターン分析を実行"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="レポートを生成"
    )
    
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="プレビューを表示"
    )
    
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="動画を保存しない"
    )
    
    parser.add_argument(
        "--homography",
        type=str,
        default=None,
        help="ホモグラフィ行列ファイルパス"
    )
    
    args = parser.parse_args()
    
    # パイプライン初期化
    pipeline = Basketball3x3Pipeline(
        detector_model=args.detector_model,
        pose_model=args.pose_model,
        tracker_type=args.tracker,
        device=args.device,
        output_dir=args.output
    )
    
    # ホモグラフィ読み込み
    if args.homography:
        if pipeline.load_homography(args.homography):
            print(f"ホモグラフィ行列読み込み: {args.homography}")
    
    # 処理実行
    results = pipeline.process_video(
        source=args.source,
        analyze_patterns=args.analyze_patterns,
        generate_report=args.generate_report,
        show_preview=args.show_preview,
        save_video=not args.no_save_video
    )
    
    return results


if __name__ == "__main__":
    main()


