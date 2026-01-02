"""
動画からフレームを抽出するスクリプト
TrackID3x3の動画から学習用フレームを生成

使用方法:
    python scripts/extract_frames.py \
        --input external_repos/TrackID3x3/videos \
        --output data/yolo_dataset/images \
        --fps 5
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 5.0,
    max_frames: int = None
):
    """
    動画からフレームを抽出
    
    Args:
        video_path: 動画ファイルパス
        output_dir: 出力ディレクトリ
        target_fps: 抽出するFPS
        max_frames: 最大フレーム数（Noneで無制限）
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Warning: Cannot open {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # フレームスキップ間隔
    frame_interval = max(1, int(video_fps / target_fps))
    
    video_name = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームをスキップ
        if frame_count % frame_interval == 0:
            # ファイル名はラベルファイルと一致させる
            output_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
            output_path = output_dir / output_filename
            
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return saved_count


def match_images_to_labels(images_dir: Path, labels_dir: Path):
    """
    画像ファイルとラベルファイルをマッチング
    存在しない画像のラベルは削除、ラベルのない画像は移動しない
    """
    # ラベルファイル一覧
    label_files = set(f.stem for f in labels_dir.glob('*.txt'))
    
    # 画像ファイル一覧
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    matched = 0
    for img_file in image_files:
        if img_file.stem not in label_files:
            # ラベルがない画像は削除
            img_file.unlink()
        else:
            matched += 1
    
    # 画像がないラベルを削除
    image_stems = set(f.stem for f in images_dir.glob('*.jpg'))
    for label_file in labels_dir.glob('*.txt'):
        if label_file.stem not in image_stems:
            label_file.unlink()
    
    return matched


def process_videos(
    input_dir: Path,
    output_dir: Path,
    target_fps: float = 5.0,
    subsets: list = None
):
    """
    複数の動画を処理
    """
    if subsets is None:
        subsets = ['Indoor', 'Outdoor']
    
    total_frames = 0
    
    for subset in subsets:
        subset_dir = input_dir / subset
        
        if not subset_dir.exists():
            print(f"Warning: {subset_dir} does not exist")
            continue
        
        print(f"\nProcessing {subset}...")
        
        video_files = list(subset_dir.glob('*.mp4')) + list(subset_dir.glob('*.mov'))
        
        for video_file in tqdm(video_files, desc=f"Extracting {subset}"):
            # train/valの振り分けはラベルと同じにする
            saved = extract_frames_from_video(
                video_file,
                output_dir / 'temp',
                target_fps
            )
            total_frames += saved
    
    print(f"\nTotal frames extracted: {total_frames}")
    
    # ラベルとマッチング
    print("\nMatching images to labels...")
    
    # trainとvalに分配
    temp_dir = output_dir / 'temp'
    train_labels = output_dir / 'labels' / 'train'
    val_labels = output_dir / 'labels' / 'val'
    train_images = output_dir / 'images' / 'train'
    val_images = output_dir / 'images' / 'val'
    
    train_images.mkdir(parents=True, exist_ok=True)
    val_images.mkdir(parents=True, exist_ok=True)
    
    # trainラベルに対応する画像を移動
    train_matched = 0
    for label_file in train_labels.glob('*.txt'):
        img_name = label_file.stem + '.jpg'
        src = temp_dir / img_name
        if src.exists():
            shutil.move(str(src), str(train_images / img_name))
            train_matched += 1
    
    # valラベルに対応する画像を移動
    val_matched = 0
    for label_file in val_labels.glob('*.txt'):
        img_name = label_file.stem + '.jpg'
        src = temp_dir / img_name
        if src.exists():
            shutil.move(str(src), str(val_images / img_name))
            val_matched += 1
    
    # 一時ディレクトリ削除
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    print(f"Train images: {train_matched}")
    print(f"Val images: {val_matched}")
    
    return train_matched, val_matched


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from TrackID3x3 videos"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='external_repos/TrackID3x3/videos',
        help='Input videos directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/yolo_dataset',
        help='Output directory'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=5.0,
        help='Target FPS for frame extraction'
    )
    parser.add_argument(
        '--subsets',
        nargs='+',
        default=['Indoor', 'Outdoor'],
        help='Video subsets to process'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        print("\nPlease download videos from:")
        print("  https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX")
        print(f"\nAnd place them in: {input_dir}")
        return
    
    process_videos(input_dir, output_dir, args.fps, args.subsets)


if __name__ == "__main__":
    main()


