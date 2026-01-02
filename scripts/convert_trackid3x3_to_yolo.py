"""
TrackID3x3データセットをYOLO形式に変換するスクリプト

MOT形式 (frame_id, track_id, x, y, w, h, conf, class, visibility, ?)
  → YOLO形式 (class x_center y_center width height) [0-1正規化]

使用方法:
    python scripts/convert_trackid3x3_to_yolo.py --input external_repos/TrackID3x3 --output data/yolo_dataset
"""

import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
import json
import random


def parse_mot_annotation(line: str) -> dict:
    """
    MOT形式のアノテーションをパース
    形式: frame_id, track_id, x, y, w, h, conf, class, visibility, ?
    """
    parts = line.strip().split(',')
    if len(parts) < 6:
        return None
    
    return {
        'frame_id': int(parts[0]),
        'track_id': int(parts[1]),
        'x': float(parts[2]),
        'y': float(parts[3]),
        'w': float(parts[4]),
        'h': float(parts[5]),
        'conf': float(parts[6]) if len(parts) > 6 and parts[6] != '-1' else 1.0,
        'class_id': int(parts[7]) if len(parts) > 7 and parts[7] != '-1' else 0,
        'visibility': float(parts[8]) if len(parts) > 8 and parts[8] != '-1' else 1.0
    }


def mot_to_yolo(annotation: dict, img_width: int, img_height: int) -> str:
    """
    MOT形式をYOLO形式に変換
    
    MOT: x, y, w, h (左上座標 + サイズ、ピクセル)
    YOLO: class x_center y_center width height (中心座標 + サイズ、0-1正規化)
    """
    # 中心座標を計算
    x_center = (annotation['x'] + annotation['w'] / 2) / img_width
    y_center = (annotation['y'] + annotation['h'] / 2) / img_height
    width = annotation['w'] / img_width
    height = annotation['h'] / img_height
    
    # 範囲チェック
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    # クラス0 = person (選手)
    class_id = 0
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_mot_file(
    mot_file: Path,
    output_labels_dir: Path,
    video_name: str,
    img_width: int = 1920,
    img_height: int = 1080
) -> dict:
    """
    MOTファイルを処理してYOLO形式のラベルファイルを生成
    """
    frame_annotations = defaultdict(list)
    
    with open(mot_file, 'r') as f:
        for line in f:
            annotation = parse_mot_annotation(line)
            if annotation is None:
                continue
            
            # 可視性が低いものは除外
            if annotation['visibility'] < 0.3:
                continue
            
            yolo_line = mot_to_yolo(annotation, img_width, img_height)
            frame_annotations[annotation['frame_id']].append(yolo_line)
    
    # フレームごとにラベルファイルを生成
    for frame_id, annotations in frame_annotations.items():
        label_filename = f"{video_name}_frame_{frame_id:06d}.txt"
        label_path = output_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
    
    return {
        'video_name': video_name,
        'total_frames': len(frame_annotations),
        'total_annotations': sum(len(a) for a in frame_annotations.values())
    }


def create_dataset_yaml(output_dir: Path, class_names: list = None):
    """
    YOLO学習用のdata.yamlを生成
    """
    if class_names is None:
        class_names = ['player']  # 3x3バスケでは選手のみ
    
    yaml_content = f"""# TrackID3x3 Dataset for YOLO11 Fine-tuning
# 3x3 Basketball Player Detection

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: {len(class_names)}
names: {class_names}

# Training settings (recommended for 3x3 basketball)
# - High augmentation for different court conditions
# - Consider mosaic and mixup for player occlusion handling
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created: {yaml_path}")
    return yaml_path


def split_dataset(
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """
    データセットをtrain/valに分割
    """
    random.seed(seed)
    
    # ラベルファイル一覧
    label_files = list(labels_dir.glob('*.txt'))
    random.shuffle(label_files)
    
    # 分割
    split_idx = int(len(label_files) * train_ratio)
    train_files = label_files[:split_idx]
    val_files = label_files[split_idx:]
    
    # ディレクトリ作成
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    
    # ファイルをコピー
    for f in train_files:
        shutil.copy(f, output_dir / 'labels' / 'train' / f.name)
    
    for f in val_files:
        shutil.copy(f, output_dir / 'labels' / 'val' / f.name)
    
    print(f"Split dataset:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    
    return len(train_files), len(val_files)


def process_trackid3x3(
    input_dir: Path,
    output_dir: Path,
    subsets: list = None
):
    """
    TrackID3x3データセット全体を処理
    """
    if subsets is None:
        subsets = ['Indoor', 'Outdoor', 'Drone']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir = output_dir / 'temp_labels'
    temp_labels_dir.mkdir(exist_ok=True)
    
    stats = []
    
    # 各サブセットを処理
    for subset in subsets:
        mot_dir = input_dir / 'ground_truth' / subset / 'MOT'
        
        if not mot_dir.exists():
            print(f"Warning: {mot_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {subset}...")
        
        # 解像度設定（サブセット別）
        if subset == 'Indoor':
            img_width, img_height = 1920, 1080
        elif subset == 'Outdoor':
            img_width, img_height = 1920, 1080
        else:  # Drone
            img_width, img_height = 3840, 2160  # 4K想定
        
        # MOTファイルを処理
        for mot_file in mot_dir.glob('*.txt'):
            video_name = mot_file.stem
            result = process_mot_file(
                mot_file,
                temp_labels_dir,
                video_name,
                img_width,
                img_height
            )
            stats.append(result)
            print(f"  Processed: {video_name} ({result['total_frames']} frames, {result['total_annotations']} annotations)")
    
    # データセット分割
    print("\nSplitting dataset...")
    train_count, val_count = split_dataset(temp_labels_dir, output_dir)
    
    # data.yaml生成
    create_dataset_yaml(output_dir)
    
    # 一時ディレクトリ削除
    shutil.rmtree(temp_labels_dir)
    
    # 統計情報保存
    stats_path = output_dir / 'conversion_stats.json'
    with open(stats_path, 'w') as f:
        json.dump({
            'subsets': subsets,
            'files': stats,
            'train_count': train_count,
            'val_count': val_count
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Conversion completed!")
    print(f"  Output: {output_dir}")
    print(f"  Total videos: {len(stats)}")
    print(f"  Total frames: {sum(s['total_frames'] for s in stats)}")
    print(f"  Total annotations: {sum(s['total_annotations'] for s in stats)}")
    print(f"{'='*60}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert TrackID3x3 dataset to YOLO format"
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='external_repos/TrackID3x3',
        help='TrackID3x3 repository path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/yolo_dataset',
        help='Output directory for YOLO dataset'
    )
    parser.add_argument(
        '--subsets',
        nargs='+',
        default=['Indoor', 'Outdoor'],
        help='Subsets to process (Indoor, Outdoor, Drone)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        print("Please clone TrackID3x3 first:")
        print("  git clone https://github.com/open-starlab/TrackID3x3.git external_repos/TrackID3x3")
        return
    
    process_trackid3x3(input_dir, output_dir, args.subsets)


if __name__ == "__main__":
    main()


