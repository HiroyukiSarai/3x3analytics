"""
YOLO11 ファインチューニングスクリプト
TrackID3x3データセットを使用して3x3バスケットボール専用モデルを学習

使用方法:
    # 基本的な学習
    python scripts/finetune_yolo11.py --data data/yolo_dataset/data.yaml --epochs 100
    
    # 骨格推定モデルの学習
    python scripts/finetune_yolo11.py --data data/yolo_dataset/data.yaml --task pose --epochs 100
    
    # 学習再開
    python scripts/finetune_yolo11.py --resume runs/detect/train/weights/last.pt
"""

import argparse
from pathlib import Path
from datetime import datetime
import yaml


def finetune_detection(
    data_yaml: str,
    base_model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/detect",
    name: str = None,
    resume: str = None,
    **kwargs
):
    """
    物体検出モデルのファインチューニング
    
    Args:
        data_yaml: データセット設定ファイル
        base_model: ベースモデル (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt)
        epochs: エポック数
        imgsz: 入力画像サイズ
        batch: バッチサイズ
        device: GPUデバイス (0, 0,1, cpu)
        project: プロジェクトディレクトリ
        name: 実験名
        resume: 学習再開するチェックポイント
    """
    from ultralytics import YOLO
    
    print("=" * 60)
    print("YOLO11 Fine-tuning for 3x3 Basketball Detection")
    print("=" * 60)
    
    if resume:
        print(f"Resuming from: {resume}")
        model = YOLO(resume)
    else:
        print(f"Base model: {base_model}")
        model = YOLO(base_model)
    
    # 実験名の設定
    if name is None:
        name = f"basketball3x3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 3x3バスケットボール用の推奨設定
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        
        # データ拡張（3x3バスケ向け最適化）
        'hsv_h': 0.015,      # 色相変動（ジャージ色への適応）
        'hsv_s': 0.7,        # 彩度変動
        'hsv_v': 0.4,        # 明度変動（照明条件）
        'degrees': 10.0,     # 回転（カメラ角度）
        'translate': 0.1,    # 平行移動
        'scale': 0.5,        # スケール
        'shear': 2.0,        # せん断
        'perspective': 0.0,  # 透視変換（コートは平面）
        'flipud': 0.0,       # 上下反転（バスケでは不自然）
        'fliplr': 0.5,       # 左右反転
        'mosaic': 1.0,       # モザイク（オクルージョン対策）
        'mixup': 0.1,        # MixUp
        'copy_paste': 0.1,   # Copy-Paste
        
        # 学習設定
        'optimizer': 'AdamW',
        'lr0': 0.01,         # 初期学習率
        'lrf': 0.01,         # 最終学習率係数
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 損失関数の重み
        'box': 7.5,          # Box損失
        'cls': 0.5,          # 分類損失
        'dfl': 1.5,          # DFL損失
        
        # その他
        'patience': 50,      # Early stopping
        'save': True,
        'save_period': 10,   # 10エポックごとに保存
        'cache': True,       # データセットキャッシュ
        'workers': 8,
        'close_mosaic': 10,  # 最後の10エポックはモザイクなし
        'amp': True,         # 混合精度学習
        
        # 追加の引数
        **kwargs
    }
    
    print(f"\nTraining configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    print(f"  Device: {device}")
    print(f"  Project: {project}/{name}")
    print()
    
    # 学習実行
    results = model.train(**train_args)
    
    # 結果を表示
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print(f"Last model: {project}/{name}/weights/last.pt")
    print("=" * 60)
    
    return results


def finetune_pose(
    data_yaml: str,
    base_model: str = "yolo11n-pose.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/pose",
    name: str = None,
    **kwargs
):
    """
    骨格推定モデルのファインチューニング
    
    注意: TrackID3x3の骨格データは10キーポイントのため、
    COCO (17キーポイント) とは異なるフォーマットへの対応が必要
    """
    from ultralytics import YOLO
    
    print("=" * 60)
    print("YOLO11 Fine-tuning for 3x3 Basketball Pose Estimation")
    print("=" * 60)
    
    print(f"Base model: {base_model}")
    model = YOLO(base_model)
    
    if name is None:
        name = f"basketball3x3_pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 骨格推定用設定
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'project': project,
        'name': name,
        
        # 骨格推定特有の設定
        'pose': 12.0,        # Pose損失の重み
        'kobj': 1.0,         # Keypoint objectness損失
        
        # データ拡張
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,
        
        # 学習設定
        'optimizer': 'AdamW',
        'lr0': 0.01,
        'patience': 50,
        'amp': True,
        
        **kwargs
    }
    
    print(f"\nTraining configuration:")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Task: Pose estimation")
    print()
    
    results = model.train(**train_args)
    
    print("\n" + "=" * 60)
    print("Pose training completed!")
    print(f"Best model: {project}/{name}/weights/best.pt")
    print("=" * 60)
    
    return results


def validate_model(
    model_path: str,
    data_yaml: str,
    device: str = "0"
):
    """
    学習済みモデルの検証
    """
    from ultralytics import YOLO
    
    print("=" * 60)
    print("Model Validation")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    results = model.val(
        data=data_yaml,
        device=device,
        split='val'
    )
    
    print("\nValidation Results:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    device: str = "0"
):
    """
    モデルをエクスポート
    
    Args:
        format: 出力形式 (onnx, torchscript, tflite, coreml, etc.)
    """
    from ultralytics import YOLO
    
    print("=" * 60)
    print(f"Exporting model to {format}")
    print("=" * 60)
    
    model = YOLO(model_path)
    
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        device=device
    )
    
    print(f"\nExported: {export_path}")
    
    return export_path


def create_training_notebook(output_path: str = "notebooks/finetune_yolo11.ipynb"):
    """
    Google Colab用のファインチューニングノートブックを生成
    """
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# YOLO11 Fine-tuning for 3x3 Basketball\n",
                    "\n",
                    "TrackID3x3データセットを使用して3x3バスケットボール専用の物体検出モデルを学習します。"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# 環境セットアップ\n",
                    "!pip install ultralytics\n",
                    "!pip install roboflow  # オプション: データセット管理"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Google Driveマウント（データセット用）\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# TrackID3x3リポジトリをクローン\n",
                    "!git clone https://github.com/open-starlab/TrackID3x3.git\n",
                    "\n",
                    "# データセットをダウンロード（Google Driveから手動でコピー）\n",
                    "# https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "from ultralytics import YOLO\n",
                    "\n",
                    "# ベースモデルをロード\n",
                    "model = YOLO('yolo11n.pt')  # または yolo11s.pt, yolo11m.pt\n",
                    "\n",
                    "# ファインチューニング\n",
                    "results = model.train(\n",
                    "    data='data.yaml',\n",
                    "    epochs=100,\n",
                    "    imgsz=640,\n",
                    "    batch=16,\n",
                    "    device=0,\n",
                    "    project='basketball3x3',\n",
                    "    name='yolo11_finetune',\n",
                    "    \n",
                    "    # 3x3バスケ向け設定\n",
                    "    hsv_h=0.015,\n",
                    "    hsv_s=0.7,\n",
                    "    hsv_v=0.4,\n",
                    "    degrees=10.0,\n",
                    "    flipud=0.0,\n",
                    "    fliplr=0.5,\n",
                    "    mosaic=1.0,\n",
                    "    mixup=0.1,\n",
                    ")"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# 検証\n",
                    "results = model.val()\n",
                    "print(f'mAP50: {results.box.map50}')\n",
                    "print(f'mAP50-95: {results.box.map}')"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# モデルを保存（Google Driveへ）\n",
                    "!cp basketball3x3/yolo11_finetune/weights/best.pt /content/drive/MyDrive/models/yolo11_basketball3x3.pt"
                ],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"Created notebook: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO11 Fine-tuning for 3x3 Basketball",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='data/yolo_dataset/data.yaml',
        help='Path to data.yaml'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='yolo11n.pt',
        help='Base model (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt)'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default='detect',
        choices=['detect', 'pose'],
        help='Task type'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='CUDA device (0, 0,1, cpu)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint'
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        default=None,
        help='Validate a trained model'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export a trained model'
    )
    
    parser.add_argument(
        '--export-format',
        type=str,
        default='onnx',
        help='Export format (onnx, torchscript, tflite, etc.)'
    )
    
    parser.add_argument(
        '--create-notebook',
        action='store_true',
        help='Create Colab notebook'
    )
    
    args = parser.parse_args()
    
    # ノートブック生成
    if args.create_notebook:
        create_training_notebook()
        return
    
    # 検証モード
    if args.validate:
        validate_model(args.validate, args.data, args.device)
        return
    
    # エクスポートモード
    if args.export:
        export_model(args.export, args.export_format, args.imgsz, args.device)
        return
    
    # 学習モード
    if args.task == 'detect':
        finetune_detection(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume
        )
    elif args.task == 'pose':
        # 骨格推定モデル
        if 'pose' not in args.model:
            args.model = args.model.replace('.pt', '-pose.pt')
        
        finetune_pose(
            data_yaml=args.data,
            base_model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )


if __name__ == "__main__":
    main()


