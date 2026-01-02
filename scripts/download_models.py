"""
モデルダウンロードスクリプト
YOLO11モデルとRe-IDモデルをダウンロード
"""

import os
import sys
from pathlib import Path


def download_yolo_models():
    """YOLOモデルをダウンロード"""
    try:
        from ultralytics import YOLO
        
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        print("YOLOモデルをダウンロード中...")
        
        # 検出モデル
        print("  - yolo11n.pt (検出用)")
        model_det = YOLO("yolo11n.pt")
        
        # 骨格推定モデル
        print("  - yolo11n-pose.pt (骨格推定用)")
        model_pose = YOLO("yolo11n-pose.pt")
        
        print("YOLOモデルのダウンロード完了!")
        
    except ImportError:
        print("Error: ultralytics がインストールされていません")
        print("  pip install ultralytics")
        sys.exit(1)


def download_reid_models():
    """Re-IDモデルをダウンロード"""
    import urllib.request
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # OSNet Re-ID model
    reid_url = "https://drive.google.com/uc?export=download&id=1mJ9AhkZJ4WyHTR5XLEgH4C94kcCvGNjM"
    reid_path = models_dir / "osnet_x0_25_msmt17.pt"
    
    if not reid_path.exists():
        print("Re-IDモデルをダウンロード中...")
        print("  注意: GoogleドライブからのダウンロードはgdownをGUI使用してください")
        print(f"  URL: {reid_url}")
        print(f"  保存先: {reid_path}")
    else:
        print(f"Re-IDモデルは既に存在します: {reid_path}")


def main():
    print("=" * 60)
    print("3x3 Basketball Analysis - Model Downloader")
    print("=" * 60)
    print()
    
    download_yolo_models()
    print()
    download_reid_models()
    
    print()
    print("=" * 60)
    print("完了!")
    print("=" * 60)


if __name__ == "__main__":
    main()


