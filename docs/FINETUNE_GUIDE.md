# YOLO11 ファインチューニングガイド

3x3バスケットボール専用のYOLO11モデルを学習する手順です。

## 📋 前提条件

- Python 3.10+
- NVIDIA GPU (RTX 3060以上推奨)
- CUDA 11.8+
- 30GB+ のディスク空き容量

## 🚀 クイックスタート

### Step 1: 環境構築

```bash
# 仮想環境を作成
cd /Users/saraihiroyuki/3x3analytics
python -m venv venv
source venv/bin/activate

# PyTorchをインストール（GPU版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 必要なパッケージをインストール
pip install ultralytics
pip install -r requirements.txt
```

### Step 2: TrackID3x3 動画データのダウンロード

⚠️ **重要**: 動画ファイルはGitリポジトリには含まれていません。以下のGoogle Driveからダウンロードしてください。

📥 **ダウンロードリンク**: [TrackID3x3 Google Drive](https://drive.google.com/drive/folders/1aWqMwQKr5xKMjqms7-raYluSlxPsGvwX)

ダウンロード後、以下のように配置してください：

```
external_repos/TrackID3x3/
├── videos/
│   ├── Indoor/
│   │   ├── basket_S1T1_pre.mp4
│   │   ├── basket_S1T2_pre.mp4
│   │   └── ...
│   └── Outdoor/
│       ├── IMG_0104.mp4
│       ├── IMG_0105.mp4
│       └── ...
```

### Step 3: 動画からフレームを抽出

```bash
# フレーム抽出スクリプトを実行
python scripts/extract_frames.py \
    --input external_repos/TrackID3x3/videos \
    --output data/yolo_dataset/images \
    --fps 5  # 5FPSでサンプリング
```

または、Google Colab上で直接学習する場合はこのステップは不要です。

### Step 4: ファインチューニング実行

```bash
# データセット変換は完了済み
# data/yolo_dataset/data.yaml が作成されています

# ファインチューニングを実行
python scripts/finetune_yolo11.py \
    --data data/yolo_dataset/data.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --batch 16 \
    --device 0

# より大きいモデルを使用する場合
python scripts/finetune_yolo11.py \
    --data data/yolo_dataset/data.yaml \
    --model yolo11s.pt \  # または yolo11m.pt
    --epochs 100 \
    --batch 8 \
    --device 0
```

## 📊 データセット統計

現在のデータセット変換結果：

| 項目 | 値 |
|------|-----|
| 動画数 | 54 |
| 総フレーム数 | 150,809 |
| 総アノテーション数 | 904,607 |
| Train/Val分割 | 80%/20% |

### サブセット内訳

| サブセット | 動画数 | 用途 |
|-----------|-------|------|
| Indoor | 42 | 屋内固定カメラ |
| Outdoor | 12 | 屋外固定カメラ |

## 🎯 推奨設定

### GPUメモリ別のバッチサイズ

| GPUメモリ | モデル | 推奨バッチサイズ |
|----------|-------|----------------|
| 8GB | yolo11n | 16 |
| 12GB | yolo11n | 32 |
| 12GB | yolo11s | 16 |
| 24GB | yolo11m | 16 |
| 24GB | yolo11l | 8 |

### 学習時間の目安

| モデル | エポック数 | 時間 (RTX 3080) |
|-------|-----------|----------------|
| yolo11n | 100 | ~4時間 |
| yolo11s | 100 | ~8時間 |
| yolo11m | 100 | ~16時間 |

## 🔧 トラブルシューティング

### CUDA Out of Memory

```bash
# バッチサイズを減らす
python scripts/finetune_yolo11.py --batch 8

# 画像サイズを減らす
python scripts/finetune_yolo11.py --imgsz 512
```

### 学習の再開

```bash
python scripts/finetune_yolo11.py \
    --resume runs/detect/basketball3x3_*/weights/last.pt
```

## 📈 学習結果の確認

学習完了後、以下のファイルが生成されます：

```
runs/detect/basketball3x3_*/
├── weights/
│   ├── best.pt      # 最良モデル
│   └── last.pt      # 最終モデル
├── results.csv      # メトリクス履歴
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── results.png
```

### モデルの検証

```bash
python scripts/finetune_yolo11.py \
    --validate runs/detect/basketball3x3_*/weights/best.pt \
    --data data/yolo_dataset/data.yaml
```

### モデルのエクスポート

```bash
# ONNX形式でエクスポート
python scripts/finetune_yolo11.py \
    --export runs/detect/basketball3x3_*/weights/best.pt \
    --export-format onnx
```

## 🎓 Google Colab での学習

GPUリソースがない場合は、Google Colabを使用できます：

```bash
# Colabノートブックを生成
python scripts/finetune_yolo11.py --create-notebook
```

生成されたノートブック `notebooks/finetune_yolo11.ipynb` をColabにアップロードして実行してください。

## 📁 ファインチューニング後の使用

```python
from ultralytics import YOLO

# ファインチューニング済みモデルをロード
model = YOLO('runs/detect/basketball3x3_*/weights/best.pt')

# 推論
results = model('game_video.mp4')

# または main.py で使用
# python main.py --detector-model runs/detect/basketball3x3_*/weights/best.pt --source video.mp4
```

## 📚 参考リンク

- [Ultralytics YOLO11 Training](https://docs.ultralytics.com/modes/train/)
- [TrackID3x3 Paper](https://arxiv.org/abs/2503.18282)
- [Custom Dataset Training](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/)


