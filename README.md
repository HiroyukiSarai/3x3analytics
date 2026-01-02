# 🏀 3x3 Basketball Tactical Analysis System

3人制バスケットボール（3x3）の試合映像から、得点に繋がる戦術パターンを自動抽出・分析するシステムです。

## ✨ 機能

- **Phase 1: 物体検出・トラッキング** - YOLO11 + BoxMOT
- **Phase 2: 座標変換・可視化** - コート検出・ホモグラフィ変換
- **Phase 3: 骨格推定・チーム分類** - YOLO-pose + K-means
- **Phase 4: 戦術パターン分析** - DTW + クラスタリング
- **Phase 5: レポート・ダッシュボード** - Streamlit Web UI

## 📁 プロジェクト構成

```
3x3analytics/
├── src/                    # ソースコード
│   ├── detection/          # Phase 1: 物体検出
│   ├── tracking/           # Phase 1: トラッキング
│   ├── homography/         # Phase 2: 座標変換
│   ├── pose/               # Phase 3: 骨格推定
│   ├── analysis/           # Phase 4: 戦術分析
│   └── visualization/      # Phase 5: 可視化
├── configs/                # 設定ファイル
├── data/                   # データ
│   ├── raw/                # 入力動画
│   ├── processed/          # 処理済みデータ
│   └── annotations/        # アノテーション
├── models/                 # 学習済みモデル
├── outputs/                # 出力結果
├── notebooks/              # Jupyter notebooks
├── app.py                  # Streamlit ダッシュボード
├── main.py                 # メインパイプライン
└── requirements.txt        # 依存パッケージ
```

## 🚀 クイックスタート

### 1. 環境構築

```bash
# Python仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# PyTorchインストール（GPU版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 依存パッケージインストール
pip install -r requirements.txt
```

### 2. 動画分析の実行

```bash
# 基本的な使用方法
python main.py --source data/raw/game.mp4 --output outputs/

# 戦術パターン分析付き
python main.py --source data/raw/game.mp4 --output outputs/ --analyze-patterns

# レポート生成付き
python main.py --source data/raw/game.mp4 --output outputs/ --analyze-patterns --generate-report

# プレビュー表示
python main.py --source data/raw/game.mp4 --show-preview
```

### 3. Streamlit ダッシュボード起動

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` を開いてダッシュボードにアクセスできます。

## 📊 使用例

### Python API

```python
from main import Basketball3x3Pipeline

# パイプライン初期化
pipeline = Basketball3x3Pipeline(
    detector_model="yolo11n.pt",
    tracker_type="bytetrack",
    device="cuda:0",
    output_dir="outputs"
)

# 動画処理
results = pipeline.process_video(
    source="game.mp4",
    analyze_patterns=True,
    generate_report=True
)

print(f"検出パターン: {results['patterns_detected']}種類")
print(f"検出アクション: {results['actions_detected']}回")
```

### 個別モジュールの使用

```python
# 物体検出
from src.detection.detector import BasketballDetector
detector = BasketballDetector(model_path="yolo11n.pt")
detections = detector.detect(frame)

# トラッキング
from src.tracking.tracker import BasketballTracker
tracker = BasketballTracker(tracker_type="bytetrack")
tracks = tracker.update(detections['raw_detections'], frame)

# 戦術パターン分析
from src.analysis.pattern_analyzer import PatternAnalyzer
analyzer = PatternAnalyzer(n_clusters=8)
patterns = analyzer.analyze_plays(plays)
```

## ⚙️ 設定

設定ファイル `configs/default.yaml` で各種パラメータを調整できます：

```yaml
detection:
  model_path: "yolo11n.pt"
  conf_threshold: 0.5

tracking:
  tracker_type: "bytetrack"
  max_trajectory_length: 90

analysis:
  pattern:
    n_clusters: 8
    method: "kmeans"
```

## 📈 出力

### トラッキングデータ (JSON)

```json
{
  "frame_id": 100,
  "timestamp": 3.33,
  "players": [
    {
      "track_id": 1,
      "team": "A",
      "bbox": [100, 200, 150, 400],
      "court_position": [750, 500]
    }
  ],
  "ball": {
    "bbox": [200, 300, 220, 320],
    "confidence": 0.85
  }
}
```

### 戦術パターン

```json
{
  "pattern_id": "pattern_01",
  "pattern_name": "ピック＆ロール",
  "occurrences": 25,
  "success_rate": 0.72
}
```

## 🛠️ 技術スタック

| カテゴリ | 技術 |
|----------|------|
| 物体検出 | Ultralytics YOLO11 |
| トラッキング | BoxMOT (ByteTrack/BoT-SORT) |
| 骨格推定 | YOLO11-pose |
| 時系列分析 | DTW, tslearn |
| 可視化 | OpenCV, Plotly, Matplotlib |
| WebUI | Streamlit |

## 📚 参考資料

- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [BoxMOT](https://github.com/mikel-brostrom/boxmot)
- [TrackID3x3 Dataset](https://github.com/open-starlab/TrackID3x3)

## 📝 ライセンス

MIT License

## 🤝 コントリビューション

プルリクエストを歓迎します！バグ報告や機能リクエストは Issue でお願いします。


