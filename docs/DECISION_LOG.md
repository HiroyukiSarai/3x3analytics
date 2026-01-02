# 3x3バスケットボール分析システム - 意思決定ログ

## 📅 2024年12月31日 - アーキテクチャ変更

### 決定内容

**現在の実装（YOLO + SimpleTracker）から、Roboflow/SAM2ベースのアプローチに切り替える**

---

### 背景・理由

#### 現在の実装の問題点

| 項目 | 現状 | 問題 |
|------|------|------|
| 選手検出 | YOLO11n（汎用モデル） | バスケ特化していない |
| トラッキング | SimpleTracker（IoU） | 重なりでID喪失 |
| チーム分類 | なし | 手動でないと区別できない |
| 背番号認識 | なし | 選手の特定ができない |
| コート検出 | 手動4点指定 | 毎回手作業が必要 |

#### Roboflow/SAM2アプローチの優位性

| 項目 | 技術 | メリット |
|------|------|---------|
| 選手検出 | RF-DETR | バスケ特化、高精度 |
| トラッキング | SAM2（Meta） | 重なっても追跡継続 |
| チーム分類 | SigLIP + K-means | ユニフォーム色で自動分類 |
| 背番号認識 | SmolVLM2 | VLMで数字を読み取り |
| コート検出 | 自動検出モデル | ラインを自動認識 |

---

## 📚 公式リソース一覧（Roboflow）

### 📖 メインチュートリアル

| リソース | URL |
|---------|-----|
| **ブログ記事（メイン）** | https://blog.roboflow.com/identify-basketball-players/ |

### 📓 Colabノートブック

| ノートブック | URL | 内容 |
|-------------|-----|------|
| **選手検出・追跡・識別** | https://colab.research.google.com/github/roboflow/sports/blob/main/examples/basketball/basketball_player_tracking.ipynb | **メインノートブック** |
| **Make or Miss（シュート判定）** | https://colab.research.google.com/github/roboflow/sports/blob/main/examples/basketball/make_or_miss.ipynb | シュート成功/失敗判定 |
| **RF-DETRファインチューニング** | https://colab.research.google.com/github/roboflow/rf-detr/blob/main/notebooks/rf_detr_fine_tuning.ipynb | カスタムデータ学習 |
| **SAM2動画セグメンテーション** | https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb | トラッキング |

### 🗃️ データセット（Roboflow Universe）

| データセット | URL | 用途 |
|-------------|-----|------|
| **Basketball Player Detection** | https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2 | 選手検出 |
| **Basketball Court Keypoint** | https://universe.roboflow.com/roboflow-universe-projects/basketball-court-keypoints | コート検出 |
| **Basketball Jersey Number OCR** | https://universe.roboflow.com/roboflow-universe-projects/basketball-jersey-number-ocr | 背番号認識 |

### 🔧 GitHubリポジトリ

| リポジトリ | URL | 内容 |
|-----------|-----|------|
| **RF-DETR** | https://github.com/roboflow/rf-detr | 検出モデル |
| **Supervision** | https://github.com/roboflow/supervision | 可視化ライブラリ |
| **Sports** | https://github.com/roboflow/sports | スポーツ分析例 |
| **SkalskiP（開発者）** | https://github.com/SkalskiP | 開発者の最新プロジェクト |

---

## 🚀 実装計画

### Phase 1: 環境構築・検証（Google Colab）

```
1. メインノートブックを開く
   → https://colab.research.google.com/github/roboflow/sports/blob/main/examples/basketball/basketball_player_tracking.ipynb

2. Roboflow APIキーを取得
   → https://app.roboflow.com

3. サンプル動画でテスト実行

4. 自分の3x3動画でテスト
```

### Phase 2: 3x3カスタマイズ

```
1. コート寸法を3x3規格（15m x 11m）に変更
2. 必要に応じて3x3画像でファインチューニング
3. ホモグラフィ変換を3x3コートに最適化
```

### Phase 3: ローカル/本番環境

```
1. Colabで動作確認したコードをローカルに移植
2. Streamlitダッシュボードと統合
3. パフォーマンス最適化
```

---

## ⚙️ 必要な環境

| 項目 | 要件 |
|------|------|
| **GPU** | 必須（Colab無料版でも可） |
| **Python** | 3.9+ |
| **主要ライブラリ** | roboflow, inference, supervision, sam2, transformers |

---

## ✅ チェックリスト

- [x] `feat/basketball`ブランチをクローン
- [x] 依存関係をインストール
- [x] 旧ファイルをアーカイブ
- [x] 新アナライザー作成（`analyzer.py`）
- [x] ViewTransformer統合
- [ ] 3x3動画でテスト
- [ ] チーム分類機能追加
- [ ] 背番号認識追加

---

## 📝 備考

- 現在の実装（`app_simple.py`）は参照用として保持
- 切り替え後も座標変換（ホモグラフィ）のロジックは再利用可能
- Colabで動作確認後、ローカル実装に移行予定
