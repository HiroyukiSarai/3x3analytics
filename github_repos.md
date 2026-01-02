# GitHub リポジトリ クローンコマンド一覧

## 🚀 クイックスタート（コピペ用）

### GitHub CLI版（推奨）
```bash
# プロジェクトディレクトリ作成
mkdir -p 3x3_basketball_analysis/external_repos
cd 3x3_basketball_analysis/external_repos

# Phase 1: 物体検出・トラッキング
gh repo clone ultralytics/ultralytics
gh repo clone mikel-brostrom/boxmot
gh repo clone FoundationVision/ByteTrack

# Phase 3: 骨格推定
gh repo clone open-mmlab/mmpose

# 3x3バスケ専用データセット（最重要）
gh repo clone open-starlab/TrackID3x3

# 参考実装
gh repo clone HanaFEKI/AI_BasketBall_Analysis_v1
gh repo clone abdullahtarek/basketball_analysis
gh repo clone ziqinyeow/juxtapose
```

### Git版（GitHub CLI未インストールの場合）
```bash
# プロジェクトディレクトリ作成
mkdir -p 3x3_basketball_analysis/external_repos
cd 3x3_basketball_analysis/external_repos

# Phase 1: 物体検出・トラッキング
git clone https://github.com/ultralytics/ultralytics.git
git clone https://github.com/mikel-brostrom/boxmot.git
git clone https://github.com/FoundationVision/ByteTrack.git

# Phase 3: 骨格推定
git clone https://github.com/open-mmlab/mmpose.git

# 3x3バスケ専用データセット（最重要）
git clone https://github.com/open-starlab/TrackID3x3.git

# 参考実装
git clone https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1.git
git clone https://github.com/abdullahtarek/basketball_analysis.git
git clone https://github.com/ziqinyeow/juxtapose.git
```

---

## 📦 リポジトリ詳細

### 1. ultralytics/ultralytics（必須）
| 項目 | 内容 |
|------|------|
| 用途 | YOLO11による物体検出（選手・ボール） |
| Phase | Phase 1 |
| URL | https://github.com/ultralytics/ultralytics |
| クローン | `gh repo clone ultralytics/ultralytics` |

```bash
# インストール
pip install ultralytics

# 使用例
from ultralytics import YOLO
model = YOLO("yolo11n.pt")  # 検出
model = YOLO("yolo11n-pose.pt")  # 骨格推定
```

---

### 2. mikel-brostrom/boxmot（必須）
| 項目 | 内容 |
|------|------|
| 用途 | マルチオブジェクトトラッキング（ByteTrack, BoT-SORT等） |
| Phase | Phase 1 |
| URL | https://github.com/mikel-brostrom/boxmot |
| クローン | `gh repo clone mikel-brostrom/boxmot` |

```bash
# インストール
pip install boxmot

# コマンドライン使用
boxmot track yolov8n --source video.mp4

# Python使用例
from boxmot import DeepOCSORT
tracker = DeepOCSORT(device='cuda:0')
tracks = tracker.update(detections, frame)
```

---

### 3. FoundationVision/ByteTrack（参考）
| 項目 | 内容 |
|------|------|
| 用途 | ByteTrack公式実装（学習・カスタマイズ用） |
| Phase | Phase 1（参考） |
| URL | https://github.com/FoundationVision/ByteTrack |
| クローン | `gh repo clone FoundationVision/ByteTrack` |

---

### 4. open-mmlab/mmpose（必須）
| 項目 | 内容 |
|------|------|
| 用途 | 骨格推定（RTMPose） |
| Phase | Phase 3 |
| URL | https://github.com/open-mmlab/mmpose |
| クローン | `gh repo clone open-mmlab/mmpose` |

```bash
# インストール
mim install mmpose

# RTMPose使用例
python demo/topdown_demo_with_mmdet.py \
    projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth \
    --input video.mp4 --output-root outputs/
```

---

### 5. open-starlab/TrackID3x3（最重要）
| 項目 | 内容 |
|------|------|
| 用途 | **3x3バスケ専用データセット・ベースライン** |
| Phase | 全Phase（ファインチューニング用） |
| URL | https://github.com/open-starlab/TrackID3x3 |
| クローン | `gh repo clone open-starlab/TrackID3x3` |

```
データセット構成:
- Indoor: 屋内固定カメラ映像
- Outdoor: 屋外固定カメラ映像  
- Drone: ドローン空撮映像

アノテーション:
- 6選手のバウンディングボックス
- 10キーポイント骨格データ
- 選手ID（ジャージ番号ベース）
```

---

### 6. HanaFEKI/AI_BasketBall_Analysis_v1（参考実装）
| 項目 | 内容 |
|------|------|
| 用途 | YOLO+ByteTrack+ホモグラフィの統合実装 |
| Phase | Phase 2参考 |
| URL | https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1 |
| クローン | `gh repo clone HanaFEKI/AI_BasketBall_Analysis_v1` |

```
主要機能:
- YOLO物体検出
- ByteTrackトラッキング
- ゼロショットチーム分類（Fashion CLIP）
- コートキーポイント検出
- ホモグラフィ変換（2Dコート投影）
```

---

### 7. abdullahtarek/basketball_analysis（参考実装）
| 項目 | 内容 |
|------|------|
| 用途 | コートキーポイント検出・パス検出 |
| Phase | Phase 2, 4参考 |
| URL | https://github.com/abdullahtarek/basketball_analysis |
| クローン | `gh repo clone abdullahtarek/basketball_analysis` |

```
学習済みモデル提供:
- ball_detector_model.pt（ボール検出）
- court_keypoint_detector.pt（コートキーポイント）
- player_detector.pt（選手検出）
```

---

### 8. ziqinyeow/juxtapose（統合SDK）
| 項目 | 内容 |
|------|------|
| 用途 | RTMPose + ByteTrack/BotSORTの統合SDK |
| Phase | Phase 1, 3統合 |
| URL | https://github.com/ziqinyeow/juxtapose |
| クローン | `gh repo clone ziqinyeow/juxtapose` |

```bash
# インストール
pip install juxtapose

# 使用例
from juxtapose import RTM
model = RTM(det="rtmdet-l", pose="rtmpose-l", tracker="bytetrack")
model("video.mp4")
```

---

## 📁 推奨ディレクトリ構成

```
3x3_basketball_analysis/
├── external_repos/          # クローンしたリポジトリ
│   ├── ultralytics/
│   ├── boxmot/
│   ├── ByteTrack/
│   ├── mmpose/
│   ├── TrackID3x3/          # ★最重要
│   ├── AI_BasketBall_Analysis_v1/
│   ├── basketball_analysis/
│   └── juxtapose/
├── models/                  # 学習済みモデル置き場
│   ├── yolo11n.pt
│   ├── yolo11n-pose.pt
│   ├── rtmpose-m.pth
│   └── court_keypoint.pt
├── data/
│   ├── raw/                 # 入力動画
│   ├── processed/           # 処理済みデータ
│   └── annotations/         # アノテーション
├── src/                     # 自作コード
│   ├── detection/
│   ├── tracking/
│   ├── pose/
│   ├── homography/
│   ├── analysis/
│   └── visualization/
├── outputs/                 # 出力結果
├── notebooks/               # Jupyter notebooks
├── docs/                    # ドキュメント
│   └── requirements.md
├── requirements.txt
├── setup_repos.sh           # GitHub CLI版セットアップ
└── setup_repos_git.sh       # Git版セットアップ
```

---

## 🔧 環境構築手順

```bash
# 1. リポジトリクローン
./setup_repos.sh  # または ./setup_repos_git.sh

# 2. Python仮想環境作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. PyTorchインストール（GPU版）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 依存パッケージインストール
pip install -r requirements.txt

# 5. MMPoseインストール
mim install mmpose
# または: pip install -v -e ./external_repos/mmpose

# 6. 動作確認
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "from boxmot import ByteTrack; print('BoxMOT OK')"
```

---

## 📚 参考リンク

- [Ultralytics Docs](https://docs.ultralytics.com/)
- [BoxMOT Docs](https://github.com/mikel-brostrom/boxmot#readme)
- [MMPose Docs](https://mmpose.readthedocs.io/)
- [TrackID3x3 Paper](https://arxiv.org/abs/2503.18282)
