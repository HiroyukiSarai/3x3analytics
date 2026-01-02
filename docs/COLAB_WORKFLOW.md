# Cursor ⇔ Colab ワークフロー

## 🔄 GitHub経由ワークフロー（推奨）

### 初期セットアップ

```bash
# 1. プロジェクトディレクトリに移動
cd /Users/saraihiroyuki/3x3analytics

# 2. Git初期化（未設定の場合）
git init

# 3. .gitignoreを設定
cat << 'EOF' >> .gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Models (大きすぎるのでGitで管理しない)
models/*.pt
models/*.pth
models/*.onnx

# Data
data/raw/
data/processed/
*.mp4
*.mov

# IDE
.cursor/
.vscode/
EOF

# 4. コミット
git add .
git commit -m "Initial commit with basketball AI notebook"

# 5. GitHubでリポジトリ作成後、プッシュ
git remote add origin https://github.com/YOUR_USERNAME/3x3analytics.git
git branch -M main
git push -u origin main
```

### 日常の編集フロー

```
┌────────────────────────────────────────────────────────────┐
│ 1. Cursorでノートブックを編集                               │
│    notebooks/basketball_ai_roboflow.ipynb                  │
└────────────────────┬───────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│ 2. 変更をコミット＆プッシュ                                  │
│    git add notebooks/                                       │
│    git commit -m "SAM2をSAM3に更新"                         │
│    git push                                                 │
└────────────────────┬───────────────────────────────────────┘
                     ▼
┌────────────────────────────────────────────────────────────┐
│ 3. Colabで開く                                              │
│    https://colab.research.google.com/github/               │
│    YOUR_USERNAME/3x3analytics/blob/main/                   │
│    notebooks/basketball_ai_roboflow.ipynb                  │
└────────────────────────────────────────────────────────────┘
```

## 📝 Cursorでのノートブック編集

### 方法1: Jupyter形式で直接編集
Cursorで `.ipynb` ファイルを開くと、セル単位で編集可能

### 方法2: セルを追加・編集する例

```python
# SAM2 → SAM3 への変更例（セル8-9を置き換え）

# Before (SAM2)
!git clone https://github.com/Gy920/segment-anything-2-real-time.git

# After (SAM3)
!pip install segment-anything-3
from sam3 import SAM3
segmenter = SAM3(model_type="sam3-large")
```

## 🔧 ローカル実行（GPU不要の確認用）

```bash
# Jupyter Labを起動
pip install jupyterlab
jupyter lab --notebook-dir=/Users/saraihiroyuki/3x3analytics/notebooks

# ブラウザで開く
# http://localhost:8888
```

## 📦 Colabでの環境変数設定

Cursorで編集時、Colab専用コードを分岐させる：

```python
import os

# Colab環境判定
IN_COLAB = 'google.colab' in str(get_ipython())

if IN_COLAB:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
else:
    # ローカル環境
    from dotenv import load_dotenv
    load_dotenv()  # .envファイルから読み込み
```

## 🔗 よく使うColabリンク

```
# 自分のリポジトリから開く
https://colab.research.google.com/github/{USERNAME}/3x3analytics/blob/main/notebooks/{NOTEBOOK_NAME}.ipynb

# 特定のブランチから開く
https://colab.research.google.com/github/{USERNAME}/3x3analytics/blob/{BRANCH}/notebooks/{NOTEBOOK_NAME}.ipynb
```

