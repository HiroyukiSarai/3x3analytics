#!/bin/bash
# ============================================================
# 3x3バスケットボール戦術分析システム
# GitHub CLIを使わないgit clone版セットアップスクリプト
# ============================================================

set -e

PROJECT_ROOT="$(pwd)/3x3_basketball_analysis"
REPOS_DIR="${PROJECT_ROOT}/external_repos"

echo "============================================================"
echo " 3x3バスケットボール戦術分析システム セットアップ (git版)"
echo "============================================================"

# ディレクトリ作成
mkdir -p "${REPOS_DIR}"
mkdir -p "${PROJECT_ROOT}/models"
mkdir -p "${PROJECT_ROOT}/data/raw"
mkdir -p "${PROJECT_ROOT}/data/processed"
mkdir -p "${PROJECT_ROOT}/src"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/notebooks"
mkdir -p "${PROJECT_ROOT}/docs"

cd "${REPOS_DIR}"

echo ""
echo "[Phase 1] 物体検出・トラッキング"
echo "------------------------------------------------------------"

# 1. Ultralytics YOLO11 (物体検出の基盤)
echo "Cloning: ultralytics/ultralytics (YOLO11)..."
git clone https://github.com/ultralytics/ultralytics.git

# 2. BoxMOT (複数トラッカーを統合したライブラリ)
echo "Cloning: mikel-brostrom/boxmot..."
git clone https://github.com/mikel-brostrom/boxmot.git

# 3. ByteTrack 公式 (参考実装)
echo "Cloning: FoundationVision/ByteTrack..."
git clone https://github.com/FoundationVision/ByteTrack.git

echo ""
echo "[Phase 3] 骨格推定"
echo "------------------------------------------------------------"

# 4. MMPose (RTMPose含む)
echo "Cloning: open-mmlab/mmpose..."
git clone https://github.com/open-mmlab/mmpose.git

echo ""
echo "[3x3バスケ専用] データセット"
echo "------------------------------------------------------------"

# 5. TrackID3x3 (3x3バスケ専用データセット - 最重要)
echo "Cloning: open-starlab/TrackID3x3..."
git clone https://github.com/open-starlab/TrackID3x3.git

echo ""
echo "[参考実装] バスケ分析プロジェクト"
echo "------------------------------------------------------------"

# 6. AI_BasketBall_Analysis_v1 (統合的なバスケ分析)
echo "Cloning: HanaFEKI/AI_BasketBall_Analysis_v1..."
git clone https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1.git

# 7. basketball_analysis (コートキーポイント検出)
echo "Cloning: abdullahtarek/basketball_analysis..."
git clone https://github.com/abdullahtarek/basketball_analysis.git

# 8. Juxtapose (RTMPose + ByteTrack統合SDK)
echo "Cloning: ziqinyeow/juxtapose..."
git clone https://github.com/ziqinyeow/juxtapose.git

echo ""
echo "============================================================"
echo " クローン完了！"
echo "============================================================"
echo ""
echo "クローンしたリポジトリ:"
ls -1 "${REPOS_DIR}"
echo ""
echo "次のステップ: cd ${PROJECT_ROOT} && pip install -r requirements.txt"
