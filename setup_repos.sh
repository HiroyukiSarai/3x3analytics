#!/bin/bash
# ============================================================
# 3x3バスケットボール戦術分析システム - リポジトリセットアップスクリプト
# ============================================================
# 
# 使用方法:
#   chmod +x setup_repos.sh
#   ./setup_repos.sh
#
# 前提条件:
#   - GitHub CLI (gh) がインストール済み
#   - git がインストール済み
#   - Python 3.10+ がインストール済み
#
# ============================================================

set -e  # エラー時に停止

# 色付きログ出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================
# 1. ディレクトリ構成
# ============================================================

PROJECT_ROOT="$(pwd)/3x3_basketball_analysis"
REPOS_DIR="${PROJECT_ROOT}/external_repos"
MODELS_DIR="${PROJECT_ROOT}/models"
DATA_DIR="${PROJECT_ROOT}/data"
SRC_DIR="${PROJECT_ROOT}/src"

echo ""
echo "============================================================"
echo " 3x3バスケットボール戦術分析システム セットアップ"
echo "============================================================"
echo ""

log_info "プロジェクトディレクトリを作成中..."

mkdir -p "${REPOS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${DATA_DIR}/raw"
mkdir -p "${DATA_DIR}/processed"
mkdir -p "${DATA_DIR}/annotations"
mkdir -p "${SRC_DIR}/detection"
mkdir -p "${SRC_DIR}/tracking"
mkdir -p "${SRC_DIR}/pose"
mkdir -p "${SRC_DIR}/homography"
mkdir -p "${SRC_DIR}/analysis"
mkdir -p "${SRC_DIR}/visualization"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "${PROJECT_ROOT}/notebooks"
mkdir -p "${PROJECT_ROOT}/docs"

log_success "ディレクトリ構成を作成しました"

cd "${REPOS_DIR}"

# ============================================================
# 2. Phase 1: 物体検出 (YOLO11)
# ============================================================

echo ""
echo "============================================================"
echo " Phase 1: 物体検出 - Ultralytics YOLO11"
echo "============================================================"

log_info "[1/8] Ultralytics (YOLO11) をクローン中..."

if [ ! -d "ultralytics" ]; then
    # GitHub CLI を使用
    gh repo clone ultralytics/ultralytics
    log_success "ultralytics をクローンしました"
else
    log_warning "ultralytics は既に存在します。スキップします。"
fi

# ============================================================
# 3. Phase 1: マルチオブジェクトトラッキング (BoxMOT)
# ============================================================

echo ""
echo "============================================================"
echo " Phase 1: トラッキング - BoxMOT"
echo "============================================================"

log_info "[2/8] BoxMOT (ByteTrack, BoT-SORT等) をクローン中..."

if [ ! -d "boxmot" ]; then
    gh repo clone mikel-brostrom/boxmot
    log_success "boxmot をクローンしました"
else
    log_warning "boxmot は既に存在します。スキップします。"
fi

# ByteTrack 公式（参考用）
log_info "[3/8] ByteTrack 公式リポジトリをクローン中..."

if [ ! -d "ByteTrack" ]; then
    gh repo clone FoundationVision/ByteTrack
    log_success "ByteTrack をクローンしました"
else
    log_warning "ByteTrack は既に存在します。スキップします。"
fi

# ============================================================
# 4. Phase 3: 骨格推定 (MMPose / RTMPose)
# ============================================================

echo ""
echo "============================================================"
echo " Phase 3: 骨格推定 - MMPose (RTMPose)"
echo "============================================================"

log_info "[4/8] MMPose (RTMPose含む) をクローン中..."

if [ ! -d "mmpose" ]; then
    gh repo clone open-mmlab/mmpose
    log_success "mmpose をクローンしました"
else
    log_warning "mmpose は既に存在します。スキップします。"
fi

# ============================================================
# 5. 3x3バスケ専用データセット (TrackID3x3)
# ============================================================

echo ""
echo "============================================================"
echo " 3x3専用データセット - TrackID3x3"
echo "============================================================"

log_info "[5/8] TrackID3x3 (3x3バスケ専用データセット) をクローン中..."

if [ ! -d "TrackID3x3" ]; then
    gh repo clone open-starlab/TrackID3x3
    log_success "TrackID3x3 をクローンしました"
else
    log_warning "TrackID3x3 は既に存在します。スキップします。"
fi

# ============================================================
# 6. バスケ分析参考プロジェクト
# ============================================================

echo ""
echo "============================================================"
echo " 参考プロジェクト - バスケ分析実装例"
echo "============================================================"

log_info "[6/8] AI_BasketBall_Analysis_v1 をクローン中..."

if [ ! -d "AI_BasketBall_Analysis_v1" ]; then
    gh repo clone HanaFEKI/AI_BasketBall_Analysis_v1
    log_success "AI_BasketBall_Analysis_v1 をクローンしました"
else
    log_warning "AI_BasketBall_Analysis_v1 は既に存在します。スキップします。"
fi

log_info "[7/8] basketball_analysis (コートキーポイント検出) をクローン中..."

if [ ! -d "basketball_analysis" ]; then
    gh repo clone abdullahtarek/basketball_analysis
    log_success "basketball_analysis をクローンしました"
else
    log_warning "basketball_analysis は既に存在します。スキップします。"
fi

# ============================================================
# 7. 統合ツール (Juxtapose)
# ============================================================

echo ""
echo "============================================================"
echo " 統合ツール - Juxtapose (RTMPose + Tracker)"
echo "============================================================"

log_info "[8/8] Juxtapose (スポーツ分析SDK) をクローン中..."

if [ ! -d "juxtapose" ]; then
    gh repo clone ziqinyeow/juxtapose
    log_success "juxtapose をクローンしました"
else
    log_warning "juxtapose は既に存在します。スキップします。"
fi

# ============================================================
# 8. クローン完了サマリー
# ============================================================

echo ""
echo "============================================================"
echo " クローン完了サマリー"
echo "============================================================"
echo ""

log_success "全てのリポジトリをクローンしました！"
echo ""
echo "クローンしたリポジトリ:"
ls -la "${REPOS_DIR}"

echo ""
echo "============================================================"
echo " 次のステップ: Python環境セットアップ"
echo "============================================================"
echo ""
echo "以下のコマンドを実行してください:"
echo ""
echo "  cd ${PROJECT_ROOT}"
echo "  python -m venv venv"
echo "  source venv/bin/activate  # Linux/Mac"
echo "  # または: venv\\Scripts\\activate  # Windows"
echo ""
echo "  pip install -r requirements.txt"
echo ""
