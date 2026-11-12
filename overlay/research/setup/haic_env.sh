# Source this in every new shell on the server:
#   source /root/autodl-tmp/haic/HAIC/research/setup/haic_env.sh
export HAIC_HOME="${HAIC_HOME:-/root/autodl-tmp/haic}"
export HAIC_OUT="${HAIC_OUT:-$HAIC_HOME/runs}"
export CONDA_PKGS_DIRS="$HAIC_HOME/cache/conda-pkgs"
export WANDB_MODE=offline
mkdir -p "$HAIC_OUT"

_conda_base="$(conda info --base 2>/dev/null || echo /root/miniconda3)"
# shellcheck disable=SC1091
source "$_conda_base/etc/profile.d/conda.sh"
conda activate "$HAIC_HOME/env"
cd "$HAIC_HOME/HAIC" || return

if [ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" ]; then
  echo "[haic_env] OMNI_KIT_ACCEPT_EULA is not set. Read the NVIDIA Isaac Sim EULA; if you accept it, run:"
  echo "           export OMNI_KIT_ACCEPT_EULA=YES"
fi
