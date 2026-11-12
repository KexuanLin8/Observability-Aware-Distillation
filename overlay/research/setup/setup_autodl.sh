#!/usr/bin/env bash
# One-time environment setup for HAIC on an AutoDL RTX 4090 instance (Ubuntu 22.04 image).
#
#   bash /root/autodl-tmp/haic/HAIC/research/setup/setup_autodl.sh 2>&1 | tee /root/autodl-tmp/haic/setup.log
#
# Expects the HAIC repo (research branch) already at $HAIC_HOME/HAIC.
# Everything large (conda env, pip cache, Isaac Sim, Isaac Lab, ~/.cache) goes to the data
# disk /root/autodl-tmp; the 30 GB system disk is too small for Isaac Sim.
# Idempotent: finished steps are recorded in $HAIC_HOME/.setup/ and skipped on re-run.
# It does NOT accept the Isaac Sim EULA for you (see the final step).
set -euo pipefail

HAIC_HOME="${HAIC_HOME:-/root/autodl-tmp/haic}"
REPO="$HAIC_HOME/HAIC"
ENV_PREFIX="$HAIC_HOME/env"
ISAACLAB="$HAIC_HOME/IsaacLab"
ISAACLAB_TAG="v2.3.2"
MARK="$HAIC_HOME/.setup"
mkdir -p "$MARK" "$HAIC_HOME/cache"
cd "$HAIC_HOME"  # the caller's directory may be a deleted, re-extracted repo
export PIP_CACHE_DIR="$HAIC_HOME/cache/pip"
export CONDA_PKGS_DIRS="$HAIC_HOME/cache/conda-pkgs"

log()  { echo -e "\n==== [$(date +%H:%M:%S)] $*"; }
done_() { [ -f "$MARK/$1" ]; }
mark() { touch "$MARK/$1"; }

[ -d "$REPO/research" ] || { echo "Missing $REPO/research: unpack the HAIC research repo there first."; exit 1; }
# the archive is built on Windows; files extracted as root keep a foreign uid, which git rejects
chown -R root:root "$REPO" 2>/dev/null || true

# AutoDL academic network acceleration (GitHub / pypi.nvidia.com / download.pytorch.org).
# It sets an http(s) proxy that slows down the domestic pip/apt mirrors, so those hosts
# are added to no_proxy.
if [ -f /etc/network_turbo ]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/network_turbo || true
  set -u
  mirror_hosts=$( { pip config list 2>/dev/null; cat /etc/pip.conf /root/.pip/pip.conf /root/.config/pip/pip.conf 2>/dev/null;
                    cat /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null; } |
                  grep -oE 'https?://[^/ "]+' | sed -E 's#https?://##' | sort -u | tr '\n' ',' || true )
  export no_proxy="${no_proxy:-},${mirror_hosts}localhost,127.0.0.1"
  export NO_PROXY="$no_proxy"
  echo "no_proxy=$no_proxy"
fi

log "0. checks"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
ldd --version 2>&1 | sed -n 1p || true
df -h /root/autodl-tmp | tail -1
avail_gb=$(df -BG --output=avail /root/autodl-tmp | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt 60 ]; then echo "Need >= 60 GB free on /root/autodl-tmp, have ${avail_gb} GB"; exit 1; fi

if ! done_ home_cache; then
  log "1. move ~/.cache to the data disk (pip / Omniverse / warp / torch caches)"
  mkdir -p "$HAIC_HOME/cache/home-cache"
  if [ -d /root/.cache ] && [ ! -L /root/.cache ]; then
    cp -a /root/.cache/. "$HAIC_HOME/cache/home-cache/" 2>/dev/null || true
    rm -rf /root/.cache
  fi
  [ -L /root/.cache ] || ln -s "$HAIC_HOME/cache/home-cache" /root/.cache
  for d in /root/.local/share/ov /root/.nvidia-omniverse; do
    mkdir -p "$HAIC_HOME/cache/$(basename "$d")" "$(dirname "$d")"
    if [ -d "$d" ] && [ ! -L "$d" ]; then cp -a "$d/." "$HAIC_HOME/cache/$(basename "$d")/"; rm -rf "$d"; fi
    [ -L "$d" ] || ln -s "$HAIC_HOME/cache/$(basename "$d")" "$d"
  done
  mark home_cache
fi

if ! done_ apt; then
  log "2. system packages"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git git-lfs tmux htop zip unzip cmake build-essential \
    libglu1-mesa libgl1 libegl1 libgles2 libglvnd0 libvulkan1 vulkan-tools \
    libxrandr2 libxinerama1 libxcursor1 libxi6 libsm6 libice6 libxext6 libxt6 libxkbcommon0
  git lfs install --skip-repo
  mark apt
fi

if ! done_ icd; then
  log "3. NVIDIA Vulkan / EGL ICD files (needed by headless Isaac Sim)"
  if ! ldconfig -p | grep libGLX_nvidia.so.0 >/dev/null; then
    echo "WARNING: libGLX_nvidia.so.0 not found. The instance image does not expose NVIDIA graphics"
    echo "         libraries; Isaac Sim will likely fail. Recreate the instance from a standard AutoDL image."
  fi
  if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
    mkdir -p /usr/share/vulkan/icd.d
    cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}
EOF
    echo "wrote /usr/share/vulkan/icd.d/nvidia_icd.json"
  fi
  if ! ls /usr/share/glvnd/egl_vendor.d/10_nvidia.json >/dev/null 2>&1; then
    mkdir -p /usr/share/glvnd/egl_vendor.d
    cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}
EOF
    echo "wrote /usr/share/glvnd/egl_vendor.d/10_nvidia.json"
  fi
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverVersion" || echo "WARNING: vulkaninfo did not list a GPU"
  mark icd
fi

CONDA_BASE="$(conda info --base 2>/dev/null || echo /root/miniconda3)"
set +u  # conda's activation scripts reference unset variables
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! done_ conda; then
  log "4. conda env (python 3.11) at $ENV_PREFIX"
  # conda-forge only: avoids the Anaconda default-channel terms-of-service prompt
  conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.11
  mark conda
fi
conda activate "$ENV_PREFIX"
set -u
python -m pip install --upgrade pip

if ! done_ torch; then
  log "5. PyTorch 2.7.0 + CUDA 12.8"
  pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
  python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
  mark torch
fi

if ! done_ isaacsim; then
  log "6. Isaac Sim 5.1.0 (pip, several GB)"
  pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
  mark isaacsim
fi

if ! done_ isaaclab; then
  log "7. Isaac Lab $ISAACLAB_TAG"
  if [ ! -d "$ISAACLAB/.git" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
  fi
  (cd "$ISAACLAB" && ./isaaclab.sh --install none) || echo "WARNING: isaaclab.sh returned an error; installing the core package directly below"
  # isaaclab.sh swallows per-extension pip failures, so install the core package explicitly
  # (errors now stop this script). Old sdists among its dependencies (e.g. flatdict) still
  # import pkg_resources at build time, hence setuptools 80 and no build isolation.
  pip install "setuptools==80.9.0" wheel toml
  pip install --no-build-isolation -e "$ISAACLAB/source/isaaclab"
  python -c "import isaaclab, warp; print('isaaclab OK at', isaaclab.__file__, '| warp', warp.__version__)"
  mark isaaclab
fi

if ! done_ haic; then
  log "8. HAIC (editable) + known-good pins"
  (cd "$REPO" && git lfs pull 2>/dev/null || true)
  # the USD assets are git-lfs files; make sure they are real files, not pointers
  if grep -a "git-lfs.github.com" "$REPO/active_adaptation/assets/objects/skateboard/skateboard.usd" >/dev/null; then
    echo "ERROR: USD assets are git-lfs pointers. Use the full source archive or run 'git lfs pull' with network."
    exit 1
  fi
  pip install -e "$REPO"
  # torch must stay at the CUDA 12.8 build that Isaac Sim 5.1 expects
  pip install -r "$REPO/research/setup/pins.txt"
  pip install pytest  # for research/tests
  python -c "import active_adaptation; from hydra.core.config_store import ConfigStore; assert 'ppo_haic_train.yaml' in ConfigStore.instance().repo['algo']; print('HAIC import and algo registration OK')"
  python -c "import torch; assert torch.__version__.startswith('2.7.0') and torch.version.cuda == '12.8', torch.__version__"
  mark haic
fi

log "9. package consistency"
pip check || echo "NOTE: pip check reported conflicts above; compare with research/setup/known_good_freeze.txt"
python - <<'EOF'
from importlib.metadata import version
for n in ("torch", "isaacsim", "isaaclab", "tensordict", "torchrl", "numpy", "hydra-core", "warp-lang"):
    try: print(f"  {n:12s} {version(n)}")
    except Exception as e: print(f"  {n:12s} MISSING ({e})")
EOF

log "setup finished"
cat <<EOF

Next steps (see research/README.md):
  source $REPO/research/setup/haic_env.sh
  # read the NVIDIA Isaac Sim EULA; only if you accept it:
  export OMNI_KIT_ACCEPT_EULA=YES
  python research/pipeline.py --task skateboard --arm dyn --smoke
EOF
