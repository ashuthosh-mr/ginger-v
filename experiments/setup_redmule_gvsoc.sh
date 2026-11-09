#!/usr/bin/env bash
# setup_redmule_gvsoc.sh
# ---------------------------------------------------------------------------
# Reproduce the working RedMulE simulation environment, starting from the stock
# public image `ghcr.io/pulp-platform/deeploy:main`.
#
# WHY: that public image is missing (a) the i3c pulp-sdk sources and (b) the
# RedMulE model in GVSoC. The correct CI image (ghcr.io/runwangdl/deeploy:redmule)
# is private. This script builds just the missing GVSoC-with-RedMulE from the
# PUBLIC gvsoc fork, which is enough to run RedMulE tests.
#
# USAGE (run INSIDE the container, with ~/revolution mounted at /workspace):
#   docker run -it -v ~/revolution:/workspace ghcr.io/pulp-platform/deeploy:main bash
#   bash /workspace/ginger-v/experiments/setup_redmule_gvsoc.sh
#
# Then run a test:
#   cd /workspace/ginger-v/Deeploy/DeeployTest
#   source /app/install/pulp-sdk/configs/siracusa.sh
#   export PYTHONPATH=/workspace/ginger-v/Deeploy
#   python3 testRunner_tiled_siracusa_w_redmule.py -t Tests/testFloatGEMM --cores=8
# ---------------------------------------------------------------------------
set -e

GVSOC_COMMIT=35d00d15d7249daaac0de61bd8485fba128e5959
DEEPLOY=/workspace/ginger-v/Deeploy
CMAKE=$DEEPLOY/TargetLibraries/PULPOpen/cmake/pulp-sdk-siracusa.cmake

echo "### [1/4] host C++ compiler"
if ! command -v g++ >/dev/null; then apt-get update -qq && apt-get install -y -qq g++ build-essential; fi
g++ --version | head -1

echo "### [2/4] patch Deeploy cmake: comment out i3c sources missing from this SDK"
# For any line referencing drivers/i3c/src/ that is NOT already commented, prefix with '#'
sed -i -E '/drivers\/i3c\/src\//{/^[[:space:]]*#/!s/^([[:space:]]*)/\1#/}' "$CMAKE"
echo "i3c source lines now:"; grep -nE "i3c/src" "$CMAKE" || true

echo "### [3/4] clone + build GVSoC with RedMulE model (public fork)"
mkdir -p /app/toolchain && cd /app/toolchain && rm -rf gvsoc
git clone -q https://github.com/runwangdl/gvsoc.git
cd gvsoc && git checkout -q "$GVSOC_COMMIT" && git submodule update --init --recursive -q
pip install -q -r core/requirements.txt -r gapy/requirements.txt || true
XTENSOR_INSTALL_DIR=/app/install/xtensor/include \
XTL_INSTALL_DIR=/app/install/xtl/include \
XSIMD_INSTALL_DIR=/app/install/xsimd/include \
  make all TARGETS="pulp.snitch.snitch_cluster_single siracusa" build INSTALLDIR=/app/install/gvsoc

echo "### [4/4] verify RedMulE model is present"
ls /app/install/gvsoc/models/gen_pulp_chips_siracusa_light_redmule_cpp_*.so \
  && echo "OK: RedMulE GVSoC model built." \
  || { echo "FAILED: no RedMulE model found"; exit 1; }

echo "Done. Now run a test (see header of this script)."
