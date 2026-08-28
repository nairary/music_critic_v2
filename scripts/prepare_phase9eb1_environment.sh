#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PYTHON_3_12_EXECUTABLE ENVIRONMENT_DIRECTORY" >&2
  exit 2
fi

python_executable=$1
environment_directory=$2
repository_root=$(git rev-parse --show-toplevel)
source_root="$environment_directory/sources"

python_version=$("$python_executable" -c 'import platform; print(platform.python_version())')
if [[ "$python_version" != "3.12.8" ]]; then
  echo "Phase 9E-B1 requires Python 3.12.8, observed $python_version" >&2
  exit 2
fi
if ! command -v cc >/dev/null 2>&1; then
  echo "Phase 9E-B1 requires a C compiler available as cc" >&2
  exit 2
fi
if [[ -e "$environment_directory" ]]; then
  echo "refusing to reuse existing environment directory: $environment_directory" >&2
  exit 2
fi

"$python_executable" -m venv "$environment_directory/venv"
"$environment_directory/venv/bin/python" -m pip install --upgrade pip==24.2 wheel==0.44.0 setuptools==75.1.0
"$environment_directory/venv/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 torch==2.2.2
"$environment_directory/venv/bin/python" -m pip install torch-geometric==2.6.1
"$environment_directory/venv/bin/python" -m pip install \
  --find-links https://data.pyg.org/whl/torch-2.2.0+cu118.html \
  torch-scatter==2.1.2+pt22cu118 torch-sparse==0.6.18+pt22cu118 \
  torch-cluster==1.6.3+pt22cu118 torch-spline-conv==1.2.2+pt22cu118 \
  pyg-lib==0.4.0+pt22cu118
"$environment_directory/venv/bin/python" -m pip install \
  numpy==1.26.4 pandas==2.2.3 scipy==1.14.1 scikit-learn==1.6.0 \
  partitura==1.6.0 pytorch-lightning==2.5.0.post0 torchmetrics==1.6.0 \
  wandb==0.19.1 pyyaml==6.0.2

mkdir -p "$source_root"
git clone https://github.com/manoskary/analysisgnn "$source_root/analysisgnn"
git -C "$source_root/analysisgnn" checkout --detach e115182fb29b74bdcb6bf3547ed427d967580947
git clone https://github.com/manoskary/graphmuse "$source_root/graphmuse"
git -C "$source_root/graphmuse" checkout --detach c36eedba811a24c0addf96bdd3d1df449cf753c1

for patch in "$repository_root"/analysisgnn_patches/*.patch; do
  git -C "$source_root/analysisgnn" apply --check --unidiff-zero "$patch"
  git -C "$source_root/analysisgnn" apply --unidiff-zero "$patch"
done

"$environment_directory/venv/bin/python" -m pip install --no-deps -e "$source_root/graphmuse"
"$environment_directory/venv/bin/python" -m pip install \
  --no-deps --ignore-requires-python -e "$source_root/analysisgnn"
"$environment_directory/venv/bin/python" -m pip install \
  --no-deps -e "$repository_root"
"$environment_directory/venv/bin/python" -m pip freeze > "$environment_directory/resolved-freeze.txt"

{
  git -C "$source_root/analysisgnn" rev-parse HEAD
  git -C "$source_root/graphmuse" rev-parse HEAD
  sha256sum "$repository_root"/analysisgnn_patches/*.patch
} > "$environment_directory/source-and-patch-identities.txt"

git -C "$source_root/analysisgnn" diff --check
git -C "$source_root/analysisgnn" diff > "$environment_directory/applied-analysisgnn.patch"
