#!/bin/bash
# pod_bootstrap.sh: on a fresh RunPod pod (runpod/base:1.4.0-ubuntu2404 or a PyTorch image), unpack the bundle from
# pack.sh into /root/dz, install Rust 1.94.1, clone and patch mtg-kernel, build dzk, and (TORCH=1) make a venv with
# torch for training. Idempotent; logs to /root/bootstrap.log.
#   scp /tmp/dzk_bundle.tgz root@HOST:/root/ && ssh root@HOST 'bash -s' < mtgkernel/deploy/pod_bootstrap.sh
set -euo pipefail
exec > >(tee -a /root/bootstrap.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
mkdir -p /root/dz && tar xzf /root/dzk_bundle.tgz -C /root/dz
if ! command -v cc > /dev/null || ! command -v git > /dev/null; then
  apt-get update -qq && apt-get install -y -qq build-essential git curl ca-certificates python3-venv python3-pip pkg-config > /dev/null
fi
if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  # rustup's official installer (https://sh.rustup.rs); the toolchain matches mtg-kernel's rust-toolchain.toml
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.94.1
fi
. "$HOME/.cargo/env"
cd /root/dz/mtgkernel
MK_SRC=${MK_SRC:-https://github.com/jackmaiorino/mtg-kernel.git} bash setup.sh
time cargo build --release
./target/release/dzk --help | head -5
lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz' || true
cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || true
if [ "${TORCH:-0}" = 1 ]; then
  [ -d /root/venv ] || python3 -m venv /root/venv
  /root/venv/bin/pip install -q --upgrade pip
  /root/venv/bin/pip install -q torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
  /root/venv/bin/pip install -q numpy
  /root/venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
fi
echo BOOTSTRAP_DONE
