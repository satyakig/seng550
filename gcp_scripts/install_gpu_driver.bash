#!/bin/bash

set -e -x

# Detect NVIDIA GPU
apt-get update
apt-get install -y pciutils
if ! (lspci | grep -q NVIDIA); then
  echo 'No NVIDIA card detected. Skipping installation.' >&2
  exit 0
fi

# Add non-free Debian 9 Stretch packages.
# See https://www.debian.org/distrib/packages#note
for type in deb deb-src; do
  for distro in stretch stretch-backports; do
    for component in contrib non-free; do
      echo "${type} http://deb.debian.org/debian/ ${distro} ${component}" \
          >> /etc/apt/sources.list.d/non-free.list
    done
  done
done
apt-get update

# Install proprietary NVIDIA Drivers and CUDA
# See https://wiki.debian.org/NvidiaGraphicsDrivers
export DEBIAN_FRONTEND=noninteractive
apt-get install -y "linux-headers-$(uname -r)"
# Without --no-install-recommends this takes a very long time.
apt-get install -y -t stretch-backports --no-install-recommends \
  nvidia-cuda-toolkit nvidia-kernel-common nvidia-driver nvidia-smi

# Create a system wide NVBLAS config
# See http://docs.nvidia.com/cuda/nvblas/
NVBLAS_CONFIG_FILE=/etc/nvidia/nvblas.conf
cat << EOF >> ${NVBLAS_CONFIG_FILE}
# Insert here the CPU BLAS fallback library of your choice.
# The standard libblas.so.3 defaults to OpenBLAS, which does not have the
# requisite CBLAS API.
NVBLAS_CPU_BLAS_LIB /usr/lib/libblas/libblas.so

# Use all GPUs
NVBLAS_GPU_LIST ALL

# Add more configuration here.
EOF
echo "NVBLAS_CONFIG_FILE=${NVBLAS_CONFIG_FILE}" >> /etc/environment

# Rebooting during an initialization action is not recommended, so just
# dynamically load kernel modules. If you want to run an X server, it is
# recommended that you schedule a reboot to occur after the initialization
# action finishes.
modprobe -r nouveau
modprobe nvidia-current
modprobe nvidia-drm
modprobe nvidia-uvm
modprobe drm

# Restart any NodeManagers so they pick up the NVBLAS config.
if systemctl status hadoop-yarn-nodemanager; then
  systemctl restart hadoop-yarn-nodemanager
fi
