# P2: Development Environment Setup

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** HIGH - Do Second (parallel with hardware setup)

## Goal

Document the complete development environment setup for both PCs so that OutterLink can be built, tested, and run. This becomes a guide in `docs/guides/`.

## Milestone

- Both PCs running Linux with NVIDIA drivers + CUDA
- ConnectX-5 cards installed with RDMA working between PCs
- Rust toolchain installed and builds OutterLink workspace
- NVLink verified (on 3090 Ti pairs)
- All verification tests pass (Section D of Final Pre-Plan)

## Prerequisites

- [x] Hardware inventory documented
- [ ] Physical hardware available and assembled
- [ ] SFP modules / DAC cables purchased

---

## 1. Linux Distribution

### Recommendation: Ubuntu 24.04 LTS

| Factor | Ubuntu 24.04 | Fedora 41 | Arch |
|--------|-------------|-----------|------|
| NVIDIA driver support | Excellent (official .run or PPA) | Good (RPM Fusion) | Good (AUR) |
| MLNX_OFED support | Official support | Limited | Community |
| Kernel version | 6.8 (HWE available for newer) | ~6.11 | Rolling (latest) |
| Stability | HIGH | MEDIUM | MEDIUM |
| CUDA toolkit | Official .deb packages | .run installer | AUR |
| io_uring ZC recv | Needs HWE kernel 6.15+ | May need update | Likely available |

**Ubuntu 24.04 LTS** is the safest bet: official NVIDIA + Mellanox support, stable, well-documented.

**io_uring note:** Zero-copy recv needs kernel 6.15+. Ubuntu 24.04 ships 6.8. Options:
- Install HWE kernel (when 6.15 lands in HWE)
- Use mainline kernel PPA
- Start with regular TCP, add io_uring ZC later

### Installation Checklist

- [ ] Download Ubuntu 24.04 LTS Server (minimal, no desktop needed)
- [ ] Install on both PCs
- [ ] Configure static IPs on ConnectX-5 interfaces
- [ ] Set hostname (`outterlink-pc1`, `outterlink-pc2`)
- [ ] Enable SSH for remote access
- [ ] Disable sleep/hibernate

---

## 2. NVIDIA Driver + CUDA

### Installation Order (CRITICAL)

```
1. Install MLNX_OFED FIRST
2. Install NVIDIA driver SECOND
3. Install CUDA toolkit THIRD
```

If you install NVIDIA before MLNX_OFED, you must reinstall the NVIDIA driver after MLNX_OFED to get nvidia-peermem working.

### NVIDIA Driver

```bash
# Option A: Official .run installer (recommended for control)
# Download from nvidia.com/drivers
sudo sh NVIDIA-Linux-x86_64-560.xx.xx.run

# Option B: PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-560
```

**Minimum driver version:** 550+ (for CUDA 12.x, nvidia-peermem, ReBAR)

### CUDA Toolkit

```bash
# Download from developer.nvidia.com/cuda-downloads
# Choose: Linux > x86_64 > Ubuntu > 24.04 > deb (network)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
```

**Target:** CUDA 12.6+ (latest stable)

### Post-Install

```bash
# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvidia-smi              # Shows all GPUs
nvcc --version           # Shows CUDA version
cuda-samples/deviceQuery # Compile and run - should show all GPUs
```

---

## 3. ConnectX-5 + MLNX_OFED

### MLNX_OFED Installation

```bash
# Download from nvidia.com/networking/infiniband-sw
# Choose: MLNX_OFED > LTS > Ubuntu 24.04 > x86_64

# Install
sudo ./mlnxofedinstall --add-kernel-support --without-fw-update
sudo /etc/init.d/openibd restart

# Verify
ibv_devices              # Should show mlx5_0, mlx5_1, etc.
ibv_devinfo              # Detailed device info
```

### Network Configuration

```bash
# Identify ConnectX-5 interfaces
ip link show | grep -i mlx

# Assign static IPs (example for direct cable)
# PC1:
sudo ip addr add 192.168.100.1/24 dev enp1s0f0  # Port 1
sudo ip addr add 192.168.101.1/24 dev enp1s0f1  # Port 2

# PC2:
sudo ip addr add 192.168.100.2/24 dev enp1s0f0
sudo ip addr add 192.168.101.2/24 dev enp1s0f1

# Set MTU to 9000 (jumbo frames for better throughput)
sudo ip link set enp1s0f0 mtu 9000
sudo ip link set enp1s0f1 mtu 9000
```

### RDMA Verification

```bash
# On PC1 (server):
rping -s -v -C 3

# On PC2 (client):
rping -c -v -C 3 -a 192.168.100.1

# Bandwidth test:
# PC1: ib_write_bw -d mlx5_0
# PC2: ib_write_bw -d mlx5_0 192.168.100.1
```

### Link Bonding (Optional, for higher bandwidth)

```bash
# Bond two 100GbE ports into one 200Gbps link
sudo modprobe bonding
# Create bond interface with mode 802.3ad (LACP) or balance-rr
# Detailed bonding config TBD based on switch/direct cable setup
```

---

## 4. BIOS Configuration

### Both PCs

| Setting | Value | Why |
|---------|-------|-----|
| Above 4G Decoding | **Enabled** | Required for ReBAR and large BAR1 |
| Resizable BAR | **Enabled** | Exposes full GPU VRAM through BAR1 (OpenDMA) |
| IOMMU | **Disabled** or Passthrough | Required for PCIe P2P DMA |
| CSM | **Disabled** | Required for ReBAR |
| SR-IOV | Optional | Not needed for Phase 1 |
| Boot mode | UEFI | Required for ReBAR |

### Kernel Boot Parameters

```bash
# Add to /etc/default/grub GRUB_CMDLINE_LINUX:
# For Intel CPU:
intel_iommu=off

# For AMD CPU (Threadripper):
amd_iommu=off
# OR for passthrough mode (less disruptive):
iommu=pt

# Apply:
sudo update-grub
sudo reboot
```

---

## 5. Rust Toolchain

```bash
# Install rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install stable toolchain with components
rustup default stable
rustup component add clippy rustfmt

# Verify
cargo --version
rustc --version
```

### Development Tools

```bash
# Useful Rust tools
cargo install cargo-watch     # Auto-rebuild on file changes
cargo install cargo-expand    # Expand macros for debugging
cargo install cargo-audit     # Security audit of dependencies
```

---

## 6. NVLink Verification

```bash
# Check NVLink topology
nvidia-smi topo -m

# Expected output should show NV# for NVLink-connected GPU pairs
# Example:
#         GPU0  GPU1
# GPU0     X    NV12
# GPU1    NV12   X

# NVLink bandwidth test
# Use cuda-samples p2pBandwidthLatencyTest
cd /usr/local/cuda/samples/1_Utilities/p2pBandwidthLatencyTest
make
./p2pBandwidthLatencyTest
```

---

## 7. Verification Checklist

Run all of these after setup is complete:

| # | Test | Command | Expected Result |
|---|------|---------|----------------|
| V1 | GPUs visible | `nvidia-smi` | All GPUs listed with correct VRAM |
| V2 | CUDA works | `deviceQuery` (CUDA sample) | PASS for all GPUs |
| V3 | NVLink active | `nvidia-smi topo -m` | NV# links between paired GPUs |
| V4 | ConnectX-5 visible | `ibv_devices` | mlx5 devices listed |
| V5 | RDMA works | `rping -s` / `rping -c` | Connection success |
| V6 | Network bandwidth | `iperf3 -c <IP>` | >10 GB/s per 100GbE link |
| V7 | RDMA bandwidth | `ib_write_bw` | >10 GB/s |
| V8 | PCIe topology | `lspci -tv` | GPU + NIC under same root complex |
| V9 | ReBAR active | `nvidia-smi -q \| grep BAR1` | BAR1 Memory = full VRAM size |
| V10 | IOMMU disabled | `dmesg \| grep -i iommu` | No IOMMU messages or "passthrough" |
| V11 | Rust builds | `cargo build --all` (in outterlink repo) | Compiles |
| V12 | nvidia-peermem loaded | `lsmod \| grep nvidia_peermem` | Module loaded |

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| nvidia-smi not found | Reinstall NVIDIA driver, check PATH |
| ibv_devices shows nothing | Reinstall MLNX_OFED, check `lspci` for ConnectX |
| rping fails | Check IPs, firewall (`ufw disable`), check cables |
| ReBAR shows 256MB | Enable Above 4G Decoding + ReBAR in BIOS, disable CSM |
| NVLink not showing | Check physical bridge, check riser compatibility |
| RDMA bandwidth low | Check MTU (9000), check PCIe topology, disable IOMMU |
| CUDA P2P fails | Check IOMMU is disabled, check PCIe ACS |
| nvidia-peermem won't load | Reinstall NVIDIA driver AFTER MLNX_OFED |

---

## Risks

| Risk | Mitigation |
|------|-----------|
| MLNX_OFED version incompatible with kernel | Use LTS version, check compatibility matrix |
| NVIDIA driver conflicts with MLNX_OFED | Follow install order (MLNX_OFED first) |
| ReBAR not supported on MS-01 Ultra | Check BIOS options, update BIOS if needed |
| Kernel too old for io_uring ZC | Start with regular TCP, upgrade kernel later |

## Related Documents

- [Hardware Inventory](../pre-planning/01-hardware-inventory.md)
- [Final Pre-Plan](../pre-planning/02-FINAL-PREPLAN.md)
- [R4: ConnectX-5 + Transport](../research/R4-connectx5-transport-stack.md)
