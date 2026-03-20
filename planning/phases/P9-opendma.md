# P9: OpenDMA -- Non-Proprietary Direct NIC-to-GPU VRAM Access

**Created:** 2026-03-19
**Last Updated:** 2026-03-19
**Status:** Plan
**Priority:** CRITICAL -- Killer Feature

## Goal

Implement OpenDMA: a non-proprietary mechanism for direct RDMA between ConnectX-5 NIC and GPU VRAM via PCIe BAR1, bypassing NVIDIA's artificial GPUDirect RDMA restriction. This enables zero-copy, zero-CPU NIC-to-GPU transfers on ALL NVIDIA GPUs including GeForce, achieving ~2us latency vs ~12us with host-staged transfers.

## Milestone

- Custom kernel module (`opendma.ko`) loaded and functional
- ConnectX-5 performs RDMA WRITE directly into GPU VRAM via BAR1
- Data integrity verified: CUDA write -> RDMA read and RDMA write -> CUDA read both produce correct data
- Bandwidth within 80% of theoretical PCIe limit (~10+ GB/s on 100GbE)
- OpenDMA transport backend integrated into OutterLink, selectable at runtime
- Graceful fallback to host-staged when OpenDMA hardware requirements not met

## Prerequisites

- [x] R5: GPUDirect restriction analysis complete
- [x] R7: Non-proprietary GPU DMA research complete
- [x] ADR-002: OpenDMA naming decided
- [ ] P2: Development environment fully operational (Linux, NVIDIA driver, MLNX_OFED, CUDA)
- [ ] P4: Rust workspace with Transport trait defined
- [ ] P8: Host-staged RDMA transport working (baseline for comparison)
- [ ] Hardware verified: ReBAR enabled, IOMMU off, GPU+NIC same root complex
- [ ] tinygrad patches compatibility with our CUDA/driver version confirmed (R12)

---

## Architecture

### Complete Data Flow

```
SEND PATH (local GPU VRAM -> remote GPU VRAM):

  Local GPU VRAM
       |
       | (GPU MMU page tables map VRAM -> BAR1 aperture)
       | (tinygrad patch: kbusEnableStaticBar1Mapping maps all 24GB)
       v
  GPU BAR1 (PCIe Base Address Register 1)
       |
       | (standard PCIe Memory Read TLP)
       | (no proprietary API -- just PCIe bus transactions)
       v
  PCIe Root Complex
       |
       | (peer-to-peer if same root complex, or via CPU if different)
       v
  ConnectX-5 DMA Engine
       |
       | (RDMA WRITE verb, source = GPU BAR1 physical address)
       | (mlx5 driver creates WQE with BAR1 DMA address)
       v
  100GbE Wire (RoCE v2)
       |
       v
  Remote ConnectX-5 DMA Engine
       |
       | (RDMA WRITE, destination = remote GPU BAR1 physical address)
       v
  Remote PCIe Root Complex
       |
       v
  Remote GPU BAR1
       |
       | (remote GPU MMU translates BAR1 -> VRAM)
       v
  Remote GPU VRAM


RECEIVE PATH: Identical in reverse.
```

### Kernel Module Stack

```
 Userspace
 =========================================================
   OutterLink Server (Rust)
      |
      | ioctl(opendma_fd, OPENDMA_REGISTER_REGION, ...)
      | ioctl(opendma_fd, OPENDMA_GET_MR_INFO, ...)
      |
 ---------------------------------------------------------
 Kernel Space
 =========================================================
      |
      v
   opendma.ko (OUR MODULE)
      |
      |-- Finds GPU via PCI subsystem (vendor=0x10de)
      |-- Reads BAR1 physical address: pci_resource_start(gpu_pdev, 1)
      |-- Reads BAR1 size: pci_resource_len(gpu_pdev, 1)
      |-- Registers BAR1 memory with RDMA subsystem via ONE of:
      |     Option A: ib_register_peer_memory_client() [MLNX_OFED]
      |     Option B: pci_p2pdma_add_resource() [upstream kernel]
      |     Option C: manual ioremap + custom DMA mapping
      |
      v
   mlx5_ib (ConnectX-5 RDMA driver, part of MLNX_OFED)
      |
      |-- Creates Memory Region (MR) backed by GPU BAR1 pages
      |-- Posts RDMA WRITE/READ work requests targeting BAR1
      |
      v
   mlx5_core (ConnectX-5 core driver)
      |
      |-- Programs NIC DMA engine with BAR1 physical addresses
      |-- NIC issues PCIe Memory Write TLPs to GPU BAR1
      |
      v
   nvidia.ko / nvidia-open.ko (PATCHED with tinygrad BAR1 mapping)
      |
      |-- GPU MMU page tables configured to map BAR1 -> all VRAM
      |-- kbusEnableStaticBar1Mapping_GH100 called during init
      |-- Incoming PCIe writes to BAR1 land in correct VRAM locations
```

### Userspace vs Kernelspace Responsibilities

| Layer | Responsibility |
|-------|---------------|
| **Userspace (Rust)** | Detect OpenDMA availability, open /dev/opendma, register GPU memory regions, obtain RDMA keys, create QP/CQ via libibverbs, post RDMA work requests |
| **opendma.ko** | Map GPU BAR1 to RDMA-accessible memory, provide ioctl interface for region registration, handle invalidation callbacks |
| **Patched nvidia.ko** | GPU MMU page table setup (BAR1 -> VRAM mapping), CUDA compute (unchanged) |
| **mlx5 drivers** | RDMA operations using registered GPU BAR1 memory |

---

## Plan A: tinygrad Patches + Custom RDMA Module

This is the primary approach. It has the highest probability of working on our hardware.

### Step 1: Obtain and Apply tinygrad Patches

**Source repositories:**
- Primary: `github.com/tinygrad/open-gpu-kernel-modules` (branch matching our driver version)
- Forks with broader GPU support: `github.com/aikitoria/open-gpu-kernel-modules`
- RTX 3090 Ti specific: `github.com/forkProj/open-gpu-kernel-modules-P2P`

**Process:**

```bash
# 1. Identify current NVIDIA driver version
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# Expected: 550.x or 570.x

# 2. Clone the matching tinygrad fork
# For driver 550.x:
git clone -b 550.90.07-p2p https://github.com/tinygrad/open-gpu-kernel-modules.git
# For driver 570.x:
git clone -b 570.124.04-p2p https://github.com/aikitoria/open-gpu-kernel-modules.git

# 3. Build the patched kernel modules
cd open-gpu-kernel-modules
make modules -j$(nproc)

# 4. Unload existing NVIDIA modules (order matters)
sudo rmmod nvidia_peermem   # if loaded
sudo rmmod nvidia_uvm
sudo rmmod nvidia_modeset
sudo rmmod nvidia

# 5. Install patched modules
sudo make modules_install
sudo depmod -a

# 6. Load patched modules
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo modprobe nvidia_modeset

# 7. Verify GPU still works
nvidia-smi
# Run a CUDA sample to verify compute is functional
./deviceQuery
```

### Step 2: Understanding kbusEnableStaticBar1Mapping

This is the critical function that makes OpenDMA possible. Here is what it does at the hardware level:

**What happens WITHOUT the patch (stock driver):**
1. GPU BAR1 is a 256MB (or 24GB with ReBAR) PCIe aperture
2. The GPU's internal MMU has page tables that map BAR1 virtual addresses to VRAM physical addresses
3. On GeForce, only a subset of VRAM is mapped through BAR1 (display framebuffer, some driver allocations)
4. CUDA allocations use BAR0 register-based access or DMA engines, NOT BAR1

**What happens WITH the patch:**
1. `kbusEnableStaticBar1Mapping_GH100` is called during driver initialization
2. It programs the GPU's GMMU (Graphics Memory Management Unit) page tables
3. Every 4KB page of VRAM gets a corresponding BAR1 page table entry
4. The mapping is: `BAR1_offset = VRAM_physical_offset` (identity mapping)
5. Result: Any PCIe device can read/write any VRAM location by targeting `BAR1_base + offset`

**Key patch modifications:**

1. **GMMU aperture type change:** The patches change `GMMU_APERTURE_PEER` to `GMMU_APERTURE_SYS_NONCOH` (system non-coherent). This tells the GPU MMU to treat peer accesses as system memory accesses that go through BAR1 instead of through NVIDIA's proprietary peer mailbox mechanism.

2. **BAR1 base address injection:** The patches put the BAR1 physical base address into the `fabricBaseAddress` field for `GMMU_APERTURE_PEER` entries. This is normally used for NVSwitch fabric addresses on datacenter GPUs. On GeForce, repurposing it for BAR1 addresses enables P2P access.

3. **Static mapping enable:** Forces `kbusEnableStaticBar1Mapping_GH100` to be called even on non-GH100 GPUs (Ampere GA102 in our case). The function itself is architecture-independent -- it programs generic GMMU page tables.

**RTX 3090 Ti (GA102) specific considerations:**

- GA102 uses the same GMMU architecture as GH100 for BAR1 mapping
- With ReBAR enabled, BAR1 = 24GB = full VRAM, so static mapping covers everything
- WITHOUT ReBAR, BAR1 = 256MB, and only 256MB of VRAM is accessible (requires windowing, which is complex and not recommended)
- The `kbusEnableStaticBar1Mapping` function checks `BAR1_size >= FB_size`. If BAR1 is 256MB and VRAM is 24GB, it FAILS with "BAR1 size is not large enough to map FB size"
- Therefore: **ReBAR is MANDATORY for OpenDMA on RTX 3090 Ti**

### Step 3: Verify BAR1 Mapping Works

After loading patched drivers:

```bash
# Check BAR1 size (must equal VRAM size for full mapping)
nvidia-smi -q | grep -A2 "BAR1"
# Expected:
#   BAR1 Memory Usage
#       Total                : 24576 MiB    <-- must be 24GB, not 256MB

# Check BAR1 physical address
sudo lspci -v -s <GPU_BDF> | grep "Memory at"
# Look for BAR1 (the second "Memory at" line, 64-bit prefetchable)
# Example: Memory at 4000000000 (64-bit, prefetchable) [size=24G]

# Verify GPU VRAM is accessible through BAR1
# Write test pattern via CUDA, read back via BAR1 ioremap (see validation section)
```

### Step 4: Test nvidia_p2p_get_pages on Patched Driver

Before writing our own module, test if the standard nvidia-peermem path works:

```bash
# Load nvidia-peermem
sudo modprobe nvidia_peermem

# Check if it loaded
lsmod | grep nvidia_peermem

# Try a GPUDirect RDMA test (this normally fails on GeForce)
# Use perftest with GPU memory:
ib_write_bw --use_cuda=0 -d mlx5_0  # server
ib_write_bw --use_cuda=0 -d mlx5_0 <server_ip>  # client
```

**If nvidia_p2p_get_pages() succeeds:** The patched driver has opened up the P2P API on GeForce. We can use nvidia-peermem as-is. This is the best case -- no custom module needed. Skip to Integration.

**If nvidia_p2p_get_pages() still fails (EXPECTED):** The product class check is in the GSP firmware or a code path not affected by BAR1 mapping patches. We proceed with our custom module.

### Step 5: Custom Kernel Module Design -- `opendma.ko`

This is the core deliverable. The module registers GPU BAR1 memory with the RDMA subsystem, bypassing nvidia_p2p_get_pages entirely.

Three sub-options for RDMA registration, ordered by preference:

---

#### Option A: Peer Memory Client (MLNX_OFED Required)

This is the most direct path. We implement the same `peer_memory_client` interface that nvidia-peermem uses, but instead of calling `nvidia_p2p_get_pages()`, we directly use the BAR1 physical address from PCI config space.

**Kernel Module: `opendma.ko`**

```c
// SPDX-License-Identifier: GPL-2.0
/*
 * OpenDMA - Non-proprietary GPU VRAM RDMA via PCIe BAR1
 *
 * Registers GPU BAR1 (VRAM) as RDMA-accessible peer memory,
 * bypassing NVIDIA's nvidia_p2p_get_pages() restriction on GeForce GPUs.
 *
 * Requires:
 *   - NVIDIA open kernel modules with tinygrad BAR1 mapping patches
 *   - ReBAR enabled (full VRAM exposed through BAR1)
 *   - MLNX_OFED with peer_memory_client support
 *   - IOMMU disabled
 */

#include <linux/module.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>
#include <rdma/peer_mem.h>

#define OPENDMA_NAME    "opendma"
#define OPENDMA_VERSION "1.0.0"
#define NVIDIA_VENDOR_ID 0x10DE

/* Represents one GPU's BAR1 region */
struct opendma_gpu {
    struct pci_dev *pdev;
    resource_size_t bar1_phys;   /* BAR1 physical base address */
    resource_size_t bar1_size;   /* BAR1 size in bytes */
    void __iomem   *bar1_iova;   /* ioremap'd BAR1 (for verification only) */
    struct list_head list;
};

/* Per-registration context (one per ib_reg_mr call) */
struct opendma_context {
    struct opendma_gpu *gpu;
    u64 vram_offset;             /* Offset into VRAM/BAR1 */
    u64 size;                    /* Size of registered region */
    unsigned long npages;
    struct page **pages;         /* Pseudo page array for DMA mapping */
    dma_addr_t *dma_addrs;      /* DMA addresses for each page */
    void *core_context;          /* IB core opaque context */
};

static LIST_HEAD(gpu_list);
static DEFINE_MUTEX(gpu_lock);
static void *peer_reg_handle;

/* ----------------------------------------------------------------
 * GPU Discovery
 * ---------------------------------------------------------------- */

/*
 * Find all NVIDIA GPUs and record their BAR1 physical addresses.
 * Called during module init.
 */
static int opendma_discover_gpus(void)
{
    struct pci_dev *pdev = NULL;
    struct opendma_gpu *gpu;
    int count = 0;

    while ((pdev = pci_get_device(NVIDIA_VENDOR_ID, PCI_ANY_ID, pdev))) {
        resource_size_t bar1_start, bar1_len;

        /* BAR1 is the VRAM aperture on NVIDIA GPUs */
        bar1_start = pci_resource_start(pdev, 1);
        bar1_len = pci_resource_len(pdev, 1);

        if (bar1_len == 0) {
            pr_debug("opendma: %s BAR1 size is 0, skipping\n",
                     pci_name(pdev));
            continue;
        }

        /* Require ReBAR: BAR1 must be at least 1GB to be useful */
        if (bar1_len < (1ULL << 30)) {
            pr_warn("opendma: %s BAR1 is only %llu MB. "
                    "Enable ReBAR in BIOS for full VRAM access.\n",
                    pci_name(pdev),
                    (unsigned long long)bar1_len >> 20);
            continue;
        }

        gpu = kzalloc(sizeof(*gpu), GFP_KERNEL);
        if (!gpu)
            return -ENOMEM;

        gpu->pdev = pdev;
        pci_dev_get(pdev);  /* Take reference */
        gpu->bar1_phys = bar1_start;
        gpu->bar1_size = bar1_len;

        /* Do NOT ioremap the entire BAR1 here.
         * We only need the physical address for DMA mappings.
         * ioremap of 24GB would consume excessive kernel VA space.
         * A small verification window can be mapped on demand. */
        gpu->bar1_iova = NULL;

        mutex_lock(&gpu_lock);
        list_add_tail(&gpu->list, &gpu_list);
        mutex_unlock(&gpu_lock);

        pr_info("opendma: Found GPU %s BAR1 at 0x%llx size %llu MB\n",
                pci_name(pdev),
                (unsigned long long)bar1_start,
                (unsigned long long)bar1_len >> 20);
        count++;
    }

    pr_info("opendma: Discovered %d GPU(s) with usable BAR1\n", count);
    return count > 0 ? 0 : -ENODEV;
}

/* Find which GPU owns a given VRAM virtual address.
 *
 * CUDA allocations return device pointers that are virtual addresses
 * within the GPU's unified address space. For BAR1-mapped VRAM,
 * we need to translate these to BAR1 physical offsets.
 *
 * With tinygrad static BAR1 mapping, the translation is:
 *   BAR1_physical = BAR1_base + VRAM_offset
 *   where VRAM_offset = cuda_device_ptr (mod VRAM size)
 *
 * The userspace component (OutterLink server) must provide the
 * VRAM offset, not the CUDA virtual address, since the kernel
 * module cannot query CUDA's address space.
 */
static struct opendma_gpu *opendma_find_gpu(int gpu_index)
{
    struct opendma_gpu *gpu;
    int i = 0;

    mutex_lock(&gpu_lock);
    list_for_each_entry(gpu, &gpu_list, list) {
        if (i == gpu_index) {
            mutex_unlock(&gpu_lock);
            return gpu;
        }
        i++;
    }
    mutex_unlock(&gpu_lock);
    return NULL;
}

/* ----------------------------------------------------------------
 * Peer Memory Client Callbacks
 * ----------------------------------------------------------------
 * These implement the interface defined in <rdma/peer_mem.h>.
 * The IB core calls these when an application registers memory
 * (via ibv_reg_mr) that we claim ownership of.
 * ---------------------------------------------------------------- */

/*
 * acquire: Called by IB core to check if a virtual address belongs to us.
 *
 * For OpenDMA, the userspace application uses a special virtual address
 * range (obtained via mmap on /dev/opendma) that we recognize.
 * Alternatively, we use a custom registration ioctl and the acquire
 * callback matches based on registered ranges.
 */
static int opendma_acquire(unsigned long addr, size_t size,
                           void *peer_mem_private_data,
                           char *peer_mem_name, void **client_context)
{
    /* We use ioctl-based registration, not acquire-based.
     * This callback returns 0 (not ours) for all addresses.
     * Instead, our ioctl directly calls ib_reg_mr with the
     * DMA addresses we provide.
     *
     * See "Alternative Registration Path" below.
     */
    return 0;
}

/*
 * get_pages: Pin the physical pages for a memory region.
 *
 * For GPU BAR1, there are no "pages" to pin -- BAR1 is MMIO space
 * with fixed physical addresses. We create a pseudo-page array
 * that the IB core can use for DMA mapping.
 */
static int opendma_get_pages(unsigned long addr, size_t size,
                             u32 write, int force,
                             struct sg_table *sg_head,
                             void *client_context,
                             u64 core_context)
{
    struct opendma_context *ctx = client_context;
    struct scatterlist *sg;
    unsigned long npages;
    int i;

    if (!ctx || !ctx->gpu)
        return -EINVAL;

    npages = DIV_ROUND_UP(size, PAGE_SIZE);
    ctx->npages = npages;
    ctx->core_context = (void *)core_context;

    /* Allocate scatter-gather table */
    if (sg_alloc_table(sg_head, npages, GFP_KERNEL))
        return -ENOMEM;

    /* Fill SG entries with BAR1 physical addresses.
     * Each entry is one PAGE_SIZE chunk of BAR1.
     * Physical address = BAR1_base + vram_offset + (page_index * PAGE_SIZE)
     */
    sg = sg_head->sgl;
    for (i = 0; i < npages; i++, sg = sg_next(sg)) {
        /* We use sg_set_page with a dummy page, then override
         * the DMA address in dma_map. This is the pattern used
         * by nvidia-peermem and io_peer_mem. */
        sg_set_page(sg, NULL, PAGE_SIZE, 0);
        sg->dma_address = ctx->gpu->bar1_phys
                        + ctx->vram_offset
                        + ((u64)i * PAGE_SIZE);
        sg->dma_length = PAGE_SIZE;
    }

    return 0;
}

/*
 * dma_map: Create DMA mappings for the registered pages.
 *
 * For BAR1, the "DMA address" IS the physical PCIe address.
 * No IOMMU translation needed (IOMMU must be disabled).
 * We just confirm the addresses are already set in the SG table.
 */
static int opendma_dma_map(struct sg_table *sg_head,
                           void *client_context,
                           struct device *dma_device,
                           int dmasync, int *nmap)
{
    struct opendma_context *ctx = client_context;

    if (!ctx)
        return -EINVAL;

    /* With IOMMU disabled, PCIe physical = DMA address.
     * The SG entries already have the correct DMA addresses
     * from get_pages. We just need to tell IB core how many
     * mappings we have. */
    *nmap = ctx->npages;
    return 0;
}

/*
 * dma_unmap: Release DMA mappings.
 */
static int opendma_dma_unmap(struct sg_table *sg_head,
                             void *client_context,
                             struct device *dma_device)
{
    /* Nothing to unmap -- BAR1 addresses are static */
    return 0;
}

/*
 * put_pages: Release pinned pages.
 */
static void opendma_put_pages(struct sg_table *sg_head,
                              void *client_context)
{
    /* Free the SG table we allocated in get_pages */
    sg_free_table(sg_head);
}

/*
 * get_page_size: Return the page size for this memory type.
 */
static unsigned long opendma_get_page_size(void *client_context)
{
    /* BAR1 is mapped in 4KB pages by the GPU MMU.
     * We could potentially use 2MB huge pages if the GPU MMU
     * supports it, but 4KB is safe and correct. */
    return PAGE_SIZE;
}

/*
 * release: Final cleanup for a registration.
 */
static void opendma_release(void *client_context)
{
    struct opendma_context *ctx = client_context;

    if (ctx) {
        kfree(ctx->dma_addrs);
        kfree(ctx->pages);
        kfree(ctx);
    }
}

static const struct peer_memory_client opendma_client = {
    .name           = OPENDMA_NAME,
    .version        = OPENDMA_VERSION,
    .acquire        = opendma_acquire,
    .get_pages      = opendma_get_pages,
    .dma_map        = opendma_dma_map,
    .dma_unmap      = opendma_dma_unmap,
    .put_pages      = opendma_put_pages,
    .get_page_size  = opendma_get_page_size,
    .release        = opendma_release,
};

/* ----------------------------------------------------------------
 * ioctl Interface for Userspace
 * ----------------------------------------------------------------
 * Userspace (OutterLink server) uses ioctls to:
 *   1. Query discovered GPUs and their BAR1 info
 *   2. Register VRAM regions for RDMA access
 *   3. Obtain rkey/lkey for RDMA verbs
 * ---------------------------------------------------------------- */

#define OPENDMA_IOC_MAGIC 'O'

struct opendma_gpu_info {
    __u32 gpu_index;
    __u64 bar1_phys;
    __u64 bar1_size;
    char  pci_bdf[16];       /* e.g., "0000:01:00.0" */
};

struct opendma_register_region {
    __u32 gpu_index;
    __u64 vram_offset;       /* Offset into VRAM (= BAR1 offset) */
    __u64 size;              /* Size in bytes */
    __u64 handle;            /* Output: registration handle */
};

#define OPENDMA_IOC_GPU_COUNT    _IOR(OPENDMA_IOC_MAGIC, 1, __u32)
#define OPENDMA_IOC_GPU_INFO     _IOWR(OPENDMA_IOC_MAGIC, 2, struct opendma_gpu_info)
#define OPENDMA_IOC_REGISTER     _IOWR(OPENDMA_IOC_MAGIC, 3, struct opendma_register_region)
#define OPENDMA_IOC_UNREGISTER   _IOW(OPENDMA_IOC_MAGIC, 4, __u64)

/* ioctl implementations omitted for brevity -- straightforward
 * switch-case on cmd with copy_from_user / copy_to_user. */

/* ----------------------------------------------------------------
 * Module Init / Exit
 * ---------------------------------------------------------------- */

static int __init opendma_init(void)
{
    int ret;

    pr_info("opendma: Initializing OpenDMA v%s\n", OPENDMA_VERSION);

    /* Discover GPUs */
    ret = opendma_discover_gpus();
    if (ret < 0) {
        pr_err("opendma: No GPUs found with usable BAR1. "
               "Ensure ReBAR is enabled in BIOS.\n");
        return ret;
    }

    /* Register as peer memory client with MLNX_OFED IB core */
    peer_reg_handle = ib_register_peer_memory_client(&opendma_client,
                                                     NULL);
    if (!peer_reg_handle) {
        pr_err("opendma: Failed to register peer memory client. "
               "Is MLNX_OFED installed?\n");
        /* Cleanup GPU list */
        return -EINVAL;
    }

    pr_info("opendma: Peer memory client registered successfully\n");

    /* Create character device /dev/opendma for ioctl interface */
    /* (misc_register or cdev_add -- implementation straightforward) */

    return 0;
}

static void __exit opendma_exit(void)
{
    struct opendma_gpu *gpu, *tmp;

    /* Unregister peer memory client */
    if (peer_reg_handle)
        ib_unregister_peer_memory_client(peer_reg_handle);

    /* Free GPU list */
    mutex_lock(&gpu_lock);
    list_for_each_entry_safe(gpu, tmp, &gpu_list, list) {
        if (gpu->bar1_iova)
            iounmap(gpu->bar1_iova);
        pci_dev_put(gpu->pdev);
        list_del(&gpu->list);
        kfree(gpu);
    }
    mutex_unlock(&gpu_lock);

    /* Destroy character device */

    pr_info("opendma: Unloaded\n");
}

module_init(opendma_init);
module_exit(opendma_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("OutterLink Project");
MODULE_DESCRIPTION("OpenDMA: Non-proprietary GPU VRAM RDMA via PCIe BAR1");
MODULE_VERSION(OPENDMA_VERSION);
```

**Kbuild file:**

```makefile
# opendma/Kbuild
obj-m := opendma.o

# Need MLNX_OFED headers for peer_mem.h
EXTRA_CFLAGS += -I/usr/src/ofa_kernel/default/include
```

**Build:**

```bash
make -C /lib/modules/$(uname -r)/build M=$(pwd)/opendma modules
sudo insmod opendma/opendma.ko
```

---

#### Option B: P2PDMA Framework (Upstream Kernel, No MLNX_OFED Dependency)

This approach uses the kernel's built-in P2PDMA framework instead of MLNX_OFED's peer memory client API. It is more portable but requires the mlx5 driver to support P2PDMA consumption (which the inbox mlx5 may not fully support for arbitrary BARs).

```c
// SPDX-License-Identifier: GPL-2.0
/*
 * OpenDMA P2PDMA Provider - registers GPU BAR1 as P2P DMA resource
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>

#define NVIDIA_VENDOR_ID 0x10DE

static struct pci_dev *gpu_pdev;

static int opendma_p2p_init(void)
{
    struct pci_dev *pdev = NULL;
    resource_size_t bar1_size;
    int ret;

    /* Find first NVIDIA GPU */
    pdev = pci_get_device(NVIDIA_VENDOR_ID, PCI_ANY_ID, NULL);
    if (!pdev) {
        pr_err("opendma-p2p: No NVIDIA GPU found\n");
        return -ENODEV;
    }

    bar1_size = pci_resource_len(pdev, 1);
    if (bar1_size < (1ULL << 30)) {
        pr_err("opendma-p2p: BAR1 too small (%llu MB). Enable ReBAR.\n",
               (unsigned long long)bar1_size >> 20);
        pci_dev_put(pdev);
        return -EINVAL;
    }

    /* Enable PCI device if not already */
    ret = pci_enable_device(pdev);
    if (ret) {
        pr_err("opendma-p2p: Failed to enable PCI device\n");
        pci_dev_put(pdev);
        return ret;
    }

    pci_set_master(pdev);

    /* Register BAR1 as P2P DMA resource.
     * This creates struct pages for the BAR1 region with
     * pgmap->type = MEMORY_DEVICE_PCI_P2PDMA.
     * Other PCIe devices (ConnectX-5) can then DMA to these pages.
     */
    ret = pci_p2pdma_add_resource(pdev, 1 /* BAR1 */,
                                  bar1_size, 0 /* offset */);
    if (ret) {
        pr_err("opendma-p2p: Failed to register BAR1 as P2P resource: %d\n",
               ret);
        pci_dev_put(pdev);
        return ret;
    }

    /* Publish so orchestrator drivers can discover this P2P memory */
    pci_p2pmem_publish(pdev, true);

    gpu_pdev = pdev;
    pr_info("opendma-p2p: Registered GPU %s BAR1 (%llu MB) as P2P resource\n",
            pci_name(pdev), (unsigned long long)bar1_size >> 20);

    return 0;
}

static void opendma_p2p_exit(void)
{
    if (gpu_pdev) {
        /* P2PDMA resources are cleaned up automatically via devres */
        pci_dev_put(gpu_pdev);
    }
    pr_info("opendma-p2p: Unloaded\n");
}

module_init(opendma_p2p_init);
module_exit(opendma_p2p_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("OpenDMA: P2PDMA provider for GPU BAR1");
```

**Limitation:** The P2PDMA framework requires the RDMA driver (mlx5) to be a P2PDMA *client*. As of kernel 6.8, mlx5 does not natively consume P2PDMA pages from arbitrary providers. This would require either:
1. Patches to mlx5 to support P2PDMA page types in memory registration
2. An intermediary that allocates P2PDMA pages and presents them to mlx5 as regular DMA-mapped memory

This makes Option B a longer-term path than Option A.

---

#### Option C: Direct ioremap + Manual DMA Mapping

The most manual approach. We bypass both peer memory and P2PDMA frameworks and directly construct DMA mappings that point the ConnectX-5 at GPU BAR1 addresses.

```c
/*
 * Option C: Direct approach
 *
 * 1. Get BAR1 physical address from PCI config
 * 2. Use libibverbs from userspace to create a memory region (MR)
 *    with the physical BAR1 address as the DMA address
 * 3. Post RDMA work requests using this MR
 *
 * This requires a custom userspace RDMA flow that bypasses
 * ibv_reg_mr (which expects virtual memory) and instead uses
 * a kernel module to create the MR with explicit DMA addresses.
 *
 * Implementation: The kernel module creates an MR via
 * ib_alloc_mr() + ib_map_mr_sg() with a manually constructed
 * scatterlist pointing to BAR1 physical addresses.
 *
 * This is the fallback if MLNX_OFED peer memory API is unavailable
 * and P2PDMA is not supported.
 */
```

This approach is the most fragile and requires deep knowledge of mlx5 internals. Use only if Options A and B both fail.

---

### Step 6: Address Translation -- CUDA Pointer to BAR1 Offset

A critical challenge: CUDA returns device pointers (virtual addresses in GPU's unified VA space), but OpenDMA needs physical VRAM offsets to map to BAR1.

**Solution: cuMemGetAddressRange + cuPointerGetAttribute**

```c
// Userspace (in OutterLink server)

// Given a CUDA device pointer, find its VRAM offset:
CUdeviceptr dptr = /* from cuMemAlloc */;
size_t size;
CUdeviceptr base;

// Get the allocation base and size
cuMemGetAddressRange(&base, &size, dptr);

// For static BAR1 mapping with tinygrad patches,
// the VRAM offset = device_pointer value itself
// (tinygrad patches create identity mapping: VA = PA within VRAM)
//
// BUT: This assumption must be verified on our hardware.
// If CUDA uses non-identity VA mapping, we need:
//   1. cuMemGetAllocationGranularity to understand alignment
//   2. A custom CUDA kernel that reads threadIdx-based VRAM addresses
//      and reports them back, establishing the VA->PA mapping
//
// Safest approach: Pre-allocate VRAM regions via cuMemAlloc,
// then query their physical backing via NVIDIA's internal APIs
// or by writing known patterns and scanning BAR1.

// Register with OpenDMA kernel module:
struct opendma_register_region reg = {
    .gpu_index = 0,
    .vram_offset = (uint64_t)base,  // Assumes identity mapping
    .size = size,
};
ioctl(opendma_fd, OPENDMA_IOC_REGISTER, &reg);
```

**Alternative approach for address translation:** Write a small CUDA kernel that stores its own global memory physical address (accessible via PTX `%globaltimer` or similar intrinsics, or by writing a known pattern to a known VA and reading it back through BAR1 ioremap to establish the mapping).

The safest approach for the initial implementation:

1. Allocate a VRAM region via `cuMemAlloc`
2. Write a known test pattern via CUDA kernel
3. Read BAR1 sequentially (via small ioremap windows) searching for the pattern
4. The BAR1 offset where the pattern is found = the VRAM offset for that allocation
5. Cache this mapping for future use

This is slow but correct, and only needs to happen once per allocation.

---

## Plan B: nouveau Upstream RDMA Patches

### Current Status (as of March 2026)

NVIDIA engineer Yonatan Maman's patch series "GPU Direct RDMA (P2P DMA) for Device Private Pages":
- **v1:** October 2024 (4 patches)
- **RFC:** December 2024 (5 patches)
- **v2:** July 2025 (5 patches, addressed review feedback from Christoph Hellwig, Jason Gunthorpe, Leon Romanovsky)
- **Merge status:** NOT YET MERGED into mainline kernel

### What the Patches Do

Five patches touching mm/hmm, nouveau/dmem, IB/core, and RDMA/mlx5:

1. **mm/hmm:** Adds P2P page operations to `struct pagemap_ops`, allowing GPU drivers to expose device-private pages for P2P DMA
2. **nouveau/dmem:** Introduces `struct nouveau_dmem_hmm_p2p` with `p2p_start_addr` (BAR1 virtual address from `pci_alloc_p2pmem()`) and `p2p_size`. During `nouveau_dmem_init`, BAR1 accessibility is verified and struct pages (`PCI_P2P_PAGE`) are assigned for all BAR1 pages
3. **IB/core:** P2P DMA infrastructure for device private pages
4. **RDMA/mlx5:** Enables P2P DMA with fallback -- when P2P DMA mapping fails (inaccessible bridges), falls back to standard host-memory DMA
5. **RDMA/mlx5:** Enables ATS (Address Translation Service) for ODP (On-Demand Paging) memory

### The Fundamental Problem: nouveau Has No CUDA

nouveau is the open-source NVIDIA driver. It can manage GPU memory and set up page tables, but it has no CUDA compute capability. CUDA requires NVIDIA's proprietary driver.

**Can nouveau and nvidia.ko coexist?**

No. Both drivers bind to the same PCI device. Only one can be loaded at a time. You cannot use nouveau for memory management and nvidia.ko for CUDA compute simultaneously.

**Potential workaround:** Use nouveau on a *different* GPU solely for RDMA testing/validation, while the primary GPU runs nvidia.ko for CUDA. This is useful for development but not for production.

### When to Pursue Plan B

- If tinygrad patches fail on RTX 3090 Ti (Plan A fails)
- If the nouveau patches get merged into mainline (reduces maintenance burden)
- If a way is found to split GPU functionality between drivers (unlikely)
- As a reference implementation to understand the correct kernel integration patterns

### How to Test

```bash
# 1. Apply Yonatan Maman's v2 patches to kernel source
cd /usr/src/linux-$(uname -r)
git am /path/to/nouveau-rdma-v2/*.patch

# 2. Build kernel with nouveau + mlx5 + P2PDMA support
make menuconfig
# Enable: CONFIG_DRM_NOUVEAU, CONFIG_PCI_P2PDMA, CONFIG_MLX5_CORE,
#         CONFIG_MLX5_INFINIBAND, CONFIG_ZONE_DEVICE

# 3. Boot with nouveau (unload nvidia.ko first)
# WARNING: No CUDA available with nouveau
sudo modprobe -r nvidia
sudo modprobe nouveau

# 4. Verify RDMA path
# Use perftest tools to test RDMA to GPU memory via nouveau
```

---

## Plan C: Linux P2PDMA Framework (Custom Provider)

### How It Works

The P2PDMA framework (kernel 4.20+) is designed for exactly this use case: inter-device DMA over PCIe.

1. A **provider** module registers a BAR as P2P-capable memory
2. A **client** driver uses these pages for DMA transfers
3. The kernel checks PCIe topology to ensure the devices can communicate

### What We Build

A kernel module that:
1. Claims the GPU's BAR1 as a P2PDMA resource
2. Publishes it for discovery
3. The mlx5 driver (or a wrapper) allocates P2PDMA pages and uses them for RDMA

### Topology Requirement

P2PDMA performs a compatibility check: provider and client must be behind the same PCIe root port, or the root complex must be whitelisted. This is checked via `pci_p2pdma_distance_many()`.

**Critical:** Run `lspci -tv` to verify GPU and ConnectX-5 share a root port. If they are on different root complexes, P2PDMA will refuse the transfer. Override with kernel parameter `pci=disable_acs_redir=*` and potentially patching the P2PDMA whitelist.

### Integration with mlx5

The challenge: mlx5's memory registration path (`mlx5_ib_reg_user_mr`) expects either:
- User virtual memory (standard path)
- Peer memory pages (via peer_memory_client, Option A)
- ODP (On-Demand Paging) with device-private pages (nouveau patches path)

mlx5 does NOT natively consume arbitrary P2PDMA pages for RDMA memory regions.

**Bridge solution:** Write a kernel module that:
1. Registers GPU BAR1 as P2PDMA provider
2. Allocates P2PDMA pages from the provider
3. Maps these pages into a scatterlist
4. Creates an mlx5 MR using `mlx5_ib_reg_dm_mr()` or by directly constructing the MKC (Memory Key Context) in mlx5's firmware

This requires deep mlx5 internals knowledge and is the most complex option.

---

## Validation and Testing

### Test 1: Verify BAR1 Maps All VRAM

```bash
# After loading patched driver with ReBAR enabled
nvidia-smi -q | grep -A3 "BAR1 Memory"
# Expected:
#   BAR1 Memory Usage
#       Total                             : 24576 MiB
#       Used                              : <some value> MiB
#       Free                              : <rest> MiB

# Check PCI BAR1 directly
sudo lspci -vvv -s $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1) \
    | grep -A1 "Region 1"
# Expected: Region 1: Memory at <hex> (64-bit, prefetchable) [size=24G]
```

### Test 2: Write via CUDA, Read via BAR1 ioremap

This test verifies that tinygrad's BAR1 mapping is working -- data written by CUDA is readable through BAR1.

```c
// test_bar1_readback.c - Kernel module test
// Loads as a kernel module, ioremaps a small window of BAR1,
// reads data written by a CUDA program

// CUDA side (userspace):
// 1. cuMemAlloc(&dptr, 4096)
// 2. Launch kernel that writes 0xDEADBEEF to first 4 bytes
// 3. Print dptr value (this is the VRAM address)
// 4. Trigger ioctl to kernel module with dptr value

// Kernel module side:
// 1. ioremap(BAR1_base + vram_offset, 4096)
// 2. value = ioread32(mapped_addr)
// 3. if (value == 0xDEADBEEF) -> SUCCESS
// 4. iounmap()
```

**Test program (userspace CUDA):**

```c
// test_bar1_cuda.cu
#include <cuda.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

__global__ void write_pattern(int *ptr) {
    ptr[0] = 0xDEADBEEF;
    ptr[1] = 0xCAFEBABE;
    ptr[2] = 0x12345678;
    ptr[3] = 0xABCD0123;
}

int main() {
    int *d_ptr;
    cudaMalloc(&d_ptr, 4096);
    write_pattern<<<1, 1>>>(d_ptr);
    cudaDeviceSynchronize();

    printf("CUDA device pointer: %p\n", d_ptr);
    printf("Wrote pattern to GPU VRAM. Now read via BAR1...\n");

    // Tell kernel module to read BAR1 at this offset
    int fd = open("/dev/opendma", O_RDWR);
    struct { uint64_t offset; uint64_t size; } req = {
        .offset = (uint64_t)d_ptr,
        .size = 4096
    };
    ioctl(fd, /* OPENDMA_IOC_VERIFY */ 0x4F05, &req);
    close(fd);

    cudaFree(d_ptr);
    return 0;
}
```

### Test 3: RDMA Write to GPU BAR1, Read Back via CUDA

This is the reverse direction test and the core OpenDMA validation.

```
Setup:
  PC1 (sender): ConnectX-5, has data in host memory
  PC2 (receiver): ConnectX-5 + GPU with OpenDMA module loaded

Steps:
  1. PC2: cuMemAlloc 1MB region on GPU
  2. PC2: Register the VRAM region with opendma.ko -> get rkey
  3. PC2: Share rkey + BAR1 physical address with PC1 (via TCP control channel)
  4. PC1: ibv_post_send RDMA_WRITE to PC2's BAR1 address using rkey
  5. PC2: cudaMemcpy DtoH the 1MB region -> verify data matches what PC1 sent
```

### Test 4: Bandwidth Benchmark

```bash
# Using perftest with custom memory registration
# PC2 (server, GPU side):
ib_write_bw --use_opendma=0 -d mlx5_0 -s 1048576 -n 1000

# PC1 (client):
ib_write_bw -d mlx5_0 -s 1048576 -n 1000 <PC2_IP>

# Expected: ~10-12 GB/s on 100GbE, limited by network
# Compare with host-staged baseline:
# ib_write_bw -d mlx5_0 -s 1048576 -n 1000  (host memory, no GPU)
```

Note: perftest does not natively support OpenDMA. We will need to write a custom benchmark tool that uses our kernel module's registered MR for the RDMA verbs, or modify perftest to accept pre-registered DMA addresses.

### Test 5: Latency Benchmark

```bash
# Similar to bandwidth but with small messages
# PC2: ib_write_lat --use_opendma=0 -d mlx5_0 -s 64
# PC1: ib_write_lat -d mlx5_0 -s 64 <PC2_IP>
# Expected: <5us for 64B RDMA write to GPU VRAM
```

### Test 6: Stress Test

```
1. Sustained transfer: Write 24GB (full VRAM) continuously for 1 hour
2. Error injection: Unplug network cable during transfer, verify graceful recovery
3. Concurrent CUDA + RDMA: Run CUDA kernels while RDMA writes are in progress
   - Verify no data corruption
   - Verify no GPU hangs
   - Measure performance impact of concurrent access
4. Multi-region: Register 100 non-contiguous VRAM regions, RDMA to all simultaneously
5. Hot path: Rapid register/unregister cycles (1000 per second)
```

---

## Integration with OutterLink

### Transport Trait Implementation

```rust
// crates/outterlink-common/src/transport.rs

/// Transport capabilities detected at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportMode {
    /// Regular TCP with host-staged GPU memory
    TcpHostStaged,
    /// TCP with io_uring zero-copy + host-staged
    TcpZeroCopy,
    /// RDMA with host-staged GPU memory
    RdmaHostStaged,
    /// OpenDMA: direct NIC-to-GPU VRAM RDMA
    OpenDma,
}

/// The Transport trait with OpenDMA-aware methods
pub trait Transport: Send + Sync {
    /// Detect the best available transport mode
    fn detect_mode(&self) -> TransportMode;

    /// Standard send/recv (host memory)
    async fn send(&self, buf: &[u8]) -> Result<usize>;
    async fn recv(&self, buf: &mut [u8]) -> Result<usize>;

    /// GPU-aware send/recv
    /// With OpenDMA: zero-copy DMA between NIC and GPU VRAM
    /// Without OpenDMA: cudaMemcpy to host staging buffer, then network send
    async fn send_gpu(&self, gpu_ptr: u64, size: usize, device: i32) -> Result<()>;
    async fn recv_gpu(&self, gpu_ptr: u64, size: usize, device: i32) -> Result<()>;
}
```

### OpenDMA Transport Backend

```rust
// crates/outterlink-common/src/transport/opendma.rs

use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

/// OpenDMA transport backend
pub struct OpenDmaTransport {
    /// File descriptor to /dev/opendma
    opendma_fd: i32,
    /// RDMA resources (QP, CQ, PD)
    rdma: RdmaResources,
    /// Registered GPU VRAM regions
    regions: Vec<RegisteredRegion>,
}

struct RegisteredRegion {
    gpu_index: u32,
    vram_offset: u64,
    size: u64,
    handle: u64,
    /// RDMA memory region key for remote access
    rkey: u32,
    lkey: u32,
}

impl OpenDmaTransport {
    /// Attempt to initialize OpenDMA.
    /// Returns None if hardware requirements not met.
    pub fn try_new() -> Option<Self> {
        // 1. Check if /dev/opendma exists (module loaded)
        let fd = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/opendma")
            .ok()?;

        // 2. Query GPU count
        let mut gpu_count: u32 = 0;
        unsafe {
            if libc::ioctl(fd.as_raw_fd(), OPENDMA_IOC_GPU_COUNT, &mut gpu_count) != 0 {
                return None;
            }
        }

        if gpu_count == 0 {
            return None;
        }

        // 3. Initialize RDMA resources (QP, CQ, PD via libibverbs)
        let rdma = RdmaResources::new()?;

        Some(Self {
            opendma_fd: fd.as_raw_fd(),
            rdma,
            regions: Vec::new(),
        })
    }

    /// Register a GPU VRAM region for RDMA access
    pub fn register_vram(&mut self, gpu_index: u32, offset: u64, size: u64)
        -> Result<&RegisteredRegion>
    {
        let mut reg = OpendmaRegisterRegion {
            gpu_index,
            vram_offset: offset,
            size,
            handle: 0,
        };

        unsafe {
            if libc::ioctl(self.opendma_fd, OPENDMA_IOC_REGISTER, &mut reg) != 0 {
                return Err(/* ioctl error */);
            }
        }

        // Create RDMA MR using the registered region
        // The peer memory client in opendma.ko handles the DMA mapping
        let mr = self.rdma.reg_mr_peer(offset, size)?;

        self.regions.push(RegisteredRegion {
            gpu_index,
            vram_offset: offset,
            size,
            handle: reg.handle,
            rkey: mr.rkey(),
            lkey: mr.lkey(),
        });

        Ok(self.regions.last().unwrap())
    }
}

impl Transport for OpenDmaTransport {
    fn detect_mode(&self) -> TransportMode {
        TransportMode::OpenDma
    }

    async fn send_gpu(&self, gpu_ptr: u64, size: usize, device: i32) -> Result<()> {
        // Find the registered region containing this pointer
        let region = self.find_region(gpu_ptr, size)?;

        // Calculate BAR1 offset
        let offset_in_region = gpu_ptr - region.vram_offset;

        // Post RDMA WRITE using region's lkey
        // Source: local GPU BAR1 (via lkey)
        // Destination: remote GPU BAR1 (via rkey, exchanged during connection setup)
        self.rdma.post_write(
            region.lkey,
            region.vram_offset + offset_in_region,
            size,
            self.remote_rkey,
            self.remote_addr + offset_in_region,
        ).await
    }

    async fn recv_gpu(&self, gpu_ptr: u64, size: usize, device: i32) -> Result<()> {
        // For RDMA WRITE model: receiver does not need to post recv.
        // The sender writes directly to the receiver's GPU VRAM.
        // This method waits for completion notification.
        self.rdma.poll_cq().await
    }

    // ... send/recv for host memory delegates to base RDMA transport
}
```

### Runtime Detection and Fallback

```rust
// crates/outterlink-server/src/transport_factory.rs

/// Create the best available transport for this system
pub fn create_transport(config: &TransportConfig) -> Box<dyn Transport> {
    // Try OpenDMA first (if enabled in config)
    if config.enable_opendma {
        if let Some(opendma) = OpenDmaTransport::try_new() {
            log::info!("OpenDMA available: direct NIC-to-GPU VRAM RDMA enabled");
            return Box::new(opendma);
        }
        log::warn!("OpenDMA requested but not available, falling back");
    }

    // Try RDMA host-staged
    if let Some(rdma) = RdmaHostStagedTransport::try_new() {
        log::info!("Using RDMA with host-staged GPU memory");
        return Box::new(rdma);
    }

    // Fall back to TCP
    log::info!("Using TCP transport");
    Box::new(TcpTransport::new(config))
}
```

### Configuration

```toml
# outterlink.toml

[transport]
# Transport mode selection:
# "auto"  - detect best available (OpenDMA > RDMA > TCP)
# "opendma" - require OpenDMA, fail if unavailable
# "rdma"  - RDMA with host-staged, no OpenDMA
# "tcp"   - TCP only
mode = "auto"

[opendma]
# Enable/disable OpenDMA even if hardware supports it
enabled = true
# Path to OpenDMA device
device_path = "/dev/opendma"

# Per-GPU OpenDMA configuration
# GPUs not listed here use host-staged even if OpenDMA is available
[[opendma.gpus]]
index = 0
enabled = true
# Pre-register VRAM regions at startup (MB)
# Larger regions = fewer registrations but more upfront cost
preallocate_mb = 4096

[[opendma.gpus]]
index = 1
enabled = true
preallocate_mb = 4096
```

### Memory Region Registration Lifecycle

```
Application calls cuMemAlloc(size)
    |
    v
OutterLink server intercepts, allocates on real GPU
    |
    v
Server checks: is this GPU OpenDMA-enabled?
    |
    +-- NO: Track allocation in host-staged path (standard)
    |
    +-- YES:
        |
        v
    Determine VRAM offset for this allocation
    (via CUDA pointer -> BAR1 offset translation)
        |
        v
    ioctl(opendma_fd, REGISTER, {gpu, offset, size})
        |
        v
    opendma.ko creates peer memory entry
    mlx5 creates MR with BAR1-backed DMA addresses
        |
        v
    Store {gpu_ptr -> (rkey, lkey, remote_addr)} mapping
        |
        v
    When cuMemcpyHtoD/DtoH is intercepted:
        Use RDMA WRITE/READ with stored rkey/lkey
        Data flows: NIC DMA <-> GPU BAR1 <-> VRAM (zero-copy)

Application calls cuMemFree(ptr)
    |
    v
    ioctl(opendma_fd, UNREGISTER, handle)
    Destroy MR, remove mapping
```

---

## Hardware Requirements and BIOS Settings

### Mandatory Settings

| Setting | Location | Value | Why |
|---------|----------|-------|-----|
| Above 4G Decoding | BIOS > Advanced > PCI | **Enabled** | Required for large BAR allocation (24GB needs addresses above 4GB) |
| Resizable BAR (ReBAR) | BIOS > Advanced > PCI | **Enabled** | Exposes full 24GB VRAM through BAR1 (vs 256MB without) |
| CSM (Compatibility Support Module) | BIOS > Boot | **Disabled** | CSM blocks ReBAR; must use pure UEFI boot |
| IOMMU (VT-d / AMD-Vi) | BIOS > Advanced > CPU | **Disabled** | P2P DMA fails with IOMMU enabled (TLPs get remapped) |

### Kernel Boot Parameters

```bash
# /etc/default/grub
# GRUB_CMDLINE_LINUX="intel_iommu=off pci=disable_acs_redir=*"
#
# OR for AMD:
# GRUB_CMDLINE_LINUX="amd_iommu=off pci=disable_acs_redir=*"

# intel_iommu=off / amd_iommu=off:
#   Completely disables IOMMU. Required for P2P DMA.
#   WARNING: Disables VT-d. Do NOT use if running VMs with PCIe passthrough.
#
# iommu=pt (alternative, less aggressive):
#   Passthrough mode. May work if the IOMMU is configured to not remap
#   P2P TLPs, but not guaranteed. Test with iommu=off first.
#
# pci=disable_acs_redir=*:
#   Disables PCIe Access Control Services redirect for all devices.
#   ACS forces P2P TLPs through the root complex even when devices
#   are behind the same switch. Disabling allows direct peer traffic.
#   The wildcard (*) disables for all devices.
#   More targeted: pci=disable_acs_redir=<GPU_BDF>;<NIC_BDF>
```

### PCIe Topology Verification

```bash
# Check that GPU and ConnectX-5 share a root complex
lspci -tv

# Ideal topology:
# -[0000:00]-+-00.0  Intel Root Complex
#            +-01.0-[01]--+-00.0  NVIDIA RTX 3090 Ti
#            |             \-00.1  NVIDIA RTX 3090 Ti (audio)
#            +-02.0-[02]----00.0  Mellanox ConnectX-5
#            ...
#
# Both under [0000:00] root complex = GOOD
# If on different root complexes, P2P traffic goes through CPU (slower but works)
# If behind different root PORTS of same complex, need ACS disabled

# Detailed P2P distance check (once opendma.ko has P2PDMA support):
# The kernel logs the P2P compatibility during pci_p2pdma_add_resource
dmesg | grep p2pdma
```

### Hardware Checklist

| Item | Check Command | Expected Result |
|------|--------------|-----------------|
| GPU visible | `nvidia-smi` | RTX 3090 Ti listed |
| CUDA works | `./deviceQuery` (CUDA sample) | PASS |
| BAR1 = full VRAM | `nvidia-smi -q \| grep -A2 BAR1` | Total: 24576 MiB |
| BAR1 physical address | `sudo lspci -vvv -s <BDF> \| grep "Region 1"` | Non-zero, size=24G |
| ConnectX-5 visible | `ibv_devices` | mlx5_0 listed |
| RDMA functional | `rping -s` / `rping -c` | Connection succeeds |
| IOMMU disabled | `dmesg \| grep -i iommu` | No IOMMU init messages |
| ACS disabled/overridden | `sudo lspci -vvv -s <BDF> \| grep ACS` | ACSCtl: disabled or not present |
| Same root complex | `lspci -tv` | GPU and NIC under same root |
| Patched driver loaded | `dmesg \| grep -i "static bar1"` | BAR1 mapping success message |

---

## Risks Specific to OpenDMA

### Risk 1: GSP Firmware Blocking (HIGH)

**What:** The GPU System Processor (GSP) runs signed, proprietary firmware. Even with tinygrad patches in the open kernel module, the GSP firmware may independently validate BAR1 access patterns and reject unauthorized P2P transactions.

**Evidence for:** GSP firmware on Turing+ GPUs handles most GPU management. The nvidia_p2p_get_pages() restriction may be enforced in firmware, not just in the kernel module.

**Evidence against:** The tinygrad patches successfully enable GPU-to-GPU P2P on RTX 4090, meaning the GSP does NOT independently block BAR1 writes from other PCIe devices. If it blocked GPU-to-GPU P2P, it would also block NIC-to-GPU.

**Mitigation:** Test on real hardware early. If GSP blocks: fall back to Plan B (nouveau patches) or Plan E (host-staged). The architectural design supports graceful fallback.

**Assessment:** MEDIUM probability of blocking. The tinygrad P2P success on 4090 suggests the GSP does not independently police BAR1 access. The patches program the GMMU correctly, and the GSP respects the GMMU configuration even when initiated by a patched driver.

### Risk 2: tinygrad Patches Not Working on RTX 3090 Ti (MEDIUM)

**What:** The tinygrad patches were developed and tested primarily on RTX 4090 (Ada Lovelace, AD102). RTX 3090 Ti is Ampere (GA102), a different architecture generation. Some HAL methods may behave differently.

**Evidence for concern:** Issues have been reported with mixed-generation setups (3090 + 4090 + 5090). The error "BAR1 size is not large enough to map FB size" has been seen on some GPUs when the driver attempts to force static BAR1 mapping.

**Evidence against:** The forkProj/open-gpu-kernel-modules-P2P fork specifically targets broader GPU compatibility. The GMMU architecture is largely consistent across Ampere and Ada Lovelace.

**Mitigation:**
1. Try the tinygrad patches first (simplest)
2. If they fail, try the forkProj fork (broader compatibility)
3. If both fail, try the aikitoria fork (tested on 5090, may support 3090 Ti)
4. If all fail on 3090 Ti: test on RTX 4090 if available, or pivot to nouveau (Plan B)

### Risk 3: Stability Under Sustained Transfers (MEDIUM)

**What:** BAR1 is normally used for display framebuffer and small driver-internal accesses. Sustained high-bandwidth RDMA writes to BAR1 may expose bugs in the GPU memory controller or cause thermal issues.

**Mitigation:**
1. Start with short transfer tests, increase duration gradually
2. Monitor GPU temperature during sustained transfers: `nvidia-smi -l 1`
3. Monitor for GPU errors: `nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total --format=csv`
4. If stability issues arise: implement rate limiting or periodic pauses

### Risk 4: Interaction with CUDA Memory Management (MEDIUM)

**What:** CUDA's memory allocator (cuMemAlloc) uses its own address space management. RDMA writes to BAR1 bypass CUDA's knowledge -- if CUDA relocates a VRAM page (unlikely with pinned allocations, but possible during driver operations), the RDMA write could hit the wrong location.

**Mitigation:**
1. Use `cuMemAlloc` with `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED` for stable allocations
2. Register regions with opendma.ko AFTER cuMemAlloc returns (not during)
3. Unregister BEFORE cuMemFree (maintain strict ordering)
4. Consider using `cuMemAddressReserve` / `cuMemMap` for explicit physical backing control (CUDA VMM API)
5. Use invalidation callbacks (in nvidia_p2p_get_pages pattern) to detect page migration -- though we bypass nvidia_p2p, we can monitor CUDA events

### Risk 5: MLNX_OFED Peer Memory API Deprecation (LOW)

**What:** NVIDIA (Mellanox) is moving toward DMA-BUF instead of peer memory for GPU RDMA. The `ib_register_peer_memory_client` API may be removed from future MLNX_OFED releases.

**Mitigation:**
1. Pin MLNX_OFED version that supports peer memory (current LTS)
2. Prepare P2PDMA fallback (Option B in module design)
3. Monitor DMA-BUF evolution -- if DMA-BUF becomes viable, migrate to it

### Risk 6: PCIe Topology Incompatible (LOW)

**What:** If GPU and ConnectX-5 are on different PCIe root ports with ACS enabled, P2P TLPs are routed through the CPU's root complex. This adds latency and may reduce bandwidth.

**Mitigation:**
1. Verify topology with `lspci -tv` BEFORE investing in OpenDMA development
2. If different root ports: disable ACS (`pci=disable_acs_redir=*`)
3. If different root complexes entirely: P2P still works but through CPU -- measure actual performance impact
4. If performance is unacceptable: physically move NIC to a PCIe slot closer to the GPU

---

## Legal Considerations

### nvidia-open Kernel Modules License

The NVIDIA open kernel modules are **dual-licensed MIT/GPL**. Modifying them (applying tinygrad patches) is explicitly permitted by both licenses.

- MIT license: Allows modification, distribution, commercial use with minimal restrictions
- GPL: Standard kernel module license, requires derived works to also be GPL

**We do NOT distribute patched NVIDIA drivers.** OutterLink provides:
1. Instructions for users to apply patches themselves
2. A script that automates the patching process
3. Our `opendma.ko` module (GPL, our own code)

### opendma.ko License

Our kernel module MUST be GPL (or GPL-compatible) because:
1. It uses kernel APIs (`pci_resource_start`, `ioremap`, etc.)
2. It links against MLNX_OFED's peer memory API
3. Kernel modules using GPL-only symbols must be GPL

License header: `MODULE_LICENSE("GPL");`

### NVIDIA EULA

The GeForce EULA states: "GeForce SOFTWARE is not licensed for datacenter deployment."

**Our position:**
1. OutterLink is a development/research tool
2. Users can use it in any configuration they choose
3. We do not distribute patched NVIDIA software
4. The EULA applies to NVIDIA's software, not to our independent module
5. OpenDMA bypasses NVIDIA's software entirely -- it uses standard PCIe mechanisms

### Precedent

- tinygrad's P2P patches have been public on GitHub since 2024 with no legal action from NVIDIA
- AMD's ROCnRDMA (fully open source GPU RDMA for consumer GPUs) has been public for years
- The Linux kernel's P2PDMA framework is designed for exactly this use case

### Risk Assessment

| Action | Legal Risk |
|--------|-----------|
| Developing opendma.ko (GPL module using PCIe APIs) | **None** -- standard kernel development |
| Providing instructions to patch nvidia-open | **Minimal** -- informational, user applies patches |
| Distributing patched nvidia.ko binaries | **Medium** -- would not do this |
| Using OpenDMA in commercial datacenter | **Medium** -- EULA question, but EULA applies to NVIDIA's software, not ours |
| Publishing OpenDMA research/benchmarks | **None** -- academic freedom |

---

## Implementation Phases

### Phase 9.1: Environment Verification (1-2 days)

**Files:** None (hardware/BIOS work)

**Steps:**
1. Enable ReBAR in BIOS on both PCs
2. Disable IOMMU in BIOS (or kernel boot param)
3. Disable ACS (kernel boot param)
4. Verify BAR1 size shows full VRAM (`nvidia-smi -q`)
5. Verify PCIe topology (`lspci -tv`)
6. Run RDMA baseline benchmark (`ib_write_bw` host-to-host)

**Acceptance criteria:**
- [ ] BAR1 shows 24576 MiB (24GB)
- [ ] IOMMU disabled confirmed
- [ ] GPU and NIC share root complex (or ACS disabled)
- [ ] Host-to-host RDMA works at >10 GB/s

### Phase 9.2: Patched Driver (2-3 days)

**Files:** Build scripts in `opendma/scripts/`

**Steps:**
1. Clone tinygrad patches for matching driver version
2. Build patched nvidia-open kernel modules
3. Load patched modules, verify GPU still works
4. Verify BAR1 mapping: write via CUDA, read small window via ioremap
5. Test if nvidia-peermem works on GeForce with patched driver (quick test)

**Acceptance criteria:**
- [ ] Patched driver loads without errors
- [ ] nvidia-smi shows all GPUs
- [ ] CUDA deviceQuery passes
- [ ] BAR1 data matches CUDA-written data (verified via ioremap test)

### Phase 9.3: Kernel Module Development (1-2 weeks)

**Files to create:**
- `opendma/Kbuild`
- `opendma/opendma.c` (main module, Option A peer memory client)
- `opendma/opendma_p2p.c` (Option B P2PDMA provider, fallback)
- `opendma/opendma_ioctl.h` (ioctl definitions, shared with userspace)
- `opendma/opendma_internal.h` (internal structures)
- `opendma/Makefile`

**Steps:**
1. Implement GPU discovery (PCI enumeration, BAR1 address extraction)
2. Implement peer memory client callbacks
3. Implement ioctl interface (char device)
4. Build and load module
5. Verify: module loads, discovers GPU, registers with IB core
6. Verify: `cat /sys/kernel/mm/memory_peers/opendma/`

**Acceptance criteria:**
- [ ] `opendma.ko` loads without errors
- [ ] `dmesg` shows GPU discovery and BAR1 info
- [ ] `/sys/kernel/mm/memory_peers/opendma/` exists
- [ ] ioctl returns GPU info correctly

### Phase 9.4: RDMA Verification (1 week)

**Files to create:**
- `opendma/tests/test_bar1_readback.c` (kernel test module)
- `opendma/tests/test_bar1_cuda.cu` (CUDA test program)
- `opendma/tests/test_rdma_write.c` (RDMA write to BAR1 test)
- `opendma/tests/test_rdma_roundtrip.c` (full roundtrip test)
- `opendma/tests/bench_bandwidth.c` (bandwidth benchmark)
- `opendma/tests/bench_latency.c` (latency benchmark)
- `opendma/tests/Makefile`

**Steps:**
1. Test 2: CUDA write -> BAR1 read (data integrity)
2. Test 3: RDMA write -> CUDA read (reverse data integrity)
3. Test 4: Bandwidth benchmark
4. Test 5: Latency benchmark
5. Test 6: Stress test (1 hour sustained)

**Acceptance criteria:**
- [ ] Data integrity: zero bit errors over 1GB transfer
- [ ] Bandwidth: >8 GB/s on 100GbE
- [ ] Latency: <5us for 64B RDMA write to VRAM
- [ ] Stability: 1 hour sustained transfer without errors
- [ ] Concurrent CUDA+RDMA: no corruption or hangs

### Phase 9.5: Rust Integration (1 week)

**Files to modify:**
- `crates/outterlink-common/src/transport.rs` -- add OpenDma variant to TransportMode
- `crates/outterlink-common/src/transport/opendma.rs` -- new file, OpenDMA transport implementation
- `crates/outterlink-common/src/transport/mod.rs` -- add opendma module
- `crates/outterlink-server/src/transport_factory.rs` -- add OpenDMA detection and creation
- `crates/outterlink-server/src/config.rs` -- add OpenDMA configuration
- `opendma/opendma_ioctl.h` -- ioctl definitions (shared)

**Steps:**
1. Create Rust FFI bindings for opendma ioctl interface
2. Implement OpenDmaTransport struct
3. Implement Transport trait for OpenDmaTransport
4. Implement runtime detection (check /dev/opendma exists, query GPUs)
5. Implement fallback logic in transport_factory
6. Add configuration options to outterlink.toml
7. Integration test: end-to-end GPU memory transfer via OpenDMA

**Acceptance criteria:**
- [ ] OutterLink detects OpenDMA availability at startup
- [ ] `send_gpu` / `recv_gpu` use RDMA directly to GPU VRAM
- [ ] Falls back to host-staged when OpenDMA unavailable
- [ ] Configuration file controls per-GPU OpenDMA enable/disable

### Phase 9.6: Documentation and Polish (2-3 days)

**Files to create:**
- `docs/guides/opendma-setup.md` -- user guide for enabling OpenDMA
- `docs/specs/opendma-architecture.md` -- technical specification
- `docs/decisions/ADR-003-opendma-implementation.md` -- implementation decision record

**Steps:**
1. Write setup guide (BIOS settings, patching, module loading)
2. Write architecture document (data flow, module interfaces)
3. Write ADR documenting why Option A was chosen over B/C
4. Add OpenDMA benchmarks to project README

**Acceptance criteria:**
- [ ] A user with matching hardware can follow the guide and get OpenDMA working
- [ ] Architecture document covers all module interfaces
- [ ] Benchmarks show improvement over host-staged

---

## Estimated Scope

| Component | Files | Language | Lines (est.) |
|-----------|-------|----------|-------------|
| Kernel module (opendma.ko) | 5 | C | ~1500 |
| Test suite | 6 | C/CUDA | ~1000 |
| Rust transport backend | 3 | Rust | ~800 |
| Build scripts | 3 | Shell | ~200 |
| Documentation | 3 | Markdown | ~500 |
| **Total** | **20** | | **~4000** |

---

## Open Questions (to resolve during implementation)

- [ ] Does `nvidia_p2p_get_pages()` succeed on GeForce with tinygrad BAR1 patches?
- [ ] Exact driver version to target for tinygrad patches on RTX 3090 Ti?
- [ ] CUDA pointer -> VRAM offset translation: identity mapping or need explicit translation?
- [ ] PCIe topology on our specific hardware: same root complex?
- [ ] Does GSP firmware on GA102 interfere with BAR1 P2P from NIC?
- [ ] Performance impact of concurrent CUDA compute + BAR1 RDMA writes?
- [ ] Can we use 2MB pages in BAR1 mapping for better TLB efficiency?

---

## Related Documents

- [R5: GPUDirect on GeForce](../research/R5-gpudirect-geforce-restriction.md)
- [R7: Non-Proprietary GPU DMA](../research/R7-non-proprietary-gpu-dma.md)
- [R4: ConnectX-5 + Transport](../research/R4-connectx5-transport-stack.md)
- [Contingency Plans](../pre-planning/03-contingency-plans.md) -- Section 1: OpenDMA
- [ADR-002: OpenDMA Naming](../../docs/decisions/ADR-002-opendma-naming.md)
- [P2: Development Environment](P2-dev-environment.md) -- BIOS settings
- [P8: Performance Phase](P8-performance.md) -- host-staged baseline

## Key External Sources

1. [tinygrad/open-gpu-kernel-modules](https://github.com/tinygrad/open-gpu-kernel-modules) -- P2P patches
2. [aikitoria/open-gpu-kernel-modules](https://github.com/aikitoria/open-gpu-kernel-modules) -- Extended GPU support fork
3. [forkProj/open-gpu-kernel-modules-P2P](https://github.com/forkProj/open-gpu-kernel-modules-P2P) -- BAR1 P2P fork
4. [NVIDIA/open-gpu-kernel-modules -- nvidia-peermem.c](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/main/kernel-open/nvidia-peermem/nvidia-peermem.c) -- Reference peer memory client
5. [Mellanox/nv_peer_memory](https://github.com/Mellanox/nv_peer_memory) -- Legacy peer memory module
6. [sbates130272/io_peer_mem](https://github.com/sbates130272/io_peer_mem) -- Reference peer memory client for IO memory
7. [Linux P2PDMA documentation](https://docs.kernel.org/driver-api/pci/p2pdma.html)
8. [Nouveau RDMA patches v2 (Yonatan Maman)](https://www.mail-archive.com/nouveau@lists.freedesktop.org/msg47528.html)
9. [Phoronix: NVIDIA P2P DMA RDMA patches](https://www.phoronix.com/news/NVIDIA-Linux-P2P-DMA-RDMA-Priv)
10. [LWN: GPU Direct RDMA for Device Private Pages](https://lwn.net/Articles/1030499/)
11. [NVIDIA GPUDirect RDMA documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
12. [Linux PCI driver tutorial (Oleg Kutkov)](https://olegkutkov.me/2021/01/07/writing-a-pci-device-driver-for-linux/)
13. [USFCA VRAM driver example](https://www.cs.usfca.edu/~cruse/cs635f07/vram.c)
