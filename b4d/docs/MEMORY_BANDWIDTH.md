# Host memory (RAM) for the b4d writer

The writer moves every frame through host memory at least once: the frame grabber writes it, then the CPU compresses
it or the GPU uploads it. This guide gives the RAM traffic of each writer path and the number of memory modules (DIMMs)
needed to sustain it. The measurements were made on one AMD EPYC 9754 (128 cores, 12 DDR5-4800 memory channels) with
12 DIMMs, and two NVIDIA RTX A6000 GPUs (PCIe Gen4 x16).

## Bandwidth against DIMM count

Use one DIMM per channel and populate the channels in a balanced configuration (AMD EPYC 9754: 4, 6, 8, 10 or 12 DIMMs).
The bandwidth grows with the number of populated channels (38.4 GB/s per channel for DDR5-4800, theoretical). The
capacity per DIMM adds no bandwidth, and the PCIe bandwidth to the GPUs does not depend on the DIMMs.

**Table 1.** Host memory bandwidth on the AMD EPYC 9754 (64 threads, one per core, 64 MB buffers per thread).

| DIMMs (channels) | Theoretical (GB/s) | Read (GB/s) | Write (GB/s) | Copy, read + write (GB/s) |
|---|---|---|---|---|
| 4 | 154 | not measured | not measured | not measured |
| 8 | 307 | 272 | 194 | 229 |
| 12 | 461 | 358–360 | 243 | 331 |

The write rates of the README (Table 2 and Fig. 2) were measured with 8 DIMMs. With 12 DIMMs the CPU writer (Teledyne
Kinetix, 128 cores) sustained about the same rate (3,770 against 3,827 fps), and two GPUs in zero copy sustained 2,558
against 2,448 fps (within one × 1.10 ladder step): at these rates, 8 DIMMs are enough.

## RAM traffic of each writer path

The traffic was measured by displacement: a memory-copy load on 16 cores saturates RAM (about 365 GB/s on its own), and
the writer streams the Teledyne Kinetix (8 bit, 3,200 × 3,200) at a fixed rate for 60 s next to it. The bandwidth the
load loses while the writer streams is taken as the writer's traffic.

**Table 2.** RAM traffic of the writer (Teledyne Kinetix, 8 bit, zero dropped frames, in-memory sink).

| Path | Offered rate (fps) | Frames written (GB/s) | RAM traffic (GB/s) | RAM traffic per byte written (bytes) | Including the camera's write (bytes per byte written) |
|---|---|---|---|---|---|
| 1 GPU, zero copy | 800 | 8.2 | 10 | 1.2 | 2.2 |
| 2 GPUs, zero copy | 1,500 | 15.4 | 21 | 1.35 | 2.35 |
| 2 GPUs, multiscale (level 0 on the GPU) | 1,500 | 15.4 | 55 | 3.6 | 4.6 |
| CPU, zero copy | 2,000 | 20.5 | 78 | 3.8 | 4.8 |
| CPU, frames copied into the writer | 600 | 6.1 | 38 | 6.1 | 7.1 |

- The simulated camera reuses frames already in memory, so the RAM traffic column leaves out the camera's own write of
  each frame. In a recording, the frame grabber writes every frame into RAM by DMA: add 1 byte per byte written (last
  column).
- On the GPU, zero copy reads each frame once for the upload; the compressed output is small. Multiscale adds the
  coarser levels, which are derived and compressed on the CPU.
- On the CPU, the encoder reads each frame, writes the compressed chunks and assembles the shards. Copying frames into
  the writer adds a read and a write of every frame, and the copy and the compression compete for bandwidth. Next to the
  copy load, the copy path dropped 47 % of the frames at 1,200 fps (it sustains about 2,100 fps alone), whereas the
  other paths kept up. This is why copying frames costs 40–46 % of the CPU rate (README, Fig. 2, d).
- Limits of the method: it assumes that the total RAM throughput stays the same when the two loads mix. On the CPU paths
  the writer also shares cores with the load, so those values are upper bounds. Only 8-bit frames were measured. For
  16-bit frames the CPU value is lower: the Sony IMX411 was written at 79.7 GB/s with 8 DIMMs, which bounds its traffic
  to about 3.4 bytes per byte written.

## RAM traffic at the maximum camera rates

**Table 3.** RAM traffic of a recording at the maximum rate of the real camera, including the camera's write (camera
rate from README Table 2 × the last column of Table 2 above).

| Camera | Camera rate (GB/s) | 1 GPU, zero copy (GB/s) | 2 GPUs, zero copy (GB/s) | 2 GPUs, multiscale (GB/s) | CPU, zero copy (GB/s) | CPU, copy (GB/s) |
|---|---|---|---|---|---|---|
| Sony IMX411 | 1.9 | 4 | 4 | 9 | 9 | 13 |
| Teledyne Kinetix | 5.1 | 11 | 12 | 23 | 24 | 36 |
| Sony IMX811 | 6.1 | 13 | 14 | 28 | 29 | 43 |
| GSPRINT4521 | 21.0 | – (needs 2 GPUs) | 49 | 97 | 101 | 149 |

## Recommendations

Plan for at most about half of the copy bandwidth of Table 1. That leaves room for the start of a recording (allocating
and page-locking the block buffers is several times heavier than steady streaming), the operating system and the drive
writes of a disk sink.

- **8 DIMMs** (one per channel on 8 of 12 channels): covers every camera of Table 3 on the GPU writer and on the CPU
  writer in zero copy, including the GSPRINT4521 (about 100 GB/s). The rates of the README were measured with 8 DIMMs.
- **12 DIMMs**: for the CPU writer with frames copied into it at the GSPRINT4521 rate (about 150 GB/s), and for more than
  two GPUs. It also leaves the most room for other work on the same host during a recording.
- **4–6 DIMMs**: probably enough for the Sony IMX411, the Teledyne Kinetix and the Sony IMX811 on the GPU writer (at most
  about 30 GB/s), but not tested. Not recommended for the GSPRINT4521.
- **Capacity**: the writer holds its block buffers (blocks in flight + 1, each one block of frames: 5.2 GB for a
  512-frame Teledyne Kinetix block) plus the frame grabber's ring. The benchmarks of the README used up to about
  250–330 GB; with 96 GB DIMMs, 8 DIMMs (768 GB) are enough for all of them.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).
