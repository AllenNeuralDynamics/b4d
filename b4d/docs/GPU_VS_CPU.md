# GPU versus CPU for b4d

This page compares the write rate and the cost of the b4d pipeline (b4d compression, optional multiscale pyramid, zarr
v3 shards) on CPUs and GPUs, so that the hardware for an acquisition system can be chosen from measured numbers. Rates
were measured on one AMD EPYC 9754 (128 cores) and two NVIDIA RTX A6000 GPUs (PCIe Gen4 x16) with the default codec at
q = 1σ, frames written in place (zero copy) and an in-memory sink; they are those of Table 2 and Figure 2 of the
[README](../README.md#overview). Figures for other hardware are estimates and are marked as such. Prices are as of
2026-11-17 (Section 7); hardware prices rose by 50–90 % within 2025–2026, so check them before buying.

## 1. The work per pixel

Per pixel, b4d does one table lookup (transform and quantization), one prediction from the three pixels of the
preceding row, one sign folding and Rice coding (a code parameter per group of 128 values, then packing into eight bit
streams). A multiscale pyramid adds 14 % more pixels and the box means that produce them.

On the CPU the writer (codec, gather and shard assembly) encodes 0.33–0.39 Gpixel/s per core, similar for 8-bit and
16-bit pixels, so 16-bit frames are written at about twice the byte rate. The work is independent per row, and the
rate scales with the number of cores.

## 2. Measured rates

Highest frame rate written with zero dropped frames for at least 30 s.

| | Sony IMX411 (16-bit) | Sony IMX811 (16-bit) | GSPRINT4521 (8-bit) | Teledyne Kinetix (8-bit) |
|---|---|---|---|---|
| CPU, 128 cores (GB/s) | 79.7 | 83.2 | 42.6 | 39.2 |
| 1 × A6000 (GB/s) | 22.1 | 21.6 | 14.2 | 13.5 |
| 2 × A6000 (GB/s) | 41.9 | 41.1 | 28.7 | 25.1 |
| Rate lost to multiscale, CPU (%) | 23.3 | 22.5 | 22.6 | 19.6 |
| Rate lost to multiscale, 1 or 2 GPUs (%) | ≤ 3.4 | ≤ 3.4 | ≤ 3.4 | ≤ 3.4 |
| Rate lost to copying frames, CPU (%) | 40.1 | 41.5 | 45.7 | 44.9 |
| Rate lost to copying frames, 1 GPU (%) | 3.2 | 3.2 | 3.2 | 3.2 |
| Rate lost to copying frames, 2 GPUs (%) | 12.0 | 12.0 | 27.3 | 27.3 |

Rates were measured in steps of 10 %, so losses up to 3.4 % are within one step. With a GPU, the full-resolution level
of the pyramid is encoded on the GPU and the coarser levels on host cores that are otherwise idle; on the CPU all levels
share the same cores. With frames copied into the writer, the CPU sustains 21–23 GB/s of 8-bit and 47–49 GB/s of 16-bit frames: the copy,
not the compression, sets the rate. With two GPUs the copy doubles the host traffic.

## 3. CPU

| CPU | Cores | 16-bit frames (GB/s) | 8-bit frames (GB/s) |
|---|---|---|---|
| AMD EPYC 9754 | 128 | 79.7–83.2 | 39.2–42.6 |
| AMD EPYC 9754 | 64 | 46.0–47.1 | 23.9–24.4 |
| AMD EPYC 9996 (estimate) | 256 | ~160 | ~80 |

The 256-core estimate is twice the 128-core rate. A 256-core system costs about 23,000 USD and runs the camera
software and the storage on the same machine. The CPU draws about 500 W at full load, about 6 W per GB/s.

## 4. GPU

A GPU pipeline has three parts, and only one of them is the codec:

1. **Frames onto the GPU.** PCIe Gen4 x16 carries 31.5 GB/s per direction, PCIe Gen5 x16 63 GB/s. On one A6000 the
   16-bit write rate levels off at 19–20 GB/s in the SM sweep and 21.6–22.1 GB/s in the ladders: the encoder is limited by the transfer of the raw frames. Frames already
   in GPU memory (GPUDirect RDMA from the frame grabber) are encoded without an upload; this path was not measured.
2. **The codec.** One CUDA kernel, byte-identical to the C encoder. The rate rises approximately linearly with the
   streaming multiprocessors (SMs), at 0.18–0.20 Gpixel/s per SM; 8-bit frames reach 13.5–14.2 Gpixel/s on all 84 SMs (ladders, Table 2).
3. **Shards to storage.** The compressed data (1/3 to 1/6 of the input at q = 1σ) returns over PCIe. The coarser
   pyramid levels are computed on the CPU.

One GPU reaches its rate with 2–4 host threads. The encoder keeps each block of frames in device memory, plus about
2.5 GB of buffers, so for large 16-bit sensors the device memory sets the smallest usable card (Section 5). A PCIe Gen5
card such as the RTX PRO 6000 Blackwell could reach up to twice the A6000's 16-bit rate if its kernel keeps up; this was
not measured.

## 5. Smallest hardware for each camera

What each camera needs at its maximum frame rate, with a margin of 1.5: GPU SMs at the A6000's rate per SM
(conservative for newer cards), CPU cores at the measured rate per core, and device memory for the blocks in flight.

| Camera | Frame rate (fps) | Raw rate (Gpixel/s) | Raw rate (GB/s) | GPU SMs | GPU memory (GB) | CPU cores | Smallest GPU |
|---|---|---|---|---|---|---|---|
| Sony IMX411 | 6.3 | 1.0 | 1.9 | 8 | 22 | 5 | RTX PRO 4500 Blackwell (32 GB) |
| Sony IMX811 | 12.4 | 3.0 | 6.1 | 25 | 34 | 13 | RTX A6000 or RTX PRO 5000 Blackwell (48 GB) |
| Teledyne Kinetix | 500 | 5.1 | 5.1 | 43 | 8 | 20 | RTX 4000 Ada (48 SMs) |
| GSPRINT4521 | 1,000 | 21.0 | 21.0 | 175 | 25 | 80 | 2 × RTX A6000, or RTX PRO 6000 Blackwell (188 SMs; estimate) |

- For three of the four cameras an A6000 has more compute than needed: the Sony IMX411 needs about 8 of its 84 SMs, the
  Sony IMX811 25 and the Teledyne Kinetix 43. Its 48 GB of memory is what the Sony IMX811 needs.
- An encoder that streamed each block through a small device buffer would need only a few GB of device memory; the
  smallest card for both Sony sensors would then be an RTX 2000 Ada (22 SMs, 16 GB, about 625 USD). This mode is not
  implemented.
- On the CPU, every camera except the GSPRINT4521 needs a fraction of one socket. One 128-core EPYC 9754 covers all four
  cameras, the GSPRINT4521 at 2.0 times its maximum rate (1.6 times with multiscale on).

## 6. Cost per petabyte written

Hardware amortized over 5 years of continuous use at full rate (an upper bound on utilization; real systems acquire in
bursts, which favours the platform that is cheaper when idle), per petabyte (10¹⁵ bytes) of raw data written. 16-bit frames.

| Platform | Raw rate (GB/s) | Hardware (USD) | Hardware per PB (USD) | Power at load (kW) | Energy per PB (kWh) |
|---|---|---|---|---|---|
| 128-core CPU system | 80–83 | ~12,000 | ~0.9 | 0.7 | ~2.4 |
| 256-core CPU system (estimate) | ~160 | ~23,000 | ~0.9 | ~0.9 | ~1.6 |
| 1 × A6000 + GPU host | 22 | ~10,000 | ~2.9 | ~0.5 | ~6.3 |
| 2 × A6000 + GPU host (power: estimate) | 41–42 | ~16,000 | ~2.4 | ~0.75 | ~5.0 |
| 1 × RTX PRO 6000 + GPU host (estimate) | ≤ 42 | ~20,000 | ≥ 3.0 | ~0.8 | ≥ 5.3 |

Power at load: the CPU system measured at the wall; one A6000 measured 249 W at the sustained 16-bit rate (300 W limit,
busy 98 % of the time, 12 W idle), plus about 0.25 kW estimated for the GPU host; two A6000 assumed to draw twice the
card power. A petabyte of raw data is about 3.5 hours of the Sony IMX411 at the CPU rate. One 128-core CPU writes
3.6–3.9 times the 16-bit rate of one A6000 and 1.9–2.0 times that of two, for less than the price of one Blackwell workstation card. A CPU at full
rate is fully occupied, whereas a GPU leaves most host cores free for other work.

## 7. Prices

New hardware in the United States, as of 2026-11-17 (list or retail prices). Workstation GPU prices are set by resellers and rose by 50–90 %
during 2025–2026.

| CPU or system | Cores | Price (USD) |
|---|---|---|
| AMD EPYC 9754 | 128 | ~5,000 |
| AMD EPYC 9996 | 256 | ~15,000 |
| 128-core system (EPYC 9754, 768 GB memory; estimate) | 128 | ~12,000 |
| GPU host workstation (256 GB memory; estimate) | 16–32 | ~4,000 |

| GPU | SMs | Memory (GB) | PCIe | Price (USD) |
|---|---|---|---|---|
| RTX 2000 Ada | 22 | 16 | Gen4 x8 | ~625 |
| RTX 4000 Ada | 48 | 20 | Gen4 x16 | ~1,700 |
| RTX PRO 4000 Blackwell | 70 | 24 | Gen5 x16 | ~1,500–3,100 |
| RTX PRO 4500 Blackwell | 82 | 32 | Gen5 x16 | ~3,400 |
| RTX A6000 | 84 | 48 | Gen4 x16 | ~6,000 |
| RTX PRO 5000 Blackwell | 110 | 48 | Gen5 x16 | ~7,300–8,000 |
| RTX PRO 6000 Blackwell | 188 | 96 | Gen5 x16 | ~16,000 |

## 8. Operating considerations

| | CPU (AMD EPYC 9754 server) | GPU (NVIDIA RTX A6000 in a workstation) |
|---|---|---|
| Memory error correction | ECC DDR5, on by default | ECC device memory, off by default; `nvidia-smi -e 1` and a reboot, a few percent less memory and bandwidth |
| Host memory | ECC in a server; workstation hosts often without | Frames pass through host memory: the host needs ECC as well |
| Output check | None at run time (the reference encoder) | Compared with the C encoder on the first block of each recording |
| Load in b4d at full rate | All cores, at the power limit | 16-bit frames: 249 W of the 300 W limit, busy 98 % of the time |
| Cooling | Passive heatsink, redundant hot-swappable chassis fans | One blower fan per card, not redundant |
| Most likely failure | Chassis fan, replaced while running | Card fan; the card is replaced |
| Warranty (years) | 3, typically | 3, typically |
| Temperature monitoring | Server management controller | `nvidia-smi` |

**Error-correcting memory.** A memory error during a multi-day acquisition corrupts compressed data silently: a flipped
bit in a frame changes one pixel, but a flipped bit in a Rice stream or a chunk header makes the rest of that chunk
undecodable. Server DDR5 memory corrects such errors (ECC); for unattended acquisitions, use a host with ECC memory.
RTX workstation GPUs have ECC for their device memory, but it is off by default. It is enabled with `nvidia-smi -e 1`
(administrator, takes effect after a reboot) and reduces the usable device memory and bandwidth by a few percent. The
GPU rates on this page were measured with ECC off; the effect on the PCIe-limited 16-bit rates is expected to be small
but was not measured. The GPU encoder checks its output against the C encoder on the first block of each recording,
which detects a faulty setup but not an error later in the recording.

**Lifetime.** Server CPUs and workstation GPUs are specified for continuous operation at their power limit, and
silicon ageing depends on temperature and voltage, not on the workload. A CPU running b4d at its full rate is at its
power limit; an A6000 encoding 16-bit frames at its full rate draws 249 W of its 300 W limit. The parts most likely
to fail are the cooling: a server CPU is cooled by redundant, hot-swappable chassis fans, a blower-type workstation
GPU by one fan. Warranties are typically 3 years for both. Monitor the GPU temperature (`nvidia-smi`) and the CPU
temperature (the server's management controller), keep the filters clean, and keep a margin below the measured rate
(Section 9). Acquisitions in bursts reduce wear in proportion.

## 9. Recommendation

- **Up to ~80 GB/s of 16-bit frames:** one 128-core CPU with the native writer and frames written in place (a copy of every frame
  costs 40–46 % on the CPU). It sustains
  2.0 (GSPRINT4521) to 42 (Sony IMX411) times the maximum rate of the four cameras tested. For unattended multi-day
  acquisitions, keep a margin below the measured rate.
- **GPU:** one A6000 covers up to ~22 GB/s of 16-bit frames, two up to ~42 GB/s. For a single camera a smaller card is
  usually enough (Section 5); for large 16-bit sensors the device memory, not the compute, sets the minimum.
- **~80–160 GB/s:** a 256-core CPU (estimate), or more GPUs.
- **Multiscale:** with a GPU the pyramid costs at most 3.4 %; on the CPU it costs 20–23 %.
- **Memory:** ECC on the host and on the GPUs (`nvidia-smi -e 1`) for unattended acquisitions (Section 8).
- **Storage:** the drive must sustain the raw rate divided by the compression ratio (× 1.2 with a pyramid) for the whole
  acquisition, at its steady-state rate.

## Citing

If you use b4d, please cite it as given in [CITATION.cff](../../CITATION.cff) (DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)), together with B3D: Balázs B, Deschamps J, Albert M, Ries J,
Hufnagel L. A real-time compression library for microscopy images. bioRxiv 164624 (2017).
doi:[10.1101/164624](https://doi.org/10.1101/164624).
