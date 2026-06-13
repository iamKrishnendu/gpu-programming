"""
hybrid_pde_solver.py
====================================================================
Stencil-based solver for the 2D heat-diffusion PDE that runs on the
CPU, the GPU, or *both at once*, with efficient host<->device data
movement.

    PDE (heat / diffusion):     du/dt = alpha * (d2u/dx2 + d2u/dy2)

    Discretisation (explicit FTCS, 5-point stencil):
        u[i,j]^{n+1} = u[i,j]^n
                       + c * ( u[i+1,j] + u[i-1,j]
                             + u[i,j+1] + u[i,j-1] - 4 u[i,j] )
        with c = alpha * dt / dx^2,  stable while c <= 0.25.

Three execution backends share one interface:

  * CPUSolver    -- Numba @njit(parallel=True). prange spreads the
                    stencil sweep across CPU *threads* (one per core).

  * GPUSolver    -- Numba CUDA kernel. The grid is tiled into thread
                    blocks; each block stages its tile (+halo) in
                    shared memory so neighbouring threads reuse loads.
                    A second kernel computes the convergence residual
                    with a *warp*-level shuffle reduction, then a
                    block reduction, then an atomic max.

  * HybridSolver -- Splits the domain into a GPU slab (top) and a CPU
                    slab (bottom). Each step both halves advance in
                    parallel and exchange a single ghost row across
                    the PCIe boundary. The per-step transfer is O(width)
                    instead of O(width*height): that is what makes the
                    CPU<->GPU transition cheap.

The GPU paths degrade gracefully: if no CUDA device is present the
solver reports it and the CPU backend is used instead. The CUDA
kernels can still be exercised on a CPU via Numba's simulator:

    NUMBA_ENABLE_CUDASIM=1 python hybrid_pde_solver.py

Run normally on a CUDA machine for the real thing:

    python hybrid_pde_solver.py
====================================================================
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from numba import cuda, njit, prange, float32

# Whether a usable CUDA target exists (real device OR the simulator).
GPU_OK = cuda.is_available()

# The CPU simulator (NUMBA_ENABLE_CUDASIM=1) lacks warp-shuffle
# intrinsics, so we fall back to a shared-memory reduction there while
# keeping the true warp path for real hardware.
IS_SIM = bool(int(os.environ.get("NUMBA_ENABLE_CUDASIM", "0")))

# Tile / block dimensions for the GPU. 16x16 = 256 threads = 8 warps,
# a good default occupancy on most architectures.
TILE = 16


# --------------------------------------------------------------------
# CPU backend  --  multi-threaded stencil sweep
# --------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=True)
def _cpu_step(u, u_new, c):
    """One explicit heat-equation sweep over the interior of `u`.

    `prange` lets Numba split the outer loop across CPU threads; each
    thread owns a band of rows, so the whole grid updates in parallel.
    """
    h, w = u.shape
    for i in prange(1, h - 1):          # parallel over rows (threads)
        for j in range(1, w - 1):       # serial within a row
            u_new[i, j] = u[i, j] + c * (
                u[i + 1, j] + u[i - 1, j]
                + u[i, j + 1] + u[i, j - 1]
                - 4.0 * u[i, j]
            )


@njit(parallel=True, fastmath=True, cache=True)
def _cpu_residual(u, u_new):
    """Max absolute change over the grid -- a convergence measure.

    Each thread reduces its own rows into a private slot of `rowmax`
    (no cross-thread race); the host then takes the global max. This
    avoids relying on prange recognising a conditional max-reduction.
    """
    h, w = u.shape
    rowmax = np.zeros(h, np.float64)
    for i in prange(1, h - 1):          # one thread-band per row group
        local = 0.0
        for j in range(1, w - 1):
            d = abs(u_new[i, j] - u[i, j])
            if d > local:
                local = d
        rowmax[i] = local
    return rowmax.max()


# --------------------------------------------------------------------
# GPU backend  --  shared-memory stencil + warp-reduced residual
# --------------------------------------------------------------------
if GPU_OK:

    @cuda.jit(fastmath=True)
    def _gpu_step(u, u_new, c):
        """Tiled 5-point stencil.

        Threads in a block cooperatively load a (TILE+2)^2 patch
        (interior tile + 1-cell halo) into shared memory, sync, then
        each thread updates its own cell reading only fast on-chip
        memory. This cuts global-memory traffic ~5x versus the naive
        kernel because every interior value is read once, not five
        times.
        """
        sh = cuda.shared.array((TILE + 2, TILE + 2), float32)

        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        row = cuda.blockIdx.y * TILE + ty       # global row
        col = cuda.blockIdx.x * TILE + tx       # global col
        lr = ty + 1                             # local row (halo offset)
        lc = tx + 1                             # local col

        h, w = u.shape

        # --- cooperative load: centre cell ---
        if row < h and col < w:
            sh[lr, lc] = u[row, col]

        # --- cooperative load: halo cells (block edge threads) ---
        # each halo guard checks BOTH its own axis and the orthogonal
        # axis so threads past the array edge never read out of bounds.
        if tx == 0 and col > 0 and row < h:
            sh[lr, 0] = u[row, col - 1]
        if tx == TILE - 1 and col < w - 1 and row < h:
            sh[lr, lc + 1] = u[row, col + 1]
        if ty == 0 and row > 0 and col < w:
            sh[0, lc] = u[row - 1, col]
        if ty == TILE - 1 and row < h - 1 and col < w:
            sh[lr + 1, lc] = u[row + 1, col]

        cuda.syncthreads()                      # tile is fully staged

        # --- update interior cells only ---
        if 0 < row < h - 1 and 0 < col < w - 1:
            u_new[row, col] = sh[lr, lc] + c * (
                sh[lr - 1, lc] + sh[lr + 1, lc]
                + sh[lr, lc - 1] + sh[lr, lc + 1]
                - 4.0 * sh[lr, lc]
            )

    @cuda.jit(fastmath=True)
    def _gpu_residual(u, u_new, out_max):
        """Max |u_new - u| via warp-shuffle + block + atomic reduction.

        Reduction hierarchy (fast -> slow):
          1. within a warp   : __shfl_down butterfly, no memory
          2. across warps    : one slot per warp in shared memory
          3. across blocks   : single atomic_max to global `out_max`
        """
        sh = cuda.shared.array(TILE * TILE // 32, float32)  # 1 slot/warp

        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        row = cuda.blockIdx.y * TILE + ty
        col = cuda.blockIdx.x * TILE + tx
        h, w = u.shape

        # per-thread value
        val = float32(0.0)
        if 0 < row < h - 1 and 0 < col < w - 1:
            val = abs(u_new[row, col] - u[row, col])

        # flat lane / warp ids inside the block
        lane = (ty * TILE + tx) % 32
        warp = (ty * TILE + tx) // 32

        # 1) warp-level max reduction (shuffle down by 16,8,4,2,1)
        offset = 16
        while offset > 0:
            other = cuda.shfl_down_sync(0xFFFFFFFF, val, offset)
            if other > val:
                val = other
            offset //= 2

        # lane 0 of each warp writes the warp max
        if lane == 0:
            sh[warp] = val
        cuda.syncthreads()

        # 2) first warp reduces the per-warp maxima
        nwarps = (TILE * TILE) // 32
        if warp == 0:
            v = sh[lane] if lane < nwarps else float32(0.0)
            offset = 16
            while offset > 0:
                other = cuda.shfl_down_sync(0xFFFFFFFF, v, offset)
                if other > v:
                    v = other
                offset //= 2
            # 3) one atomic per block into the global maximum
            if lane == 0:
                cuda.atomic.max(out_max, 0, v)

    @cuda.jit(fastmath=True)
    def _gpu_residual_shared(u, u_new, out_max):
        """Block max via a shared-memory tree reduction + atomic max.

        Equivalent result to the warp-shuffle kernel, using only
        primitives the CUDA simulator also supports, so the program
        runs end-to-end with NUMBA_ENABLE_CUDASIM=1 on a GPU-less box.
        """
        sh = cuda.shared.array(TILE * TILE, float32)

        ty = cuda.threadIdx.y
        tx = cuda.threadIdx.x
        row = cuda.blockIdx.y * TILE + ty
        col = cuda.blockIdx.x * TILE + tx
        h, w = u.shape

        tid = ty * TILE + tx
        val = float32(0.0)
        if 0 < row < h - 1 and 0 < col < w - 1:
            val = abs(u_new[row, col] - u[row, col])
        sh[tid] = val
        cuda.syncthreads()

        # tree reduction: halve the active set each round
        stride = (TILE * TILE) // 2
        while stride > 0:
            if tid < stride and sh[tid + stride] > sh[tid]:
                sh[tid] = sh[tid + stride]
            cuda.syncthreads()
            stride //= 2

        if tid == 0:
            cuda.atomic.max(out_max, 0, sh[0])

    # pick the residual kernel that the active target supports
    _residual_kernel = _gpu_residual_shared if IS_SIM else _gpu_residual


# --------------------------------------------------------------------
# Backend wrappers (uniform .solve() interface)
# --------------------------------------------------------------------
class CPUSolver:
    name = "CPU (Numba threads)"

    def solve(self, u0, c, steps):
        u = u0.astype(np.float32).copy()
        u_new = u.copy()
        res = 0.0
        for _ in range(steps):
            _cpu_step(u, u_new, c)
            _enforce_bc(u_new, u)             # keep Dirichlet boundary
            u, u_new = u_new, u
        res = float(_cpu_residual(u_new, u))
        return u, res


class GPUSolver:
    name = "GPU (CUDA shared-mem + warp reduce)"

    def solve(self, u0, c, steps):
        u = u0.astype(np.float32)
        h, w = u.shape

        # Transfer ONCE; iterate fully on the device; copy back ONCE.
        d_u = cuda.to_device(u)
        d_v = cuda.to_device(u.copy())        # valid (not garbage) edges
        d_res = cuda.to_device(np.zeros(1, np.float32))

        blocks = ((w + TILE - 1) // TILE, (h + TILE - 1) // TILE)
        threads = (TILE, TILE)

        for _ in range(steps):
            _gpu_step[blocks, threads](d_u, d_v, c)
            # new state is in d_v; restore its fixed boundary from d_u
            _copy_boundary_gpu[blocks, threads](d_v, d_u)
            d_u, d_v = d_v, d_u

        _residual_kernel[blocks, threads](d_u, d_v, d_res)
        out = d_u.copy_to_host()
        return out, float(d_res.copy_to_host()[0])


class HybridSolver:
    """Top slab on GPU, bottom slab on CPU, one ghost row exchanged/step.

    Only `width` floats cross the bus per step (the shared interface
    row) instead of the entire grid -- the whole point of "efficient
    CPU<->GPU transition".
    """
    name = "Hybrid (GPU slab + CPU slab, halo exchange)"

    def __init__(self, gpu_fraction=0.5):
        self.frac = gpu_fraction

    def solve(self, u0, c, steps):
        u = u0.astype(np.float32).copy()
        h, w = u.shape

        # Row `split` is the shared interface. GPU owns rows [0, split],
        # CPU owns rows [split, h-1]; both keep `split` as a ghost.
        split = max(1, min(h - 2, int(h * self.frac)))

        # ---- GPU half (rows 0..split inclusive) lives on the device ----
        g_rows = slice(0, split + 2)            # +1 ghost row from CPU
        d_u = cuda.to_device(np.ascontiguousarray(u[g_rows]))
        d_v = cuda.to_device(np.ascontiguousarray(u[g_rows]).copy())
        gh = d_u.shape[0]
        blocks = ((w + TILE - 1) // TILE, (gh + TILE - 1) // TILE)
        threads = (TILE, TILE)

        # Pinned staging buffers => fast, async-capable single-row copies
        ghost_from_cpu = cuda.pinned_array(w, np.float32)
        ghost_from_gpu = cuda.pinned_array(w, np.float32)

        # ---- CPU half (rows split-1 .. h-1) stays in host memory ----
        cpu = np.ascontiguousarray(u[split - 1:, :])
        cpu_new = cpu.copy()

        for _ in range(steps):
            # advance both halves "simultaneously"
            _gpu_step[blocks, threads](d_u, d_v, c)
            _copy_boundary_gpu[blocks, threads](d_v, d_u)
            _cpu_step(cpu, cpu_new, c)
            _enforce_bc(cpu_new, cpu)

            d_u, d_v = d_v, d_u
            cpu, cpu_new = cpu_new, cpu

            # ---- halo exchange: ONE row each way across the boundary ----
            # GPU's last computed interior row -> CPU's top ghost row
            ghost_from_gpu[:] = d_u[gh - 2, :].copy_to_host()
            cpu[0, :] = ghost_from_gpu
            # CPU's first computed interior row -> GPU's bottom ghost row
            ghost_from_cpu[:] = cpu[1, :]
            d_u[gh - 1, :] = cuda.to_device(ghost_from_cpu)

        # ---- stitch the two halves back into one array ----
        out = np.empty_like(u)
        out[:split + 1, :] = d_u.copy_to_host()[:split + 1, :]
        out[split:, :] = cpu[1:, :]

        # residual = last-step max change, consistent with the other
        # backends: GPU slab change (device) combined with CPU slab change
        d_res = cuda.to_device(np.zeros(1, np.float32))
        _residual_kernel[blocks, threads](d_u, d_v, d_res)
        res = max(float(d_res.copy_to_host()[0]),
                  float(_cpu_residual(cpu, cpu_new)))
        return out, res


# --------------------------------------------------------------------
# Boundary conditions (fixed / Dirichlet) helpers
# --------------------------------------------------------------------
@njit(cache=True)
def _enforce_bc(dst, src):
    """Copy the four edges of `src` into `dst` (boundary held fixed)."""
    h, w = dst.shape
    for j in range(w):
        dst[0, j] = src[0, j]
        dst[h - 1, j] = src[h - 1, j]
    for i in range(h):
        dst[i, 0] = src[i, 0]
        dst[i, w - 1] = src[i, w - 1]


if GPU_OK:
    @cuda.jit
    def _copy_boundary_gpu(dst, src):
        """Hold the device-array edges fixed (Dirichlet BC)."""
        row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        h, w = dst.shape
        if row < h and col < w:
            if row == 0 or row == h - 1 or col == 0 or col == w - 1:
                dst[row, col] = src[row, col]


# --------------------------------------------------------------------
# Problem setup + driver
# --------------------------------------------------------------------
def make_problem(n):
    """A square plate: hot strip on the left edge, cool elsewhere."""
    u = np.zeros((n, n), np.float32)
    u[:, 0] = 100.0                 # left wall held at 100 degrees
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, -1] = 0.0
    return u


def run(backend, u0, c, steps):
    t0 = time.perf_counter()
    out, res = backend.solve(u0, c, steps)
    dt = time.perf_counter() - t0
    cells = (u0.shape[0] - 2) * (u0.shape[1] - 2)
    mlups = cells * steps / dt / 1e6        # mega-lattice-updates/sec
    print(f"  {backend.name:46s} {dt:7.3f}s  "
          f"{mlups:8.1f} MLUP/s  final-residual={res:.3e}  "
          f"mean={out.mean():.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Hybrid CPU/GPU PDE solver")
    ap.add_argument("--n", type=int, default=512, help="grid size NxN")
    ap.add_argument("--steps", type=int, default=400, help="time steps")
    ap.add_argument("--alpha", type=float, default=0.2, help="diffusivity")
    args = ap.parse_args()

    c = args.alpha                      # absorbs dt/dx^2; keep c <= 0.25
    assert c <= 0.25, "explicit scheme unstable: need alpha*dt/dx^2 <= 0.25"

    print(f"\nGrid {args.n}x{args.n}, {args.steps} steps, c={c}")
    print(f"CUDA target available: {GPU_OK}\n")

    u0 = make_problem(args.n)

    print("Backends:")
    ref = run(CPUSolver(), u0, c, args.steps)

    if GPU_OK:
        gpu_out = run(GPUSolver(), u0, c, args.steps)
        run(HybridSolver(gpu_fraction=0.5), u0, c, args.steps)
        # cross-check GPU vs CPU (should match to single-precision eps)
        err = float(np.max(np.abs(gpu_out - ref)))
        print(f"\nGPU vs CPU max abs difference: {err:.3e} "
              f"({'OK' if err < 1e-2 else 'CHECK'})")
    else:
        print("\n  (No CUDA device -- GPU and Hybrid backends skipped.)")
        print("  Validate kernels on CPU with: "
              "NUMBA_ENABLE_CUDASIM=1 python hybrid_pde_solver.py --n 64 --steps 20")


if __name__ == "__main__":
    main()
