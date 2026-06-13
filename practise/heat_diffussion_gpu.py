# to run python code on GPU, cuda helps    
from numba import cuda
import numpy as np
import time

def heat_kernel(old_grid, new_grid):

    row, col = cuda.grid(2)

    rows = old_grid.shape[0]
    cols = old_grid.shape[1]

    if 0 < row < rows-1 and 0 < col < cols-1:

        new_grid[row, col] = (
            old_grid[row-1, col] +
            old_grid[row+1, col] +
            old_grid[row, col-1] +
            old_grid[row, col+1]
        ) * 0.25

def main():

    N = 512
    iterations = 500

    # Create temperature grid
    grid = np.zeros((N, N), dtype=np.float32)

    # Heat source in center
    grid[N//2, N//2] = 100.0

    print("Initial center temperature:")
    print(grid[N//2-2:N//2+3, N//2-2:N//2+3])

    # Copy to GPU
    d_old = cuda.to_device(grid)
    d_new = cuda.device_array_like(grid)

    # Thread configuration
    threads_per_block = (16, 16)

    blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (N + threads_per_block[1] - 1) // threads_per_block[1]

    blocks_per_grid = (
        blocks_per_grid_x,
        blocks_per_grid_y
    )

    # Timing
    start = time.time()

    for _ in range(iterations):

        heat_kernel[
            blocks_per_grid,
            threads_per_block
        ](
            d_old,
            d_new
        )

        d_old, d_new = d_new, d_old

    cuda.synchronize()

    gpu_time = time.time() - start

    # Copy result back
    result = d_old.copy_to_host()

    print("\nFinal center region:")
    print(result[N//2-2:N//2+3, N//2-2:N//2+3])

    print(f"\nGPU Time: {gpu_time:.4f} seconds")


if __name__ == "__main__":
    main()



## As my system don't have NVIDIA GPU hence, used this collab to execute this
#https://colab.research.google.com/drive/1grX-WYIc3xZ2wfPSJ8fU4rZH10GZMvRL?hl=en#scrollTo=sX4jqcR2RaDk