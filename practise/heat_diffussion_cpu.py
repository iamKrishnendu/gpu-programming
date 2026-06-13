
# Step-1 a temparature gid : create a grid where temp is 100 degree at it's center

import numpy as np
N = 5 # grid size

grid = np.zeros((N,N)) # grid initialized with 0

grid[2,2] = 100 # center of the grid where temp is 100 deg

print(grid)


# Applying Stencil : Naighbor based computation method, which helps to detects cell state changes

for step in range(100):

    new_grid = grid.copy()

    for i in range(1, N-1):
        for j in range (1, N-1):
            new_grid [i,j] = (
        grid[i-1, j]+
        grid[i+1, j]+
        grid[i, j-1]+
        grid[i, j+1]
    )/4
            
    grid = new_grid

print(grid)

