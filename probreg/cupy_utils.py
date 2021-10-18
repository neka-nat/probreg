import cupy as cp

_BLOCK_SIZE = 16

squard_norm_outer_kernel = cp.RawKernel(
    r"""
extern "C" __global__
void squard_norm_outer_kernel(const float* x, const float* y,
                              int dim, int nx, int ny, float* res) {
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if (row < nx && col < ny) {
        res[row * ny + col] = 0.0;
        for (int i = 0; i < dim; ++i) {
            float diff = x[row * 3 + i] - y[col * 3 + i];
            res[row * ny + col] += diff * diff;
        }
    }
}
""",
    "squard_norm_outer_kernel",
)


def squared_kernel_sum(x, y):
    xc = cp.asarray(x, dtype=cp.float32, order="C")
    yc = cp.asarray(y, dtype=cp.float32, order="C")
    nx = xc.shape[0]
    ny = yc.shape[0]
    dim = xc.shape[1]
    res = cp.zeros((nx, ny), dtype=cp.float32, order="C")
    grid = ((nx + _BLOCK_SIZE - 1) // _BLOCK_SIZE, (ny + _BLOCK_SIZE - 1) // _BLOCK_SIZE, 1)
    squard_norm_outer_kernel(grid, (_BLOCK_SIZE, _BLOCK_SIZE, 1), (xc, yc, dim, nx, ny, res))
    return res.sum() / (nx * ny * dim)


def rbf_kernel(x, y, beta):
    xc = cp.asarray(x, dtype=cp.float32, order="C")
    yc = cp.asarray(y, dtype=cp.float32, order="C")
    nx = xc.shape[0]
    ny = yc.shape[0]
    dim = xc.shape[1]
    res = cp.zeros((nx, ny), dtype=cp.float32, order="C")
    grid = ((nx + _BLOCK_SIZE - 1) // _BLOCK_SIZE, (ny + _BLOCK_SIZE - 1) // _BLOCK_SIZE, 1)
    squard_norm_outer_kernel(grid, (_BLOCK_SIZE, _BLOCK_SIZE, 1), (xc, yc, dim, nx, ny, res))
    return cp.exp((-res / (2.0 * beta)))
