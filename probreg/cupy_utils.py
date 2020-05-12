import cupy as cp

_BLOCK_SIZE = 512

squard_norm_outer_kernel = cp.RawKernel(r'''
extern "C" __global__
void squard_norm_outer_kernel(const float* x, const float* y,
                              int dim, int nx, int ny, float* res) {
    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);
    if (row < nx && col < ny) {
        res[row * ny + col] = 0.0;
        for (int i = 0; i < dim; ++i) {
            float diff = x[row * 3 + i] - y[col * 3 + i];
            res[row * ny + col] += diff * diff;
        }
    }
}
''', 'squard_norm_outer_kernel')

def squared_kernel_sum(x, y):
    xc = cp.asarray(x, dtype=cp.float32, order='C')
    yc = cp.asarray(y, dtype=cp.float32, order='C')
    nx = xc.shape[0]
    ny = yc.shape[0]
    dim = xc.shape[1]
    res = cp.array((nx, ny), dtype=cp.float32, order='C')
    grid = (nx * ny) // _BLOCK_SIZE
    squard_norm_outer_kernel((grid,), (_BLOCK_SIZE,), (xc, yc, nx, ny, dim, res))
    return (res**2).sum() / (nx * ny * dim)
