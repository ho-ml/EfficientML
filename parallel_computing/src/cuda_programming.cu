#include "matmul.h"
#include <cuda_runtime.h>
#include <assert.h>

const int threadDim = 32;
const int TILE_SIZE = threadDim;

__global__ void matmul_func_cuda(
    const float *a, const float *b, float *c, int col_a, int col_b
) {
    // shared memory 할당
    __shared__ float as[TILE_SIZE][TILE_SIZE];
    __shared__ float bs[TILE_SIZE][TILE_SIZE];

    // initialize
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0;

    // implementation
    for (int i = 0; i < col_a / TILE_SIZE; i++) {
        as[threadIdx.y][threadIdx.x] = a[(blockIdx.y * TILE_SIZE + threadIdx.y) * col_a + (i * TILE_SIZE + threadIdx.x)];
        bs[threadIdx.y][threadIdx.x] = b[(i * TILE_SIZE + threadIdx.y) * col_b + (blockIdx.x * TILE_SIZE + threadIdx.x)];
        
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            val += as[threadIdx.y][k] * bs[k][threadIdx.x];

        __syncthreads();
    }

    c[row * col_b + col] = val;
}

namespace matmul{
    void MatmulOperator::mat_mul_cuda(const struct matmul_params *params){
        // initialize
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A, *data_B, *data_C;

        // sanity check
        assert(C->row % threadDim == 0);
        assert(C->column % threadDim == 0);
        assert(B->row % threadDim == 0);

        // allocate memory in GPU
        cudaMalloc(&data_A, A->column * A->row * sizeof(float));
        cudaMalloc(&data_B, B->column * B->row * sizeof(float));
        cudaMalloc(&data_C, C->column * C->row * sizeof(float));
        
        // copy data to GPU
        cudaMemcpy(data_A, A->data_ptr, A->column * A->row * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(data_B, B->data_ptr, B->column * B->row * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(data_C, C->data_ptr, C->column * C->row * sizeof(float), cudaMemcpyHostToDevice);

        // break the matrix into blocks
        const dim3 threadsPerBlock(threadDim, threadDim); 
        const dim3 numBlocks(C->column / threadsPerBlock.x, C->row / threadsPerBlock.y);

        // implementation
        matmul_func_cuda<<< numBlocks, threadsPerBlock>>>(data_A, data_B, data_C, A->column, B->column);

        // get the result from GPU
        cudaMemcpy(C->data_ptr, data_C, C->column * C->row * sizeof(float), cudaMemcpyDeviceToHost);
    }
}