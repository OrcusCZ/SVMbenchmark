#include "calc_x2.h"
#include "../cuda_utils.h"

static __global__ void kernelSumX2(float * x2, const float * x, int num_vec, int dim, int dim_aligned)
{
    for (int k = blockDim.y * blockIdx.x + threadIdx.y; k < num_vec; k += gridDim.x * blockDim.y)
    {
        float sum = 0;
        for (int d = threadIdx.x; d < dim; d += blockDim.x)
        {
            float v = x[dim_aligned * k + d];
            sum += v * v;
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0)
            x2[k] = sum;
    }
}

static __global__ void kernelSumX2T(float * x2, const float * x, int num_vec, int num_vec_aligned, int dim)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float sum = 0;
        for (int d = 0; d < dim; d++)
        {
            float v = x[num_vec_aligned * d + k];
            sum += v * v;
        }
        x2[k] = sum;
    }
}

static __global__ void kernelSumX2Sparse(float * x2, OrcusSVM1B::csr_gpu x, int num_vec)
{
    for (int k = blockDim.y * blockIdx.x + threadIdx.y; k < num_vec; k += gridDim.x * blockDim.y)
    {
        float sum = 0;
        int beg = x.rowOffsets[k];
        int end = x.rowOffsets[k + 1];
        for (int d = beg + threadIdx.x; d < end; d += blockDim.x)
        {
            float v = x.values[d];
            sum += v * v;
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0)
            x2[k] = sum;
    }
}

static __global__ void kernelSumX2Sparse(float * x2, OrcusSVM1B::ellpack_gpu x, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float sum = 0;
        int row_len = x.rowLen[k];
        for (int d = 0; d < row_len; d++)
        {
            float v = x.values[num_vec * d + k];
            sum += v * v;
        }
        x2[k] = sum;
    }
}

static __global__ void kernelSumX2Sparse(float * x2, OrcusSVM1B::jds_gpu x, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float sum = 0;
        int row_len = x.rowLen[k];
        for (int d = 0; d < row_len; d++)
        {
            float v = x.values[x.colStart[d] + k];
            sum += v * v;
        }
        x2[x.rowPerm[k]] = sum;
    }
}

void OrcusSVM1B::computeX2Dense(float * d_x2, const float * d_x, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    dim3 dimBlockSumX2(32, 8);
    kernelSumX2<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.y)), dimBlockSumX2>>>(d_x2, d_x, num_vec, dim, dim_aligned);
}

void OrcusSVM1B::computeX2DenseT(float * d_x2, const float * d_xT, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    dim3 dimBlockSumX2(256);
    kernelSumX2T<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.x)), dimBlockSumX2>>>(d_x2, d_xT, num_vec, num_vec_aligned, dim);
}

void OrcusSVM1B::computeX2Sparse(float * d_x2, const csr_gpu & x, int num_vec)
{
    dim3 dimBlockSumX2(32, 8);
    kernelSumX2Sparse<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.y)), dimBlockSumX2>>>(d_x2, x, num_vec);
}

void OrcusSVM1B::computeX2Sparse(float * d_x2, const ellpack_gpu & x, int num_vec)
{
    dim3 dimBlockSumX2(256);
    kernelSumX2Sparse<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.x)), dimBlockSumX2>>>(d_x2, x, num_vec);
}

void OrcusSVM1B::computeX2Sparse(float * d_x2, const jds_gpu & x, int num_vec)
{
    dim3 dimBlockSumX2(256);
    kernelSumX2Sparse<<<dim3(getgriddim<int>(num_vec, dimBlockSumX2.x)), dimBlockSumX2>>>(d_x2, x, num_vec);
}
