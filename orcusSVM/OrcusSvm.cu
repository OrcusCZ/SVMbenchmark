#include <iostream>
#include <cuda_runtime.h>
#include "OrcusSvm.h"
#include "cudaerror.cuh"
#include "debug.h"

template<typename T>
T getgriddim(T totallen, T blockdim)
{
    return (totallen + blockdim - (T)1) / blockdim;
}

template<typename T>
T rounduptomult(T x, T m)
{
    return ((x + m - (T)1) / m) * m;
}

__global__ void kernelInitg(float * g, int len)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
        g[i] = 1;
}

__global__ void kernelComputeK(float * K, const float * x, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x,
        j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < num_vec && j < num_vec)
    {
        float sum = 0;
        for (int d = 0; d < dim; d++)
        {
            float diff = x[dim_aligned * i + d] - x[dim_aligned * j + d];
            sum += diff * diff;
        }
        K[num_vec_aligned * j + i] = exp(-gamma * sum);
    }
#ifdef _DEBUG
    else
        K[num_vec_aligned * j + i] = 0;
#endif
}

__global__ void kernelSelectI(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_vec)
    {
        float y_ = y[i];
        float a_ = alpha[i];
        if ((y_ == 1 && a_ < C) || (y_ == -1 && a_ > 0))
            valbuf[i] = y[i] * g[i];
        else
            valbuf[i] = -FLT_MAX;
        idxbuf[i] = i;
    }
    else
        valbuf[i] = -FLT_MAX;
}

//first order search
__global__ void kernelSelectJ1(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_vec)
    {
        float y_ = y[i];
        float a_ = alpha[i];
        if ((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C))
            valbuf[i] = -y[i] * g[i]; //return negative, so we can use reducemax
        else
            valbuf[i] = -FLT_MAX;
        idxbuf[i] = i;
    }
    else
        valbuf[i] = -FLT_MAX;
}

__global__ void kernelReduceMaxIdxInplace(float * val, int * idx, int len)
{
    extern __shared__ float sval[];
    int * sidx = (int *)(sval + blockDim.x);

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len)
    {
        sval[threadIdx.x] = val[i];
        sidx[threadIdx.x] = idx[i];
    }
    else
    {
        sval[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (sval[threadIdx.x + s] > sval[threadIdx.x])
            {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        val[blockIdx.x] = sval[0];
        idx[blockIdx.x] = sidx[0];
    }
}

void reduceMaxIdxInplace(float * d_val, int * d_idx, int len, int reduce_block_size)
{
    //int orig_len = len;
    dim3 dimBlock = dim3(reduce_block_size);
    while (len > 1)
    {
        dim3 dimGrid = dim3(getgriddim(len, (int)dimBlock.x));
        kernelReduceMaxIdxInplace<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val, d_idx, len);
        len = dimGrid.x;
    }
    //export_cuda_buffer(d_val, 1, orig_len, sizeof(float), "reduceval.dat");
    //export_cuda_buffer(d_idx, 1, orig_len, sizeof(int), "reduceidx.dat");
}

__global__ void kernelComputeLambda(float * lambda, const float * y, const float * g, const float * K, const float * alpha, float C, int i, int j, int num_vec_aligned)
{
    float l1 = y[i] == 1 ? C - alpha[i] : alpha[i];
    float l2 = y[j] == 1 ? alpha[j] : C - alpha[j];
    float l3 = (y[i] * g[i] - y[j] * g[j]) / (K[num_vec_aligned * i + i] + K[num_vec_aligned * j + j] - 2 * K[num_vec_aligned * i + j]);
    *lambda = min(l1, min(l2, l3));
}

__global__ void kernelUpdateg(float * g, const float * lambda, const float * y, const float * K, int i, int j, int num_vec, int num_vec_aligned)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_vec)
    {
        g[k] += *lambda * y[k] * (K[num_vec_aligned * j + k] - K[num_vec_aligned * i + k]);
    }
}

__global__ void kernelUpdateAlpha(float * alpha, const float * lambda, const float * y, int i, int j, int num_vec)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_vec)
    {
        float l = *lambda;
        alpha[i] += l * y[i];
        alpha[j] -= l * y[j];
    }
}

void OrcusSvmTrain(float * alpha, float * rho, const float * x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps)
{
    float *d_alpha = nullptr,
        *d_x = nullptr,
        *d_y = nullptr,
        *d_g = nullptr,
        *d_K = nullptr,
        *d_reduceval = nullptr;
    int *d_reduceidx = nullptr;
    float *d_lambda = nullptr;

    size_t reduce_block_size = 256;
    size_t reduce_buff_size = rounduptomult(num_vec, reduce_block_size);

    assert_cuda(cudaMalloc(&d_alpha, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_x, num_vec_aligned * dim_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_y, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_g, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_K, num_vec_aligned * num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceval, reduce_buff_size * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceidx, reduce_buff_size * sizeof(int)));
    assert_cuda(cudaMalloc(&d_lambda, sizeof(float)));

    assert_cuda(cudaMemset(d_alpha, 0, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMemcpy(d_x, x, num_vec_aligned * dim_aligned * sizeof(float), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(d_y, y, num_vec_aligned * sizeof(float), cudaMemcpyHostToDevice));

    export_cuda_buffer(d_x, dim_aligned, num_vec_aligned, sizeof(float), "x.dat");

    dim3 dimBlock(256);
    dim3 dimGrid(getgriddim(num_vec_aligned, (size_t)dimBlock.x));
    kernelInitg<<<dimGrid, dimBlock>>>(d_g, num_vec_aligned);

    export_cuda_buffer(d_g, num_vec_aligned, 1, sizeof(float), "g.dat");

    dimBlock = dim3(16, 16);
    dimGrid = dim3(getgriddim(num_vec_aligned, (size_t)dimBlock.x), getgriddim(num_vec_aligned, (size_t)dimBlock.y));
    kernelComputeK<<<dimGrid, dimBlock>>>(d_K, d_x, gamma, num_vec, num_vec_aligned, dim, dim_aligned);

    export_cuda_buffer(d_K, num_vec_aligned, num_vec_aligned, sizeof(float), "K.dat");

    for (int iter = 0; iter < 20; iter++)
    {
        dimBlock = dim3(reduce_block_size);
        dimGrid = dim3(getgriddim(num_vec_aligned, (size_t)dimBlock.x));

        kernelSelectI<<<dimGrid, dimBlock>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec);
        //export_cuda_buffer(d_reduceval, 1, reduce_buff_size, sizeof(float), "reduceval.dat");
        //export_cuda_buffer(d_reduceidx, 1, reduce_buff_size, sizeof(int), "reduceidx.dat");
        reduceMaxIdxInplace(d_reduceval, d_reduceidx, num_vec_aligned, reduce_block_size);
        int ws_i;
        assert_cuda(cudaMemcpy(&ws_i, d_reduceidx, sizeof(int), cudaMemcpyDeviceToHost));

        kernelSelectJ1<<<dimGrid, dimBlock>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec);
        //export_cuda_buffer(d_reduceval, 1, reduce_buff_size, sizeof(float), "reduceval.dat");
        //export_cuda_buffer(d_reduceidx, 1, reduce_buff_size, sizeof(int), "reduceidx.dat");
        reduceMaxIdxInplace(d_reduceval, d_reduceidx, num_vec_aligned, reduce_block_size);
        int ws_j;
        assert_cuda(cudaMemcpy(&ws_j, d_reduceidx, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "Found working set pair: " << ws_i << ", " << ws_j << "; ";

        kernelComputeLambda<<<1, 1>>>(d_lambda, d_y, d_g, d_K, d_alpha, C, ws_i, ws_j, num_vec_aligned);
        float lambda;
        assert_cuda(cudaMemcpy(&lambda, d_lambda, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "Lambda: " << lambda << std::endl;

        kernelUpdateg<<<dimGrid, dimBlock>>>(d_g, d_lambda, d_y, d_K, ws_i, ws_j, num_vec, num_vec_aligned);
        kernelUpdateAlpha<<<1, 1>>>(d_alpha, d_lambda, d_y, ws_i, ws_j, num_vec);
    }

    assert_cuda(cudaFree(d_alpha));
    assert_cuda(cudaFree(d_x));
    assert_cuda(cudaFree(d_y));
    assert_cuda(cudaFree(d_g));
    assert_cuda(cudaFree(d_K));
    assert_cuda(cudaFree(d_reduceval));
    assert_cuda(cudaFree(d_reduceidx));
    assert_cuda(cudaFree(d_lambda));
}