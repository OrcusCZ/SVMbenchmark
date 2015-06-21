#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
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

__global__ void dummyKernel()
{
}

template<typename T>
__global__ void kernelMemset(T * mem, T v, int n)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    while (k < n)
    {
        mem[k] = v;
        k += gridDim.x * blockDim.x;
    }
}

template<typename T>
void memsetCuda(T * d_mem, T v, int n)
{
    dim3 dimBlock(256);
    dim3 dimGrid(std::min(2048, getgriddim<int>(n, dimBlock.x)));
    kernelMemset<T><<<dimGrid, dimBlock>>>(d_mem, v, n);
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

__global__ void kernelComputeKv2(float * K, const float * x, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x,
        j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < num_vec)
    {
        while (j < num_vec)
        {
            float sum = 0;
            for (int d = 0; d < dim; d++)
            {
                float diff = x[dim_aligned * i + d] - x[dim_aligned * j + d];
                sum += diff * diff;
            }
            K[num_vec_aligned * j + i] = exp(-gamma * sum);
            j += gridDim.y * blockDim.y;
        }
    }
}

__global__ void kernelPow2(float * x2, const float * x, int w, int h, int pitch)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x,
        j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < w && j < h)
    {
        int idx = pitch * j + i;
        float v = x[idx];
        x2[idx] = v * v;
    }
}

__global__ void kernelRBFExp(float * K, float gamma, int num_vec, int pitch)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x,
        j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < num_vec && j < num_vec)
    {
        int idx = pitch * j + i;
        K[idx] = exp(-gamma * K[idx]);
    }
}

void computeK(float * d_K, const float * d_x, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned, cublasHandle_t cublas)
{
    //dim3 dimBlock(16, 16);
    //dim3 dimGrid(getgriddim(num_vec_aligned, (int)dimBlock.x));
    //kernelComputeKv2<<<dimGrid, dimBlock>>>(d_K, d_x, gamma, /*num_vec*/1000, num_vec_aligned, dim, dim_aligned);

    float *d_x2 = nullptr,
        *d_x2sum = nullptr,
        *d_ones = nullptr;

    int ones_size = std::max(num_vec_aligned, dim_aligned);
    assert_cuda(cudaMalloc(&d_x2, num_vec_aligned * dim_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_x2sum, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_ones, ones_size * sizeof(float)));

    //float one_val = 1;
    //assert_cuda(cudaMemset(d_ones, *(int *)&one_val, ones_size * sizeof(float)));
    //cuMemsetD32(d_ones, *(int *)&one_val, ones_size);
    memsetCuda<float>(d_ones, 1, ones_size);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(getgriddim(dim, (int)dimBlock.x), getgriddim(num_vec, (int)dimBlock.y));
    kernelPow2<<<dimGrid, dimBlock>>>(d_x2, d_x, dim, num_vec, dim_aligned);
    float a = 1,
        b = 0;
    assert_cublas(cublasSgemv(cublas, CUBLAS_OP_T, dim, num_vec, &a, d_x2, dim_aligned, d_ones, 1, &b, d_x2sum, 1));
    a = -2;
    assert_cublas(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_vec, num_vec, dim, &a, d_x, dim_aligned, d_x, dim_aligned, &b, d_K, num_vec_aligned));

    a = 1;
    assert_cublas(cublasSger(cublas, num_vec, num_vec, &a, d_x2sum, 1, d_ones, 1, d_K, num_vec_aligned));
    assert_cublas(cublasSger(cublas, num_vec, num_vec, &a, d_ones, 1, d_x2sum, 1, d_K, num_vec_aligned));

    dimGrid = dim3(getgriddim(num_vec, (int)dimBlock.x), getgriddim(num_vec, (int)dimBlock.y));
    kernelRBFExp<<<dimGrid, dimBlock>>>(d_K, gamma, num_vec, num_vec_aligned);

    assert_cuda(cudaFree(d_x2));
    assert_cuda(cudaFree(d_x2sum));
    assert_cuda(cudaFree(d_ones));
}

void computeKDiag(float * d_KDiag, int num_vec)
{
    //K[i,i] is always 1 for RBF kernel, let's just use memset here
    memsetCuda<float>(d_KDiag, 1, num_vec);
}

__global__ void kernelSelectI(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < num_vec)
    {
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ == 1 && a_ < C) || (y_ == -1 && a_ > 0))
            valbuf[k] = y[k] * g[k];
        else
            valbuf[k] = -FLT_MAX;
        idxbuf[k] = k;
    }
    else
        valbuf[k] = -FLT_MAX;
}

//first order search
__global__ void kernelSelectJ1(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < num_vec)
    {
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C))
            valbuf[k] = -y[k] * g[k]; //return negative, so we can use reducemax
        else
            valbuf[k] = -FLT_MAX;
        idxbuf[k] = k;
    }
    else
        valbuf[k] = -FLT_MAX;
}

//second order search
__global__ void kernelSelectJ(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec, int num_vec_aligned, const int * i_ptr, const float * K)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < num_vec)
    {
        int i = *i_ptr;
        float y_ = y[k];
        float a_ = alpha[k];
        float th = y[i] * g[i];
        if (((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C)) && th > y[k] * g[k])
        {
            float den = K[num_vec_aligned * i + i] + K[num_vec_aligned * k + k] - 2 * K[num_vec_aligned * i + k];
            float v = th - y[k] * g[k];
            valbuf[k] = v * v / den;
        }
        else
            valbuf[k] = -FLT_MAX;
        idxbuf[k] = k;
    }
    else
        valbuf[k] = -FLT_MAX;
}

//second order search with cached K
__global__ void kernelSelectJCached(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec, int num_vec_aligned, const int * i_ptr, const float * K, const float * KDiag, const int * KCacheRemapIdx)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < num_vec)
    {
        int i = *i_ptr;
        int cache_row = KCacheRemapIdx[i];
        float y_ = y[k];
        float a_ = alpha[k];
        float th = y[i] * g[i];
        if (((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C)) && th > y[k] * g[k])
        {
            float den = KDiag[i] + KDiag[k] - 2 * K[num_vec_aligned * cache_row + k];
            float v = th - y[k] * g[k];
            valbuf[k] = v * v / den;
        }
        else
            valbuf[k] = -FLT_MAX;
        idxbuf[k] = k;
    }
    else
        valbuf[k] = -FLT_MAX;
}

__global__ void kernelReduceMaxIdx(float * val, int * idx, float * val_out, int * idx_out, int len)
{
    extern __shared__ float sval[];
    int * sidx = (int *)(sval + blockDim.x);

    int frame = blockDim.x * blockIdx.x,
        iter = 0;
    while (frame < len)
    {
        int i = frame + threadIdx.x;
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
            int shift = iter * gridDim.x;
            val_out[shift + blockIdx.x] = sval[0];
            idx_out[shift + blockIdx.x] = sidx[0];
        }
        __syncthreads();
        frame += gridDim.x * blockDim.x;
        iter++;
    }
}

void reduceMaxIdx(float * d_val, int * d_idx, float * d_val2, int * d_idx2, int len, int reduce_block_size)
{
    //int orig_len = len;
    /*dim3 dimBlock = dim3(reduce_block_size);
    while (len > 1)
    {
        dim3 dimGrid = dim3(getgriddim(len, (int)dimBlock.x));
        kernelReduceMaxIdx<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val, d_idx, d_val2, d_idx2, len);
        len = dimGrid.x;
    }*/
    dim3 dimBlock = dim3(reduce_block_size);
    dim3 dimGrid = dim3(std::min(256, getgriddim(len, (int)dimBlock.x)));
    //dummyKernel<<<dimGrid, dimBlock>>>();
    kernelReduceMaxIdx<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val, d_idx, d_val2, d_idx2, len);
    len = dimGrid.x;
    dimGrid.x = std::min(reduce_block_size, (int)getgriddim(dimGrid.x, dimBlock.x));
    kernelReduceMaxIdx<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val2, d_idx2, d_val, d_idx, len);
    //export_cuda_buffer(d_val, 1, orig_len, sizeof(float), "reduceval.dat");
    //export_cuda_buffer(d_idx, 1, orig_len, sizeof(int), "reduceidx.dat");
}

__global__ void kernelUpdateg(float * g, const float * lambda, const float * y, const float * K, const int * ws, int num_vec, int num_vec_aligned)
{
    int i = ws[0];
    int j = ws[1];
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_vec)
    {
        g[k] += *lambda * y[k] * (K[num_vec_aligned * j + k] - K[num_vec_aligned * i + k]);
    }
}

__global__ void kernelUpdategCached(float * g, const float * lambda, const float * y, const float * K, const int * ws, int num_vec, int num_vec_aligned, const int * KCacheRemapIdx)
{
    int i = ws[0];
    int j = ws[1];
    int i_cache_row = KCacheRemapIdx[i];
    int j_cache_row = KCacheRemapIdx[j];
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_vec)
    {
        g[k] += *lambda * y[k] * (K[num_vec_aligned * j_cache_row + k] - K[num_vec_aligned * i_cache_row + k]);
    }
}

__global__ void kernelUpdateAlphaAndLambda(float * alpha, float * lambda, const float * y, const float * g, const float * K, float C, const int * ws, int num_vec, int num_vec_aligned)
{
    int i = ws[0];
    int j = ws[1];
    float l1 = y[i] > 0 ? C - alpha[i] : alpha[i];
    float l2 = y[j] > 0 ? alpha[j] : C - alpha[j];
    float l3 = (y[i] * g[i] - y[j] * g[j]) / (K[num_vec_aligned * i + i] + K[num_vec_aligned * j + j] - 2 * K[num_vec_aligned * i + j]);
    float l = min(l1, min(l2, l3));

    *lambda = l;
    alpha[i] += l * y[i];
    alpha[j] -= l * y[j];
}

__global__ void kernelUpdateAlphaAndLambdaCached(float * alpha, float * lambda, const float * y, const float * g, const float * K, float C, const int * ws, int num_vec, int num_vec_aligned, const float * KDiag, const int * KCacheRemapIdx)
{
    int i = ws[0];
    int j = ws[1];
    int cache_row = KCacheRemapIdx[i];
    float l1 = y[i] > 0 ? C - alpha[i] : alpha[i];
    float l2 = y[j] > 0 ? alpha[j] : C - alpha[j];
    float l3 = (y[i] * g[i] - y[j] * g[j]) / (KDiag[i] + KDiag[j] - 2 * K[num_vec_aligned * cache_row + j]);
    float l = min(l1, min(l2, l3));

    *lambda = l;
    alpha[i] += l * y[i];
    alpha[j] -= l * y[j];
}

__device__ int d_cacheUpdateCnt;

__global__ void kernelCheckCache(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * x, const float * xT, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int lastPtrIdx)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = *i_ptr;
    if (KCacheRemapIdx[i] >= 0)
    {
        if (j == 0)
            KCacheRowIdx[cache_rows + (1 - lastPtrIdx)] = KCacheRowIdx[cache_rows + lastPtrIdx];
        return;  //item already in cache
    }
    int last = (KCacheRowIdx[cache_rows + lastPtrIdx] + 1) % cache_rows;
    if (j == 0)
    {
        KCacheRowIdx[cache_rows + (1 - lastPtrIdx)] = last;
        int del_i = KCacheRowIdx[last];
        if (del_i >= 0)
            KCacheRemapIdx[del_i] = -1;  //cache row for vector [del_i] will be overwritten, remove it from RemapIdx array
        //set correct indices
        KCacheRemapIdx[i] = last;
        KCacheRowIdx[last] = i;
        d_cacheUpdateCnt++;
    }

    //calculate cache matrix row [last], original index is [i]
    extern __shared__ float sx[];
    for (int idxshift = 0; idxshift < dim; idxshift += blockDim.x)
    {
        int idx = idxshift + threadIdx.x;
        if (idx < dim)
            //xi[idx] = xT[num_vec_aligned * idx + i];
            sx[idx] = x[dim_aligned * i + idx];
    }
    __syncthreads();
    while (j < num_vec)
    {
        float sum = 0;
        for (int d = 0; d < dim; d++)
        {
            //float diff = xi[d] - x[dim_aligned * j + d];
            float diff = sx[d] - xT[num_vec_aligned * d + j];
            //float diff = x[dim_aligned * i + d] - x[dim_aligned * j + d];
            //float diff = xT[num_vec_aligned * d + i] - xT[num_vec_aligned * d + j];
            sum += diff * diff;
        }
        K[num_vec_aligned * last + j] = expf(-gamma * sum);
        j += gridDim.x * blockDim.x;
    }
}

__global__ void kernelCheckCache_(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * x, const float * xT, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int lastPtrIdx)
{
    extern __shared__ int2 spriority[];
    int i = *i_ptr;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (KCacheRemapIdx[i] >= 0)
    {
        if (j == 0)
            KCacheRowPriority[KCacheRemapIdx[i]] = d_cacheUpdateCnt;  // refresh priority
        return;  //item already in cache
    }
    //int last = (KCacheRowIdx[cache_rows + lastPtrIdx] + 1) % cache_rows;
    int2 minpriority = make_int2(INT_MAX, 0);
    for (int k = 0; k < cache_rows; k += blockDim.x)
    {
        int idx = k + threadIdx.x;
        if (idx < cache_rows)
        {
            int v = KCacheRowPriority[idx];
            if (v < minpriority.x)
                minpriority = make_int2(v, idx);
        }
    }
    spriority[threadIdx.x] = minpriority;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (spriority[threadIdx.x + s].x < spriority[threadIdx.x].x)
                spriority[threadIdx.x] = spriority[threadIdx.x + s];
        }
        __syncthreads();
    }
    int last = spriority[0].y;
    if (j == 0)
    {
        //KCacheRowIdx[cache_rows + (1 - lastPtrIdx)] = last;
        int del_i = KCacheRowIdx[last];
        if (del_i >= 0)
            KCacheRemapIdx[del_i] = -1;  //cache row for vector [del_i] will be overwritten, remove it from RemapIdx array
        //set correct indices
        KCacheRemapIdx[i] = last;
        KCacheRowIdx[last] = i;
        KCacheRowPriority[last] = ++d_cacheUpdateCnt;
    }

    //calculate cache matrix row [last], original index is [i]
    float * sx = (float *)spriority;
    for (int idxshift = 0; idxshift < dim; idxshift += blockDim.x)
    {
        int idx = idxshift + threadIdx.x;
        if (idx < dim)
            //xi[idx] = xT[num_vec_aligned * idx + i];
            sx[idx] = x[dim_aligned * i + idx];
    }
    __syncthreads();
    while (j < num_vec)
    {
        float sum = 0;
        for (int d = 0; d < dim; d++)
        {
            //float diff = xi[d] - x[dim_aligned * j + d];
            float diff = sx[d] - xT[num_vec_aligned * d + j];
            //float diff = x[dim_aligned * i + d] - x[dim_aligned * j + d];
            //float diff = xT[num_vec_aligned * d + i] - xT[num_vec_aligned * d + j];
            sum += diff * diff;
        }
        K[num_vec_aligned * last + j] = expf(-gamma * sum);
        j += gridDim.x * blockDim.x;
    }
}

void OrcusSvmTrain(float * alpha, float * rho, const float * x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps)
{
    float *d_alpha = nullptr,
        *d_x = nullptr,
        *d_xT = nullptr,
        *d_y = nullptr,
        *d_g = nullptr,
        *d_K = nullptr,
        *d_KDiag = nullptr,
        *d_reduceval = nullptr,
        *d_reduceval2 = nullptr;
    int *d_reduceidx = nullptr,
        *d_reduceidx2 = nullptr;
    float *d_lambda = nullptr;
    int *d_workingset = nullptr,
        *d_KCacheRemapIdx = nullptr,
        *d_KCacheRowIdx = nullptr,  // items at index [cache_rows] and [cache_rows+1] are indices of last inserted item
        *d_KCacheRowPriority = nullptr;  // the higher the priority is, the later was the item added

    size_t reduce_block_size = 256;
    size_t reduce_buff_size = rounduptomult(num_vec, reduce_block_size);
    size_t ones_size = std::max(num_vec_aligned, dim_aligned);
    size_t cache_size_mb = 2000;
    size_t cache_rows = cache_size_mb * 1024 * 1024 / (num_vec_aligned * sizeof(float));
    cache_rows = std::min(cache_rows, num_vec);

    std::cout << "Cache size: " << cache_rows << " rows (" << (100.f * cache_rows / (float)num_vec) << " % of data set)" << std::endl;

    cublasHandle_t cublas;
    assert_cublas(cublasCreate(&cublas));

    assert_cuda(cudaMalloc(&d_alpha, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_x, num_vec_aligned * dim_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_xT, num_vec_aligned * dim_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_y, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_g, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceval, reduce_buff_size * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceidx, reduce_buff_size * sizeof(int)));
    assert_cuda(cudaMalloc(&d_reduceval2, reduce_buff_size / reduce_block_size * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceidx2, reduce_buff_size / reduce_block_size * sizeof(int)));
    assert_cuda(cudaMalloc(&d_lambda, sizeof(float)));
    assert_cuda(cudaMalloc(&d_workingset, 2 * sizeof(int)));
    assert_cuda(cudaMalloc(&d_KCacheRemapIdx, num_vec * sizeof(int)));
    assert_cuda(cudaMalloc(&d_KCacheRowIdx, (cache_rows + 2) * sizeof(int)));  //last 2 items are indices of last cache row
    assert_cuda(cudaMalloc(&d_KCacheRowPriority, cache_rows * sizeof(int)));
    assert_cuda(cudaMalloc(&d_KDiag, num_vec * sizeof(float)));
    assert_cuda(cudaMalloc(&d_K, cache_rows * num_vec_aligned * sizeof(float)));

    assert_cuda(cudaMemset(d_alpha, 0, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMemcpy(d_x, x, num_vec_aligned * dim_aligned * sizeof(float), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(d_y, y, num_vec_aligned * sizeof(float), cudaMemcpyHostToDevice));

    float a = 1, b = 0;
    assert_cublas(cublasSgeam(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_vec, dim, &a, d_x, dim_aligned, &b, d_x, num_vec_aligned, d_xT, num_vec_aligned));

    //export_cuda_buffer(d_x, dim_aligned, num_vec_aligned, sizeof(float), "x.dat");

    //dim3 dimBlock(256);
    //dim3 dimGrid(getgriddim(num_vec_aligned, (size_t)dimBlock.x));
    //kernelInitg<<<dimGrid, dimBlock>>>(d_g, num_vec_aligned);
    memsetCuda<float>(d_g, 1, num_vec_aligned);
    memsetCuda<int>(d_KCacheRemapIdx, -1, num_vec);
    memsetCuda<int>(d_KCacheRowIdx, -1, cache_rows + 2);
    memsetCuda<int>(d_KCacheRowPriority, -1, cache_rows);
    int cacheUpdateCnt = 0;
    assert_cuda(cudaMemcpyToSymbol(d_cacheUpdateCnt, &cacheUpdateCnt, sizeof(int), 0));

    //export_cuda_buffer(d_g, num_vec_aligned, 1, sizeof(float), "g.dat");

    //cudaEvent_t evstart, evend;
    //cudaEventCreate(&evstart);
    //cudaEventCreate(&evend);
    //cudaEventRecord(evstart);

    computeKDiag(d_KDiag, num_vec);
    //computeK(d_K, d_x, gamma, num_vec, num_vec_aligned, dim, dim_aligned, cublas);

    //cudaEventRecord(evend);
    //assert_cuda(cudaDeviceSynchronize());
    //float t;
    //cudaEventElapsedTime(&t, evstart, evend);
    //std::cout << "Kernel time: " << t << std::endl;
    //cudaEventDestroy(evstart);
    //cudaEventDestroy(evend);

    //export_cuda_buffer(d_K, num_vec_aligned, num_vec_aligned, sizeof(float), "K.dat");

    int cacheLastPtrIdx = 0;
    dim3 dimBlock(reduce_block_size);
    dim3 dimGrid(getgriddim(num_vec_aligned, (size_t)dimBlock.x));
    dim3 dimBlockCache(256);
    dim3 dimGridCache(getgriddim(num_vec_aligned, (size_t)dimBlockCache.x));
    size_t kernelCheckCacheSMSize = std::max(dim * sizeof(float), dimBlockCache.x * sizeof(int2));
    for (int iter = 0;; iter++)
    {
        kernelSelectI<<<dimGrid, dimBlock>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec);
        //export_cuda_buffer(d_reduceval, 1, reduce_buff_size, sizeof(float), "reduceval.dat");
        //export_cuda_buffer(d_reduceidx, 1, reduce_buff_size, sizeof(int), "reduceidx.dat");
        reduceMaxIdx(d_reduceval, d_reduceidx, d_reduceval2, d_reduceidx2, num_vec_aligned, reduce_block_size);
        assert_cuda(cudaMemcpy(d_workingset, d_reduceidx, sizeof(int), cudaMemcpyDeviceToDevice));

        //int * KCacheRowIdx = new int[cache_rows + 2];
        //assert_cuda(cudaMemcpy(KCacheRowIdx, d_KCacheRowIdx, (cache_rows + 2) * sizeof(int), cudaMemcpyDeviceToHost));
        //std::cout << "KCacheRowIdx: ";
        //for (int k = 0; k < cache_rows + 2; k++)
        //    std::cout << KCacheRowIdx[k] << ", ";
        //std::cout << std::endl;
        //delete[] KCacheRowIdx;

        //check if I is cached
        kernelCheckCache<<<dimGridCache, dimBlockCache, kernelCheckCacheSMSize>>>(d_workingset, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_x, d_xT, gamma, num_vec, num_vec_aligned, dim, dim_aligned, cacheLastPtrIdx);
        cacheLastPtrIdx = 1 - cacheLastPtrIdx;

        //int * KCacheRowIdx = new int[cache_rows + 2];
        //assert_cuda(cudaMemcpy(KCacheRowIdx, d_KCacheRowIdx, (cache_rows + 2) * sizeof(int), cudaMemcpyDeviceToHost));
        //std::cout << "KCacheRowIdx: ";
        //for (int k = 0; k < cache_rows + 2; k++)
        //    std::cout << KCacheRowIdx[k] << ", ";
        //std::cout << std::endl;
        //delete[] KCacheRowIdx;

        //kernelSelectJ1<<<dimGrid, dimBlock>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec);
        kernelSelectJCached<<<dimGrid, dimBlock>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec, num_vec_aligned, d_workingset, d_K, d_KDiag, d_KCacheRemapIdx);
        //export_cuda_buffer(d_reduceval, 1, reduce_buff_size, sizeof(float), "reduceval.dat");
        //export_cuda_buffer(d_reduceidx, 1, reduce_buff_size, sizeof(int), "reduceidx.dat");
        reduceMaxIdx(d_reduceval, d_reduceidx, d_reduceval2, d_reduceidx2, num_vec_aligned, reduce_block_size);
        assert_cuda(cudaMemcpy(d_workingset + 1, d_reduceidx, sizeof(int), cudaMemcpyDeviceToDevice));

        //check if J is cached
        kernelCheckCache<<<dimGridCache, dimBlockCache, kernelCheckCacheSMSize>>>(d_workingset + 1, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_x, d_xT, gamma, num_vec, num_vec_aligned, dim, dim_aligned, cacheLastPtrIdx);
        cacheLastPtrIdx = 1 - cacheLastPtrIdx;
        //workaround if caching J deleted I out of cache
        kernelCheckCache<<<dimGridCache, dimBlockCache, kernelCheckCacheSMSize>>>(d_workingset, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_x, d_xT, gamma, num_vec, num_vec_aligned, dim, dim_aligned, cacheLastPtrIdx);
        cacheLastPtrIdx = 1 - cacheLastPtrIdx;

        //int * KCacheRowIdx = new int[cache_rows + 2];
        //assert_cuda(cudaMemcpy(KCacheRowIdx, d_KCacheRowIdx, (cache_rows + 2) * sizeof(int), cudaMemcpyDeviceToHost));
        //std::cout << "KCacheRowIdx: ";
        //for (int k = 0; k < cache_rows + 2; k++)
        //    std::cout << KCacheRowIdx[k] << ", ";
        //std::cout << std::endl;
        //delete[] KCacheRowIdx;

        if (iter > 0 && iter % 1000 == 0)
        {
            int ws[2];
            float yi, yj, gi, gj;
            assert_cuda(cudaMemcpy(&ws, d_workingset, 2 * sizeof(int), cudaMemcpyDeviceToHost));
            assert_cuda(cudaMemcpy(&yi, d_y + ws[0], sizeof(float), cudaMemcpyDeviceToHost));
            assert_cuda(cudaMemcpy(&yj, d_y + ws[1], sizeof(float), cudaMemcpyDeviceToHost));
            assert_cuda(cudaMemcpy(&gi, d_g + ws[0], sizeof(float), cudaMemcpyDeviceToHost));
            assert_cuda(cudaMemcpy(&gj, d_g + ws[1], sizeof(float), cudaMemcpyDeviceToHost));
            float diff = yi * gi - yj * gj;
            std::cout << "Iter " << iter << ": " << diff << " [" << ws[0] << "," << ws[1] << "]" <<std::endl;
            if (diff < eps)
            {
                *rho = -(yi * gi + yj * gj) / 2;
                std::cout << "Optimality reached, stopping loop. rho = " << *rho << std::endl;
                break;
            }
        }

        //kernelUpdateAlphaAndLambda<<<1, 1>>>(d_alpha, d_lambda, d_y, d_g, d_K, C, d_workingset, num_vec, num_vec_aligned);
        //kernelUpdateg<<<dimGrid, dimBlock>>>(d_g, d_lambda, d_y, d_K, d_workingset, num_vec, num_vec_aligned);
        kernelUpdateAlphaAndLambdaCached<<<1, 1>>>(d_alpha, d_lambda, d_y, d_g, d_K, C, d_workingset, num_vec, num_vec_aligned, d_KDiag, d_KCacheRemapIdx);
        kernelUpdategCached<<<dimGrid, dimBlock>>>(d_g, d_lambda, d_y, d_K, d_workingset, num_vec, num_vec_aligned, d_KCacheRemapIdx);

        //float lambda;
        //int ws[2];
        //assert_cuda(cudaMemcpy(&lambda, d_lambda, sizeof(float), cudaMemcpyDeviceToHost));
        //assert_cuda(cudaMemcpy(&ws, d_workingset, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        //std::cout << "i: " << ws[0] << ", j: " << ws[1] << ", lambda: " << lambda << std::endl;
    }

    assert_cuda(cudaMemcpyFromSymbol(&cacheUpdateCnt, d_cacheUpdateCnt, sizeof(int), 0));
    std::cout << "Cache row updates: " << cacheUpdateCnt << std::endl;

    assert_cuda(cudaMemcpy(alpha, d_alpha, num_vec * sizeof(float), cudaMemcpyDeviceToHost));

    assert_cuda(cudaFree(d_K));
    assert_cuda(cudaFree(d_KDiag));
    assert_cuda(cudaFree(d_KCacheRemapIdx));
    assert_cuda(cudaFree(d_KCacheRowIdx));
    assert_cuda(cudaFree(d_KCacheRowPriority));
    assert_cuda(cudaFree(d_alpha));
    assert_cuda(cudaFree(d_x));
    assert_cuda(cudaFree(d_xT));
    assert_cuda(cudaFree(d_y));
    assert_cuda(cudaFree(d_g));
    assert_cuda(cudaFree(d_reduceval));
    assert_cuda(cudaFree(d_reduceidx));
    assert_cuda(cudaFree(d_reduceval2));
    assert_cuda(cudaFree(d_reduceidx2));
    assert_cuda(cudaFree(d_lambda));
    assert_cublas(cublasDestroy(cublas));
}