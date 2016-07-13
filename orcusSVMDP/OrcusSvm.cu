#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cassert>
#include "OrcusSvm.h"
#include "../cudaerror.h"
#include "../debug.h"

#if __cplusplus <= 199711L
#define nullptr NULL
#endif

#define assert_cuda_dev(call) \
{ \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        const char * s = cudaGetErrorString(e); \
        printf("%s:%d: %s\n", __FILE__, __LINE__, s); \
        assert(0); \
    } \
}

extern int g_cache_size;

struct csr {
	unsigned int nnz;
	unsigned int numRows;
	unsigned int numCols;
	float *values;
	unsigned int *colInd;
	unsigned int *rowOffsets;
};

struct csr_gpu {
	unsigned int nnz;
	unsigned int numRows;
	unsigned int numCols;
	float *values;
	unsigned int *colInd;
	unsigned int *rowOffsets;
    unsigned int *rowLen;
};

template<typename T>
__host__ __device__ T getgriddim(T totallen, T blockdim)
{
    return (totallen + blockdim - (T)1) / blockdim;
}

template<typename T>
T rounduptomult(T x, T m)
{
    return ((x + m - (T)1) / m) * m;
}

#define DENSE_TILE_SIZE 16

namespace OrcusSVMDP
{
//init this to 0
__device__ int d_shrunkSize;
//init this to 0
__device__ int d_nonshrunkSize;
__device__ float2 d_shrinkMaxF;
__device__ int d_alphaStatusChange[2];

__device__ int d_cacheUpdateCnt;
//contains changes to KCacheRemapIdx buffer, which should be written after kernelCheckCache ends
//each change to buffer is contained in int2 variable (x,y) such that
//KCacheRemapIdx[x] = y
//pair at index [2] is for KCacheRowPriority
__device__ int2 d_KCacheChanges[3];

__device__ int d_cacheRow;
}

using namespace OrcusSVMDP;

template<typename T>
__device__ void swap_dev(T & a, T & b)
{
    T tmp = a;
    a = b;
    b = tmp;
}

template<typename T>
__global__ static void kernelMemset(T * mem, T v, int n)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    while (k < n)
    {
        mem[k] = v;
        k += gridDim.x * blockDim.x;
    }
}

template<typename T>
static void memsetCuda(T * d_mem, T v, int n)
{
    dim3 dimBlock(256);
    dim3 dimGrid(std::min(2048, getgriddim<int>(n, dimBlock.x)));
    kernelMemset<T><<<dimGrid, dimBlock>>>(d_mem, v, n);
}

__global__ static void kernelCalcRowLen(unsigned int * rowLen, const unsigned int * rowOffsets, int numRows)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < numRows)
    {
        rowLen[k] = rowOffsets[k + 1] - rowOffsets[k];
    }
}

__global__ static void kernelPow2(float * x2, const float * x, int w, int h, int pitch)
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

__global__ static void kernelPow2SumSparse(csr_gpu x, float * x2)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    while (k < x.numRows)
    {
        float sum = 0;
        int end = x.rowOffsets[k + 1];
        for (int i = x.rowOffsets[k]; i < end; i++)
        {
            float v = x.values[i];
            sum += v * v;
        }
        x2[k] = sum;

        k += gridDim.x * blockDim.x;
    }
}

static void computeX2Dense(const float * d_x, float * d_x2sum, int num_vec, int num_vec_aligned, int dim, int dim_aligned, cublasHandle_t cublas)
{
    float *d_x2 = nullptr,
        *d_ones = nullptr;
    assert_cuda(cudaMalloc(&d_x2, num_vec_aligned * dim_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_ones, dim_aligned * sizeof(float)));
    memsetCuda<float>(d_ones, 1, dim_aligned);

    dim3 dimBlock(16, 16);
    dim3 dimGrid(getgriddim(dim, (int)dimBlock.x), getgriddim(num_vec, (int)dimBlock.y));
    kernelPow2<<<dimGrid, dimBlock>>>(d_x2, d_x, dim, num_vec, dim_aligned);
    float a = 1,
        b = 0;
    assert_cublas(cublasSgemv(cublas, CUBLAS_OP_T, dim, num_vec, &a, d_x2, dim_aligned, d_ones, 1, &b, d_x2sum, 1));

    assert_cuda(cudaFree(d_x2));
}

static void computeX2Sparse(csr_gpu & x, float * d_x2)
{
    dim3 dimBlock(256);
    dim3 dimGrid(std::min(256, getgriddim<int>(x.numRows, dimBlock.x)));
    kernelPow2SumSparse<<<dimGrid, dimBlock>>>(x, d_x2);
}

static void computeKDiag(float * d_KDiag, int num_vec)
{
    //K[i,i] is always 1 for RBF kernel, let's just use memset here
    memsetCuda<float>(d_KDiag, 1, num_vec);
}

__global__ static void kernelSelectI(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    extern __shared__ float sval[];
    int * sidx = (int *)(sval + blockDim.x);

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ == 1 && a_ < C) || (y_ == -1 && a_ > 0))
            v = y[k] * g[k];
        else
            v = -FLT_MAX;
        if (v > max_val)
        {
            max_val = v;
            max_idx = k;
        }
    }

    sval[threadIdx.x] = max_val;
    sidx[threadIdx.x] = max_idx;
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
        valbuf[blockIdx.x] = sval[0];
        idxbuf[blockIdx.x] = sidx[0];
    }
}

//first order search
__global__ static void kernelSelectJ1(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    extern __shared__ float sval[];
    int * sidx = (int *)(sval + blockDim.x);;

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C))
            v = -y[k] * g[k]; //return negative, so we can use reducemax
        else
            v = -FLT_MAX;
        if (v > max_val)
        {
            max_val = v;
            max_idx = k;
        }
    }

    sval[threadIdx.x] = max_val;
    sidx[threadIdx.x] = max_idx;
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
        valbuf[blockIdx.x] = sval[0];
        idxbuf[blockIdx.x] = sidx[0];
    }
}

//second order search with cached K
__global__ static void kernelSelectJCached(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec, int num_vec_aligned, const int * i_ptr, const float * K, const float * KDiag, const int * KCacheRemapIdx)
{
    extern __shared__ float sval[];
    int * sidx = (int *)(sval + blockDim.x);

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        int i = *i_ptr;
        int cache_row = KCacheRemapIdx[i];
        float val;
        float y_ = y[k];
        float a_ = alpha[k];
        float th = y[i] * g[i];
        if (((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C)) && th > y[k] * g[k])
        {
            float den = KDiag[i] + KDiag[k] - 2 * K[(size_t)num_vec_aligned * cache_row + k];
            float v = th - y[k] * g[k];
            val = v * v / den;
        }
        else
            val = -FLT_MAX;
        if (val > max_val)
        {
            max_val = val;
            max_idx = k;
        };
    }

    sval[threadIdx.x] = max_val;
    sidx[threadIdx.x] = max_idx;
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
        valbuf[blockIdx.x] = sval[0];
        idxbuf[blockIdx.x] = sidx[0];
    }
}

//assume grid size (1)
__global__ static void kernelReduceMaxIdx(float * val, int * idx, int * idx_out, int len)
{
    extern __shared__ float sval[];
    int * sidx = (int *)(sval + blockDim.x);

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (int i = threadIdx.x; i < len; i += blockDim.x)
    {
        float v = val[i];
        if (v > max_val)
        {
            max_val = v;
            max_idx = idx[i];
        }
    }

    sval[threadIdx.x] = max_val;
    sidx[threadIdx.x] = max_idx;
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
        idx_out[0] = sidx[0];
}

__device__ __host__ static void reduceMaxIdx(float * d_val, int * d_idx, int * d_result, int len, int reduce_block_size)
{
    //dim3 dimBlock(reduce_block_size);
    //dim3 dimGrid(std::min(reduce_block_size, getgriddim(len, (int)dimBlock.x)));
    //kernelReduceMaxIdx<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val, d_idx, d_val2, d_idx2, len);
    //len = dimGrid.x;
    //dimGrid.x = 1;
    //kernelReduceMaxIdx<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val2, d_idx2, d_val, d_idx, len);
    dim3 dimBlock(reduce_block_size);
    dim3 dimGrid(1);
    kernelReduceMaxIdx<<<dimGrid, dimBlock, dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int)>>>(d_val, d_idx, d_result, getgriddim(len, (int)dimBlock.x));
}

__global__ static void kernelUpdateg(float * g, const float * lambda, const float * y, const float * K, const int * ws, int num_vec, int num_vec_aligned)
{
    int i = ws[0];
    int j = ws[1];
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_vec)
    {
        g[k] += *lambda * y[k] * (K[(size_t)num_vec_aligned * j + k] - K[(size_t)num_vec_aligned * i + k]);
    }
}

__global__ static void kernelUpdategCached(float * g, const float * lambda, const float * y, const float * K, const int * ws, int num_vec, int num_vec_aligned, const int * KCacheRemapIdx)
{
    int i = ws[0];
    int j = ws[1];
    int i_cache_row = KCacheRemapIdx[i];
    int j_cache_row = KCacheRemapIdx[j];
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < num_vec)
    {
        g[k] += *lambda * y[k] * (K[(size_t)num_vec_aligned * j_cache_row + k] - K[(size_t)num_vec_aligned * i_cache_row + k]);
    }
}

__global__ static void kernelUpdateAlphaAndLambda(float * alpha, float * lambda, const float * y, const float * g, const float * K, float C, const int * ws, int num_vec, int num_vec_aligned)
{
    int i = ws[0];
    int j = ws[1];
    float l1 = y[i] > 0 ? C - alpha[i] : alpha[i];
    float l2 = y[j] > 0 ? alpha[j] : C - alpha[j];
    float l3 = (y[i] * g[i] - y[j] * g[j]) / (K[(size_t)num_vec_aligned * i + i] + K[(size_t)num_vec_aligned * j + j] - 2 * K[(size_t)num_vec_aligned * i + j]);
    float l = min(l1, min(l2, l3));

    *lambda = l;
    alpha[i] += l * y[i];
    alpha[j] -= l * y[j];
}

__global__ static void kernelUpdateAlphaAndLambdaCached(float * alpha, float * lambda, const float * y, const float * g, const float * K, float C, const int * ws, int num_vec_aligned, const float * KDiag, const int * KCacheRemapIdx)
{
    int i = ws[0];
    int j = ws[1];
    int cache_row = KCacheRemapIdx[i];
    float l1 = y[i] > 0 ? C - alpha[i] : alpha[i];
    float l2 = y[j] > 0 ? alpha[j] : C - alpha[j];
    float l3 = (y[i] * g[i] - y[j] * g[j]) / (KDiag[i] + KDiag[j] - 2 * K[(size_t)num_vec_aligned * cache_row + j]);
    float l = min(l1, min(l2, l3));

    *lambda = l;
    alpha[i] += l * y[i];
    alpha[j] -= l * y[j];
}

__global__ static void kernelCheckCacheFinalize(int * KCacheRemapIdx, int * KCacheRowPriority)
{
    int2 c;
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        c = d_KCacheChanges[i];
        if (c.x >= 0)
        {
            KCacheRemapIdx[c.x] = c.y;
            d_KCacheChanges[i].x = -1;
        }
    }
    c = d_KCacheChanges[2];
    if (c.x >= 0)
    {
        KCacheRowPriority[c.x] = c.y;
        d_KCacheChanges[2].x = -1;
    }
}

__global__ static void kernelCheckCachePriority(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * x, const float * xT, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int last = d_cacheRow;
    if (last < 0)
        return;
    extern __shared__ int2 spriority[];
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = *i_ptr;

    //calculate cache matrix row [last], original index is [i]
    float * sx = (float *)spriority;
    for (int idxshift = 0; idxshift < dim; idxshift += blockDim.x)
    {
        int idx = idxshift + threadIdx.x;
        if (idx < dim)
            sx[idx] = x[dim_aligned * i + idx];
    }
    __syncthreads();
    while (j < num_vec)
    {
        float sum = 0;
        for (int d = 0; d < dim; d++)
        {
            float diff = sx[d] - xT[num_vec_aligned * d + j];
            sum += diff * diff;
        }
        K[(size_t)num_vec_aligned * last + j] = expf(-gamma * sum);
        j += gridDim.x * blockDim.x;
    }
}

__global__ static void kernelCheckCachePriorityV2(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * x, const float * x2, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int last = d_cacheRow;
    if (last < 0)
        return;
    __shared__ float shsum[DENSE_TILE_SIZE][DENSE_TILE_SIZE+1];
    int block = blockDim.x * blockIdx.x;
    int i = *i_ptr;

    //calculate cache matrix row [last], original index is [i]
    while (block < num_vec)
    {
        int j = block + threadIdx.y;
        int jout = block + threadIdx.x;
        float sum = 0;
        if (j < num_vec)
        {
            for (int d = threadIdx.x; d < dim; d += DENSE_TILE_SIZE)
            {
                sum += x[dim_aligned * i + d] * x[dim_aligned * j + d];
            }
        }
        shsum[threadIdx.y][threadIdx.x] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                shsum[threadIdx.y][threadIdx.x] += shsum[threadIdx.y][threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.y == 0 && jout < num_vec)
        {
            sum = x2[i] + x2[jout] - 2 * shsum[threadIdx.x][0];
            K[(size_t)num_vec_aligned * last + jout] = expf(-gamma * sum);
        }
        __syncthreads();
        block += gridDim.x * blockDim.x;
    }
}

__global__ static void kernelMakeDenseVec(const int * i_ptr, const int * KCacheRemapIdx, csr_gpu x, float * vec)
{
    int i = *i_ptr;
    if (KCacheRemapIdx[i] >= 0)  //if [i] is already in cache, exit. we won't need any dense vector
        return;
    int j = x.rowOffsets[i] + blockDim.x * blockIdx.x + threadIdx.x;
    while (j < x.rowOffsets[i + 1])
    {
        vec[x.colInd[j]] = x.values[j];
        j += gridDim.x * blockDim.x;
    }
}

__global__ static void kernelFindCacheRow(const int * i_ptr, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows)
{
    extern __shared__ int2 spriority[];
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = *i_ptr;
    if (KCacheRemapIdx[i] >= 0)
    {
        if (j == 0)
        {
            KCacheRowPriority[KCacheRemapIdx[i]] = d_cacheUpdateCnt;  // refresh priority
            d_cacheRow = -1;
        }
        return;  //item already in cache
    }
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
    if (j == 0)
    {
        int last = spriority[0].y;
        int del_i = KCacheRowIdx[last];
        if (del_i >= 0)
            d_KCacheChanges[1] = make_int2(del_i, -1);  //cache row for vector [del_i] will be overwritten, remove it from RemapIdx array
        //set correct indices
        d_KCacheChanges[0] = make_int2(i, last);
        KCacheRowIdx[last] = i;
        d_KCacheChanges[2] = make_int2(last, ++d_cacheUpdateCnt);
        d_cacheRow = last;
    }
}

__global__ static void kernelCheckCacheSparsePriority(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * vec, const float * x2, csr_gpu x, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int last = d_cacheRow;
    if (last < 0)
        return;
    extern __shared__ int2 spriority[];
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = *i_ptr;

    //calculate cache matrix row [last], original index is [i]
    while (j < num_vec)
    {
        float sum = 0;
        int end = x.rowOffsets[j + 1];
        for (int d = x.rowOffsets[j]; d < end; d++)
        {
            sum += vec[x.colInd[d]] * x.values[d];
        }
        sum = x2[i] + x2[j] - 2 * sum;

        K[(size_t)num_vec_aligned * last + j] = expf(-gamma * sum);
        j += gridDim.x * blockDim.x;
    }
}

#define SPARSE_TILE_SIZE 16

__global__ static void kernelCheckCacheSparsePriorityV2(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * vec, const float * x2, csr_gpu x, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int last = d_cacheRow;
    if (last < 0)
        return;
    __shared__ float shsum[SPARSE_TILE_SIZE][SPARSE_TILE_SIZE+1];
    int block = blockDim.x * blockIdx.x;
    int i = *i_ptr;

    //calculate cache matrix row [last], original index is [i]
    while (block < num_vec)
    {
        int j = block + threadIdx.y;
        int jout = block + threadIdx.x;
        float sum = 0;
        if (j < num_vec)
        {
            //int end = x.rowOffsets[j + 1];
            int end = x.rowOffsets[j] + x.rowLen[j];
            for (int d = x.rowOffsets[j] + threadIdx.x; d < end; d += SPARSE_TILE_SIZE)
            {
                sum += vec[x.colInd[d]] * x.values[d];
            }
        }
        shsum[threadIdx.y][threadIdx.x] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                shsum[threadIdx.y][threadIdx.x] += shsum[threadIdx.y][threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.y == 0 && jout < num_vec)
        {
            sum = x2[i] + x2[jout] - 2 * shsum[threadIdx.x][0];
            K[(size_t)num_vec_aligned * last + jout] = expf(-gamma * sum);
        }
        __syncthreads();
        block += gridDim.x * blockDim.x;
    }
}

//make a GPU deep copy of a CPU csr matrix
static cudaError_t make_gpu_csr(csr_gpu &x_gpu, const csr &x_cpu) {
	x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;

	assert_cuda(cudaMalloc((void **)&(x_gpu.values), x_gpu.nnz * sizeof(float)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.colInd), x_gpu.nnz * sizeof(int)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.rowOffsets), (x_gpu.numRows+1) * sizeof(int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowLen), x_gpu.numRows * sizeof(int)));

	assert_cuda(cudaMemcpy(x_gpu.values, x_cpu.values, x_gpu.nnz * sizeof(float), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.colInd, x_cpu.colInd, x_gpu.nnz * sizeof(int), cudaMemcpyHostToDevice));
	assert_cuda(cudaMemcpy(x_gpu.rowOffsets, x_cpu.rowOffsets, (x_gpu.numRows+1) * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimBlock(256);
    dim3 dimGrid(getgriddim(x_gpu.numRows, dimBlock.x));
    kernelCalcRowLen<<<dimGrid, dimBlock>>>(x_gpu.rowLen, x_gpu.rowOffsets, x_gpu.numRows);

	return cudaSuccess;
} //make_gpu_csr

static cudaError_t cudaCsrFree(csr_gpu &x_gpu) {
	assert_cuda(cudaFree(x_gpu.values));
	assert_cuda(cudaFree(x_gpu.colInd));
	assert_cuda(cudaFree(x_gpu.rowOffsets));
    assert_cuda(cudaFree(x_gpu.rowLen));
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
	x_gpu.rowOffsets = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;

	return cudaSuccess;
} //cudaCsrFree

__host__ __device__ void checkCache(bool sparse, int * d_i, float * d_x, const float * d_x2, const csr_gpu & sparse_data_gpu, float * d_K, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    dim3 dimBlockCache(256);
    dim3 dimGridCache(getgriddim<int>(num_vec, dimBlockCache.x));
    size_t kernelCheckCacheSMSize = dimBlockCache.x * sizeof(int2);
    if (sparse)
    {
        assert_cuda_dev(cudaMemsetAsync(d_denseVec, 0, dim * sizeof(float)));
        dim3 dimBlock(256);
        dim3 dimGrid(min(64, getgriddim<int>(dim, dimBlock.x)));
        kernelMakeDenseVec<<<dimGrid, dimBlock>>>(d_i, d_KCacheRemapIdx, sparse_data_gpu, d_denseVec);
        kernelFindCacheRow<<<1, 256, 256 * sizeof(int2)>>>(d_i, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
        dimBlock = dim3(SPARSE_TILE_SIZE, SPARSE_TILE_SIZE);
        dimGrid = dim3(min(256, getgriddim<int>(num_vec, dimBlock.y)));
        kernelCheckCacheSparsePriorityV2<<<dimGrid, dimBlock>>>(d_i, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_denseVec, d_x2, sparse_data_gpu, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
    else
    {
        kernelCheckCacheSMSize = max(kernelCheckCacheSMSize, dim * sizeof(float));
        kernelFindCacheRow<<<1, 256, 256 * sizeof(int2)>>>(d_i, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
        dim3 dimBlock(DENSE_TILE_SIZE, DENSE_TILE_SIZE);
        dim3 dimGrid(min(256, getgriddim<int>(num_vec, dimBlock.y)));
        kernelCheckCachePriorityV2<<<dimGrid, dimBlock>>>(d_i, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_x, d_x2, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
    kernelCheckCacheFinalize<<<1, 1>>>(d_KCacheRemapIdx, d_KCacheRowPriority);
}

__global__ void kernelIterate(int num_iter, bool sparse, float * d_x, const float * d_x2, const float * d_y, float * d_alpha, float * d_g, float * d_lambda, csr_gpu sparse_data_gpu, float * d_K, const float * d_KDiag, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma, float C, float * d_reduceval, int * d_reduceidx, int * d_workingset, int reduce_block_size)
{
    dim3 dimBlock(reduce_block_size);
    dim3 dimGrid(getgriddim<size_t>(num_vec, dimBlock.x));
    size_t sharedSizeSelect = dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int);
    for (int iter = 0; iter < num_iter; iter++)
    {
        kernelSelectI<<<dimGrid, dimBlock, sharedSizeSelect>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec);
        reduceMaxIdx(d_reduceval, d_reduceidx, d_workingset, num_vec, reduce_block_size);

        //check if I is cached
        checkCache(sparse, d_workingset, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec, num_vec_aligned, dim, dim_aligned, cache_rows, gamma);

        kernelSelectJCached<<<dimGrid, dimBlock, sharedSizeSelect>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec, num_vec_aligned, d_workingset, d_K, d_KDiag, d_KCacheRemapIdx);
        reduceMaxIdx(d_reduceval, d_reduceidx, d_workingset + 1, num_vec, reduce_block_size);

        //check if J is cached
        checkCache(sparse, d_workingset + 1, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec, num_vec_aligned, dim, dim_aligned, cache_rows, gamma);

        kernelUpdateAlphaAndLambdaCached<<<1, 1>>>(d_alpha, d_lambda, d_y, d_g, d_K, C, d_workingset, num_vec_aligned, d_KDiag, d_KCacheRemapIdx);
        kernelUpdategCached<<<dimGrid, dimBlock>>>(d_g, d_lambda, d_y, d_K, d_workingset, num_vec, num_vec_aligned, d_KCacheRemapIdx);
        assert_cuda_dev(cudaDeviceSynchronize());
    }
}

void OrcusSvmTrain(float * alpha, float * rho, bool sparse, const float * x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps)
{
    float *d_alpha = nullptr,
        *d_x = nullptr,
        *d_y = nullptr,
        *d_g = nullptr,
        *d_gBar = nullptr,
        *d_K = nullptr,
        *d_KDiag = nullptr,
        *d_reduceval = nullptr;
    int *d_reduceidx = nullptr;
    //TODO: move lambda and workingset to __device__ variables, no need for dynamic allocation
    float *d_lambda = nullptr;
    int *d_workingset = nullptr,
        *d_KCacheRemapIdx = nullptr,
        *d_KCacheRowIdx = nullptr,  // items at index [cache_rows] and [cache_rows+1] are indices of last inserted item
        *d_KCacheRowPriority = nullptr;  // the higher the priority is, the later was the item added
    float *d_denseVec = nullptr;  //dense vector used to calculate K cache row for sparse data
    float *d_x2 = nullptr;

    bool useShrinking = true;
    size_t reduce_block_size = 256;
    size_t reduce_buff_size = rounduptomult(num_vec, reduce_block_size);
    size_t ones_size = std::max(num_vec_aligned, dim_aligned);
    size_t cache_size_mb = g_cache_size;
    if (cache_size_mb == 0)
    {
        size_t free_mem, total_mem;
        assert_cuda(cudaFree(nullptr));  //force CUDA init
        assert_cuda(cuMemGetInfo(&free_mem, &total_mem));
        cache_size_mb = free_mem / (1024 * 1024) - 200;  //leave 200 MB free
    }
    size_t cache_rows = cache_size_mb * 1024 * 1024 / (num_vec_aligned * sizeof(float));
    cache_rows = std::min(cache_rows, num_vec);

    std::cout << "Training data: " << (sparse ? "sparse" : "dense") << std::endl;
    std::cout << "Data size: " << num_vec << "\nDimension: " << dim << std::endl;
    std::cout << "Cache size: " << cache_rows << " rows (" << (100.f * cache_rows / (float)num_vec) << " % of data set)" << std::endl;

    cublasHandle_t cublas;
    assert_cublas(cublasCreate(&cublas));

    const csr * sparse_data = (const csr *)x;
    csr_gpu sparse_data_gpu;
    assert_cuda(cudaMalloc(&d_x2, num_vec * sizeof(float)));
    if (sparse)
    {
        assert_cuda(make_gpu_csr(sparse_data_gpu, *sparse_data));
        assert_cuda(cudaMalloc(&d_denseVec, dim * sizeof(float)));
        std::cout << "Precalculating X2" << std::endl;
        computeX2Sparse(sparse_data_gpu, d_x2);
    }
    else
    {
        assert_cuda(cudaMalloc(&d_x, num_vec_aligned * dim_aligned * sizeof(float)));
    }
    assert_cuda(cudaMalloc(&d_alpha, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_y, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_g, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_gBar, num_vec_aligned * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceval, reduce_buff_size / reduce_block_size * sizeof(float)));
    assert_cuda(cudaMalloc(&d_reduceidx, reduce_buff_size / reduce_block_size * sizeof(int)));
    assert_cuda(cudaMalloc(&d_lambda, sizeof(float)));
    assert_cuda(cudaMalloc(&d_workingset, 2 * sizeof(int)));
    assert_cuda(cudaMalloc(&d_KCacheRemapIdx, num_vec * sizeof(int)));
    assert_cuda(cudaMalloc(&d_KCacheRowIdx, (cache_rows + 2) * sizeof(int)));  //last 2 items are indices of last cache row
    assert_cuda(cudaMalloc(&d_KCacheRowPriority, cache_rows * sizeof(int)));
    assert_cuda(cudaMalloc(&d_KDiag, num_vec * sizeof(float)));
    assert_cuda(cudaMalloc(&d_K, cache_rows * num_vec_aligned * sizeof(float)));
    std::cout << "Cache size: " << (cache_rows * num_vec_aligned) << " floats\n";

    assert_cuda(cudaMemset(d_alpha, 0, num_vec_aligned * sizeof(float)));
    if (!sparse)
    {
        assert_cuda(cudaMemcpy(d_x, x, num_vec_aligned * dim_aligned * sizeof(float), cudaMemcpyHostToDevice));
        std::cout << "Precalculating X2" << std::endl;
        computeX2Dense(d_x, d_x2, num_vec, num_vec_aligned, dim, dim_aligned, cublas);
    }
    assert_cuda(cudaMemcpy(d_y, y, num_vec_aligned * sizeof(float), cudaMemcpyHostToDevice));

    int KCacheChanges[6];
    for (int i = 0; i < sizeof(KCacheChanges) / sizeof(*KCacheChanges); i++)
        KCacheChanges[i] = -1;
    assert_cuda(cudaMemcpyToSymbol(d_KCacheChanges, KCacheChanges, sizeof(KCacheChanges), 0));

    memsetCuda<float>(d_g, 1, num_vec_aligned);
    assert_cuda(cudaMemset(d_gBar, 0, num_vec_aligned * sizeof(float)));
    memsetCuda<int>(d_KCacheRemapIdx, -1, num_vec);
    memsetCuda<int>(d_KCacheRowIdx, -1, cache_rows + 2);
    memsetCuda<int>(d_KCacheRowPriority, -1, cache_rows);
    int cacheUpdateCnt = 0;
    assert_cuda(cudaMemcpyToSymbol(d_cacheUpdateCnt, &cacheUpdateCnt, sizeof(int), 0));

    std::cout << "Precalculating KDiag" << std::endl;
    computeKDiag(d_KDiag, num_vec);

    size_t num_vec_shrunk = num_vec;

    dim3 dimBlock(reduce_block_size);
    dim3 dimGrid(getgriddim(num_vec, (size_t)dimBlock.x));
    size_t sharedSizeSelect = dimBlock.x * sizeof(float) + dimBlock.x * sizeof(int);
    std::cout << "Starting iterations" << std::endl;
    int iter_step = 1000;
    for (int iter = 0;; iter += iter_step)
    {
        kernelIterate<<<1, 1>>>(iter_step, sparse, d_x, d_x2, d_y, d_alpha, d_g, d_lambda, sparse_data_gpu, d_K, d_KDiag, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma, C, d_reduceval, d_reduceidx, d_workingset, reduce_block_size);

        kernelSelectI<<<dimGrid, dimBlock, sharedSizeSelect>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec_shrunk);
        reduceMaxIdx(d_reduceval, d_reduceidx, d_workingset, num_vec_shrunk, reduce_block_size);

        //check if I is cached
        checkCache(sparse, d_workingset, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma);

        kernelSelectJCached<<<dimGrid, dimBlock, sharedSizeSelect>>>(d_reduceval, d_reduceidx, d_y, d_g, d_alpha, C, num_vec_shrunk, num_vec_aligned, d_workingset, d_K, d_KDiag, d_KCacheRemapIdx);
        reduceMaxIdx(d_reduceval, d_reduceidx, d_workingset + 1, num_vec_shrunk, reduce_block_size);

        int ws[2];
        float yi, yj, gi, gj;
        assert_cuda(cudaMemcpy(&ws, d_workingset, 2 * sizeof(int), cudaMemcpyDeviceToHost));
        assert_cuda(cudaMemcpy(&yi, d_y + ws[0], sizeof(float), cudaMemcpyDeviceToHost));
        assert_cuda(cudaMemcpy(&yj, d_y + ws[1], sizeof(float), cudaMemcpyDeviceToHost));
        assert_cuda(cudaMemcpy(&gi, d_g + ws[0], sizeof(float), cudaMemcpyDeviceToHost));
        assert_cuda(cudaMemcpy(&gj, d_g + ws[1], sizeof(float), cudaMemcpyDeviceToHost));
        float diff = yi * gi - yj * gj;
        std::cout << "Iter " << iter << ": " << diff << " [" << ws[0] << "," << ws[1] << "]" << std::endl;
        if (diff < eps)
        {
            *rho = -(yi * gi + yj * gj) / 2;
            std::cout << "Optimality reached, stopping loop. rho = " << *rho << std::endl;
            break;
        }
    }

    assert_cuda(cudaMemcpyFromSymbol(&cacheUpdateCnt, d_cacheUpdateCnt, sizeof(int), 0));
    std::cout << "Cache row updates: " << cacheUpdateCnt << std::endl;

    assert_cuda(cudaMemcpy(alpha, d_alpha, num_vec * sizeof(float), cudaMemcpyDeviceToHost));

    if (sparse)
    {
        cudaCsrFree(sparse_data_gpu);
        assert_cuda(cudaFree(d_denseVec));
    }
    else
    {
        assert_cuda(cudaFree(d_x));
    }

    assert_cuda(cudaFree(d_x2));
    assert_cuda(cudaFree(d_K));
    assert_cuda(cudaFree(d_KDiag));
    assert_cuda(cudaFree(d_KCacheRemapIdx));
    assert_cuda(cudaFree(d_KCacheRowIdx));
    assert_cuda(cudaFree(d_KCacheRowPriority));
    assert_cuda(cudaFree(d_alpha));
    assert_cuda(cudaFree(d_y));
    assert_cuda(cudaFree(d_g));
    assert_cuda(cudaFree(d_gBar));
    assert_cuda(cudaFree(d_reduceval));
    assert_cuda(cudaFree(d_reduceidx));
    assert_cuda(cudaFree(d_lambda));
    assert_cublas(cublasDestroy(cublas));
}
