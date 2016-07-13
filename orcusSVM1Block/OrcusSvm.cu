#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cfloat>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "OrcusSvm.h"
#include "../cudaerror.h"
#include "../cuda_utils.h"
#include "csr.h"
#include "calc_x2.h"

#if __cplusplus <= 199711L
#define nullptr NULL
#endif

//#define USE_TIMERS
#include "../debug.h"

#define CALC_KLOCAL
#define USE_CUBLAS
//#define USE_DAIFLETCHER
#define MAX_BLOCK_ITER 10000

#define TRANSPOSE_TILE_SIZE 16
#define DENSE_TILE_SIZE 16

#define MAX_WORKING_SET 2048

#define NUM_SORT_BLOCKS 32
#define CALC_KLOCAL_TILE_X 16
#define CALC_KLOCAL_NUM_WARPS 8

//#define WORKING_SET 2048
//#define NUM_NC (WORKING_SET/4)
//#define ELEM_PER_THREAD 2

//#define WORKING_SET 1024
//#define NUM_NC (WORKING_SET/4)
//#define ELEM_PER_THREAD 1

namespace OrcusSVM1B
{
#include "dev_vars.h"
}

using namespace OrcusSVM1B;

#include "check_cache.h"
#include "kernels_select_ws.h"
#include "find_nbestv2.h"
#include "kernels_klocal.h"
#ifdef USE_DAIFLETCHER
#include "kernels_daifletcher.h"
#else
#include "kernels_smo.h"
#endif

extern int g_cache_size;

#define STATIC_MIN(a, b) ((a) < (b) ? (a) : (b))
#define STATIC_MAX(a, b) ((a) > (b) ? (a) : (b))

__global__ static void kernelTranspose(const float * data_in, float * data_out, int width, int height, int ipitch, int opitch)
{
	__shared__ float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1];
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x,
		yIndexO = blockDim.x * blockIdx.x + threadIdx.y;
    for (int offset = blockDim.y * blockIdx.y; offset < height; offset += gridDim.y * blockDim.y)
	{
        int yIndex = offset + threadIdx.y;
        int xIndexO = offset + threadIdx.x;
		if (xIndex < width && yIndex < height)
			tile[threadIdx.y][threadIdx.x] = data_in[ipitch * yIndex + xIndex];
		__syncthreads();
		if (xIndexO < height && yIndexO < width)
			data_out[opitch * yIndexO + xIndexO] = tile[threadIdx.x][threadIdx.y];
		__syncthreads();
	}
}

static void computeKDiag(float * d_KDiag, int num_vec)
{
    //K[i,i] is always 1 for RBF kernel, let's just use memset here
    memsetCuda<float>(d_KDiag, 1, num_vec);
}

template<unsigned int WS>
__global__ static void kernelKtoHLocal(float * K, const float * y, const int * ws)
{
    for (int iy = blockDim.y * blockIdx.y + threadIdx.y; iy < WS; iy += gridDim.y * blockDim.y)
    {
        float yy = y[ws[iy]];
        for (int ix = blockDim.x * blockIdx.x + threadIdx.x; ix < WS; ix += gridDim.x * blockDim.x)
            K[WS * iy + ix] *= yy * y[ws[ix]];
    }
}

template<unsigned int WS>
__global__ static void kernelUpdateG(float * y, float * g, const float * alphadiff, const int * ws, const float * K, const int * KCacheRemapIdx, int num_vec, int num_vec_aligned)
{
    __shared__ float shAdiff[WS];
    __shared__ int shWS[WS];
    __shared__ float shY[WS];
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    for (int j = threadIdx.x; j < WS; j += blockDim.x) 
    {
        shAdiff[j] = alphadiff[j];
        shWS[j] = KCacheRemapIdx[ws[j]];
        shY[j] = y[ws[j]];
        if (blockIdx.x == 0)
        {
            if (shAdiff[j] != 0) //use some tau here
                atomicAdd(d_updateGCnt, 1);
            else
                atomicAdd(d_updateGCnt + 1, 1);
        }
    }
    __syncthreads();

    if (k < num_vec)
    {
#ifdef USE_DAIFLETCHER
        float update = 0;
        for (int i = 0; i < WS; i++)
        {
            float adiff = shAdiff[i];
			if (adiff != 0) 
                update += adiff * /*shY[i] **/ K[(size_t)num_vec_aligned * shWS[i] + k];
        }
        g[k] += update;
#else
        float update = 0;
        for (int i = 0; i < WS; i++)
        {
            float adiff = shAdiff[i];
			if (adiff != 0) 
                update += adiff * K[(size_t)num_vec_aligned * shWS[i] + k];
        }
        g[k] += y[k] * update;
#endif
    }
}

template<unsigned int WS>
__global__ static void kernelUpdateGHLocal(float * gh, const float * g, const float * y, const float * alpha, const int * ws, const float * K)
{
    __shared__ float shAlpha[WS];
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    int ws_ = ws[k];
    shAlpha[k] = alpha[ws_];
    __syncthreads();

    float aux = 0;
    for (int i = 0; i < WS; i++)
        aux += shAlpha[i] * K[WS * i + k];
    gh[k] = g[ws_] * y[ws_] - 1.0f - aux;
}

template<unsigned int WS>
static void train(float * alpha, float * rho, bool sparse, const OrcusSvm1B::Data & x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps)
{
    const int NC = WS / 4;
    const int ELEM_PER_THREAD = STATIC_MAX(WS / 1024, 1);
    std::cout << "Working set: " << WS
        << "\nNC: " << NC
        << "\nElements per thread: " << ELEM_PER_THREAD << std::endl;
    assert(WS <= MAX_WORKING_SET);

    //bool use_ellpack = dim <= 1000;
    bool use_ellpack = false;
    //num_vec = 1000;
    //num_vec_aligned = 1024;
    float *d_alpha = nullptr,
        *d_alphadiff = nullptr,
        *d_x = nullptr,
#ifdef USE_CUBLAS
        *d_xT = nullptr,
#endif
        *d_y = nullptr,
        *d_g = nullptr,
        *d_K = nullptr,
        *d_KLocal = nullptr,
        *d_KDiag = nullptr,
        *d_sortval = nullptr;
    int *d_sortidx = nullptr,
        *d_workingset = nullptr,
        *d_ws_priority = nullptr,
        *d_KCacheRemapIdx = nullptr,
        *d_KCacheRowIdx = nullptr,  // items at index [cache_rows] and [cache_rows+1] are indices of last inserted item
        *d_KCacheRowPriority = nullptr;  // the higher the priority is, the later was the item added
    float *d_denseVec = nullptr;  //dense vector used to calculate K cache row for sparse data
    float *d_x2 = nullptr;
	float *d_sortValues = nullptr;
	int *d_sortIdxs = nullptr;
#ifdef USE_CUBLAS
    float * d_xTile = nullptr;
    float * d_KTile = nullptr;
#endif
#ifdef USE_DAIFLETCHER
    float * d_df_g = nullptr,
          * d_df_gh = nullptr,
          * d_df_y = nullptr,
          * d_df_tempv = nullptr,
          * d_df_d = nullptr,
          * d_df_Ad = nullptr,
          * d_df_t = nullptr,
          * d_df_xplus = nullptr,
          * d_df_tplus = nullptr,
          * d_df_sk = nullptr,
          * d_df_yk = nullptr;
#endif
    csr_gpu sparse_data_gpu;
    ellpack_gpu ellpack_data_gpu;
    jds_gpu jds_data_gpu;
    cublasHandle_t cublas;
    try
    {

        std::cout << "Training data: " << (sparse ? "sparse" : "dense") << std::endl;
        std::cout << "Data size: " << num_vec << "\nDimension: " << dim << std::endl;

        float aux_one = 1,
            aux_zero = 0;
        assert_cuda(cudaMemcpyToSymbol(d_aux_one, &aux_one, sizeof(aux_one)));
        assert_cuda(cudaMemcpyToSymbol(d_aux_zero, &aux_zero, sizeof(aux_zero)));
        
        assert_cublas(cublasCreate(&cublas));
        
        assert_cuda(cudaMalloc(&d_x2, num_vec * sizeof(float)));
        assert_cuda(cudaMalloc(&d_alpha, num_vec_aligned * sizeof(float)));
        assert_cuda(cudaMalloc(&d_alphadiff, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_y, num_vec_aligned * sizeof(float)));
        assert_cuda(cudaMalloc(&d_g, num_vec_aligned * sizeof(float)));
        assert_cuda(cudaMalloc(&d_sortval, num_vec * sizeof(float)));
        assert_cuda(cudaMalloc(&d_sortidx, num_vec * sizeof(int)));
        assert_cuda(cudaMalloc(&d_workingset, WS * sizeof(int)));
        assert_cuda(cudaMalloc(&d_ws_priority, num_vec * sizeof(int)));
        assert_cuda(cudaMalloc(&d_KCacheRemapIdx, num_vec * sizeof(int)));
        //assert_cuda(cudaMalloc(&d_KCacheRowIdx, cache_rows * sizeof(int)));
        //assert_cuda(cudaMalloc(&d_KCacheRowPriority, cache_rows * sizeof(int)));
        assert_cuda(cudaMalloc(&d_KDiag, num_vec * sizeof(float)));
        //assert_cuda(cudaMalloc(&d_K, cache_rows * num_vec_aligned * sizeof(float)));
        assert_cuda(cudaMalloc(&d_KLocal, WS * WS * sizeof(float)));

        //assert_cuda(cudaMalloc(&d_sortValues, NUM_SORT_BLOCKS * WS * sizeof(float)));
        //assert_cuda(cudaMalloc(&d_sortIdxs, NUM_SORT_BLOCKS * WS * sizeof(int)));
        assert_cuda(cudaMalloc(&d_sortValues, NUM_SORT_BLOCKS * NC * 2 * sizeof(float)));
        assert_cuda(cudaMalloc(&d_sortIdxs, NUM_SORT_BLOCKS * NC * 2 * sizeof(int)));

        assert_cuda(cudaMemset(d_alpha, 0, num_vec_aligned * sizeof(float)));
        if (sparse)
        {
            if (use_ellpack)
                makeCudaEllpack(ellpack_data_gpu, *x.sparse);
            else
                makeCudaCsr(sparse_data_gpu, *x.sparse);
            makeCudaJds(jds_data_gpu, *x.sparse);
            assert_cuda(cudaMalloc(&d_denseVec, dim_aligned * WS * sizeof(float)));
            std::cout << "Precalculating X2" << std::endl;
            //if (use_ellpack)
            //    computeX2Sparse(d_x2, ellpack_data_gpu, num_vec);
            //else
            //    computeX2Sparse(d_x2, sparse_data_gpu, num_vec);
            computeX2Sparse(d_x2, jds_data_gpu, num_vec);
        }
        else
        {
            assert_cuda(cudaMalloc(&d_x, num_vec_aligned * dim_aligned * sizeof(float)));
            assert_cuda(cudaMemcpy(d_x, x.dense, num_vec_aligned * dim_aligned * sizeof(float), cudaMemcpyHostToDevice));
            std::cout << "Precalculating X2" << std::endl;
            computeX2Dense(d_x2, d_x, num_vec, num_vec_aligned, dim, dim_aligned);
#ifdef USE_CUBLAS
            assert_cuda(cudaMalloc(&d_xT, dim_aligned * num_vec_aligned * sizeof(*d_xT)));
            assert_cuda(cudaMalloc(&d_xTile, dim_aligned * WS * sizeof(float)));
            assert_cuda(cudaMalloc(&d_KTile, num_vec_aligned * WS * sizeof(float)));
            dim3 dimBlockT(TRANSPOSE_TILE_SIZE, TRANSPOSE_TILE_SIZE);
            kernelTranspose << <dim3(getgriddim<int>(dim, dimBlockT.x), 16), dimBlockT >> >(d_x, d_xT, dim, num_vec, dim_aligned, num_vec_aligned);
#endif
        }
        assert_cuda(cudaMemcpy(d_y, y, num_vec_aligned * sizeof(float), cudaMemcpyHostToDevice));

        size_t cache_size_mb = g_cache_size;
        if (cache_size_mb == 0)  //TODO: move cache size calculation after all allocations
        {
            size_t free_mem, total_mem;
            assert_cuda(cudaFree(nullptr));  //force CUDA init
            assert_cuda(cuMemGetInfo(&free_mem, &total_mem));
            cache_size_mb = (free_mem * 0.8) / (1024 * 1024);
        }
        size_t cache_rows = cache_size_mb * 1024 * 1024 / (num_vec_aligned * sizeof(float));
        cache_rows = std::min(cache_rows, num_vec);
        std::cout << "Cache size: " << cache_rows << " rows (" << (100.f * cache_rows / (float)num_vec) << " % of data set)" << std::endl;
        if (cache_rows < WS)
        {
            std::cout << "Cache smaller than working set, can't continue" << std::endl;
            return;
        }

        assert_cuda(cudaMalloc(&d_KCacheRowIdx, cache_rows * sizeof(int)));
        assert_cuda(cudaMalloc(&d_KCacheRowPriority, cache_rows * sizeof(int)));
        assert_cuda(cudaMalloc(&d_K, cache_rows * num_vec_aligned * sizeof(float)));

#ifdef USE_DAIFLETCHER
        assert_cuda(cudaMalloc(&d_df_g, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_gh, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_y, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_tempv, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_d, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_Ad, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_t, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_xplus, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_tplus, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_sk, WS * sizeof(float)));
        assert_cuda(cudaMalloc(&d_df_yk, WS * sizeof(float)));
#endif

        memsetCuda<int>(d_ws_priority, 0, num_vec);
#ifdef USE_DAIFLETCHER
        assert_cuda(cudaMemset(d_g, 0, num_vec_aligned * sizeof(float)));
#else
        memsetCuda<float>(d_g, 1, num_vec_aligned);
#endif
        memsetCuda<int>(d_KCacheRemapIdx, -1, num_vec);
        memsetCuda<int>(d_KCacheRowIdx, -1, cache_rows);
        memsetCuda<int>(d_KCacheRowPriority, -1, cache_rows);
        int cacheUpdateCnt = 0;
        assert_cuda(cudaMemcpyToSymbol(d_cacheUpdateCnt, &cacheUpdateCnt, sizeof(int), 0));
        int cacheChanges[6] = { -1, -1, -1, -1, -1, -1 };
        assert_cuda(cudaMemcpyToSymbol(d_KCacheChanges, cacheChanges, 3 * sizeof(int2), 0));

        std::cout << "Precalculating KDiag" << std::endl;
        computeKDiag(d_KDiag, num_vec);

        size_t num_vec_shrunk = num_vec;

        thrust::device_ptr<float> dev_sortval = thrust::device_pointer_cast(d_sortval);
        thrust::device_ptr<int> dev_sortidx = thrust::device_pointer_cast(d_sortidx);

        const int findActiveSetBlockSize = STATIC_MAX(128, NC);
        const int findCacheRowBlockSize = STATIC_MAX(512, WS / ELEM_PER_THREAD);
        const int fillWSBlockSize = STATIC_MIN(512, WS);
        dim3 dimBlock(256);
        dim3 dimGrid(getgriddim(num_vec, (size_t)dimBlock.x));
        dim3 dimBlock32x8(32, 8);
        dim3 dimBlockFindActiveSet(findActiveSetBlockSize);
        dim3 dimGridFindActiveSet(std::min<int>(dimBlockFindActiveSet.x, WS));
        dim3 dimBlockCalcKLocal(32, CALC_KLOCAL_NUM_WARPS);
        dim3 dimGridCalcKLocal(WS / (CALC_KLOCAL_TILE_X * dimBlockCalcKLocal.y), WS);
        size_t total_num_iter = 0;
        assert_cuda(cudaMemcpyToSymbol(d_total_num_iter, &total_num_iter, sizeof(total_num_iter)));
        float last_diff = 0;

        //kernel grid/block dimension check
        //kernelFindNBest
        assert(findActiveSetBlockSize >= NC && NUM_SORT_BLOCKS * NC > NC + findActiveSetBlockSize);
        //check cache
        assert(findCacheRowBlockSize * ELEM_PER_THREAD >= WS);

        //kernel timers
        float timer_check_cache = 0;
        int counter_check_cache = 0;
        float timer_local_solver = 0;
        int counter_local_solver = 0;
        float timer_Gupdate = 0;
        int counter_Gupdate = 0;
        float timer_find_active_set = 0;
        int counter_find_active_set = 0;
        float timer_find_nbest = 0;
        int counter_find_nbest = 0;
        float timer_fill_ws = 0;
        int counter_fill_ws = 0;
        float timer_calc_klocal = 0;
        int counter_calc_klocal = 0;

        int MAX_GLOBAL_ITERS = 0;
        SYNC_RESET(0);

        int updateGCnt[2] = { 0 };
        assert_cuda(cudaMemcpyToSymbol(d_updateGCnt, updateGCnt, sizeof(updateGCnt)));

        //std::ofstream fout_wsp("wsp.dat", std::ios::out | std::ios::binary);

        std::cout << "Starting iterations" << std::endl;
        for (int iter = 0;; iter++)
        {
            //std::cout << "Iter " << iter << std::endl;
#if 0
            kernelPrepareSortI<<<dimGrid, dimBlock>>>(d_sortval, d_sortidx, d_y, d_g, d_alpha, C, num_vec_shrunk);
            thrust::sort_by_key(dev_sortval, dev_sortval + num_vec, dev_sortidx, thrust::greater<float>());
            cudaMemcpy(d_workingset, d_sortidx, WS / 2 * sizeof(int), cudaMemcpyDeviceToDevice);
            //float mi, mj;
            //cudaMemcpy(&mi, d_sortval, sizeof(float), cudaMemcpyDeviceToHost);

            //for (int i = 0; i < WS / 2; i++)
            //    checkCache(sparse, d_workingset + i, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma);

            kernelPrepareSortJ<<<dimGrid, dimBlock>>>(d_sortval, d_sortidx, d_y, d_g, d_alpha, C, num_vec_shrunk);
            //kernelPrepareSortJSecondOrder<<<dimGrid, dimBlock>>>(d_sortval, d_sortidx, d_y, d_g, d_alpha, C, num_vec_shrunk, num_vec_aligned, d_workingset, d_K, d_KDiag, d_KCacheRemapIdx);
            thrust::sort_by_key(dev_sortval, dev_sortval + num_vec, dev_sortidx, thrust::greater<float>());
            cudaMemcpy(d_workingset + WS / 2, d_sortidx, WS / 2 * sizeof(int), cudaMemcpyDeviceToDevice);
            //cudaMemcpy(&mj, d_sortval, sizeof(float), cudaMemcpyDeviceToHost);

            //for (int i = WS / 2; i < WS; i++)
            //    checkCache(sparse, d_workingset + i, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma);
#else
            //std::cout << "gridDim: " << dimGrid32.x << ", blockDim: " << dimBlock.x << std::endl;
            //ACCUMULATE_KERNEL_TIME(timer_find_active_set, counter_find_active_set, (kernelFindActiveSet<findActiveSetBlocksize, WS><<<dimGridFindActiveSet, dimBlockFindActiveSet>>>(d_workingset, d_y, d_g, d_alpha, C, num_vec_shrunk)));
            if (iter == 0)
            {
                //SYNC_RESET(0);
                //ACCUMULATE_KERNEL_TIME(timer_find_active_set, counter_find_active_set, (kernelFindActiveSetV2<findActiveSetBlocksize, WS, NUM_SORT_BLOCKS><<<NUM_SORT_BLOCKS, findActiveSetBlocksize>>>(d_workingset, d_y, d_g, d_alpha, C, num_vec_shrunk, d_sortValues, d_sortIdxs, SYNC_BUFFER(0))));

                kernelPrepareSortI << <dimGrid, dimBlock >> >(d_sortval, d_sortidx, d_y, d_g, d_alpha, C, num_vec_shrunk);
                thrust::sort_by_key(dev_sortval, dev_sortval + num_vec, dev_sortidx, thrust::greater<float>());
                cudaMemcpy(d_workingset, d_sortidx, WS / 2 * sizeof(int), cudaMemcpyDeviceToDevice);
                kernelPrepareSortJ << <dimGrid, dimBlock >> >(d_sortval, d_sortidx, d_y, d_g, d_alpha, C, num_vec_shrunk);
                thrust::sort_by_key(dev_sortval, dev_sortval + num_vec, dev_sortidx, thrust::greater<float>());
                cudaMemcpy(d_workingset + WS / 2, d_sortidx, WS / 2 * sizeof(int), cudaMemcpyDeviceToDevice);
                /*std::vector<int> ws(WS);
                for (int i = 0; i < WS; i++)
                ws[i] = i;
                cudaMemcpy(d_workingset, &ws[0], WS * sizeof(int), cudaMemcpyHostToDevice);*/
            }
            else
            {
                //SYNC_RESET(0);
#if 0
                ACCUMULATE_KERNEL_TIME(timer_find_nbest, counter_find_nbest, (kernelFindNBest<findActiveSetBlockSize, NC, NUM_SORT_BLOCKS><<<NUM_SORT_BLOCKS, findActiveSetBlockSize>>>(d_workingset, d_y, d_g, d_alpha, C, num_vec_shrunk, d_ws_priority, d_sortValues, d_sortIdxs, SYNC_BUFFER(0))));
#else
                ACCUMULATE_KERNEL_TIME(timer_find_nbest, counter_find_nbest, (kernelFindNBestV2<findActiveSetBlockSize, NC, NUM_SORT_BLOCKS> << <NUM_SORT_BLOCKS, findActiveSetBlockSize >> >(d_workingset, d_y, d_g, d_alpha, C, num_vec_shrunk, d_ws_priority, d_sortValues, d_sortIdxs, SYNC_BUFFER(0))));
#endif
                //SYNC_RESET(0);
                ACCUMULATE_KERNEL_TIME(timer_fill_ws, counter_fill_ws, (kernelFillWorkingSet<WS, NC> << <1, fillWSBlockSize >> >(d_workingset, d_alpha, C, d_ws_priority, d_sortIdxs)));

                //std::vector<int> ws_priority(num_vec);
                //assert_cuda(cudaMemcpy(&ws_priority[0], d_ws_priority, num_vec * sizeof(int), cudaMemcpyDeviceToHost));
                //fout_wsp.write((const char *)&ws_priority[0], num_vec * sizeof(int));
            }
            //export_cuda_buffer(d_sortValues, WS, NUM_SORT_BLOCKS, sizeof(float), "sortValues.dat");
            //export_cuda_buffer(d_sortIdxs, WS, NUM_SORT_BLOCKS, sizeof(int), "sortIdxs.dat");
#endif
            //if (iter == 1)
            //import_cuda_buffer(d_workingset, WS, 1, sizeof(int), "E:\\ws_" + std::to_string(iter) + ".dat");
            //std::cout << "===\nWS: ";
            //print_cuda_buffer(d_workingset, WS, 1, std::cout);
            //int ws[WS];
            //for (int i = 0; i < WS; i++)
            //    ws[i] = i;
            //assert_cuda(cudaMemcpy(d_workingset, ws, WS * sizeof(int), cudaMemcpyHostToDevice));

#ifndef CALC_KLOCAL
            if (sparse)
            {
                ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCacheV2<findCacheRowBlockSize, WS, ELEM_PER_THREAD>(sparse, d_workingset, d_x, d_x2, sparse_data_gpu, ellpack_data_gpu, jds_data_gpu, use_ellpack, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma)));
            }
            else
            {
#ifdef USE_CUBLAS
                ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCacheCublas<findCacheRowBlockSize, WS, ELEM_PER_THREAD>(sparse, d_workingset, d_x, d_xT, d_xTile, d_x2, sparse_data_gpu, d_K, d_KTile, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma, cublas)));
#else
                ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCacheV2<findCacheRowBlockSize, WS, ELEM_PER_THREAD>(sparse, d_workingset, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma)));
#endif
            }
            ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelCopyKToLocal<WS><<<WS, WS>>>(d_workingset, d_K, d_KLocal, d_KCacheRemapIdx, num_vec_aligned)));
#else
#if 0
            ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelCalcKLocal<CALC_KLOCAL_TILE_X, CALC_KLOCAL_NUM_WARPS><<<dimGridCalcKLocal, dimBlockCalcKLocal>>>(d_KLocal, d_K, d_KCacheRemapIdx, d_workingset, d_x, d_x2, gamma, num_vec_aligned, dim, dim_aligned)));
#else
            if (sparse)
            {
                assert_cuda(cudaMemset(d_denseVec, 0, dim_aligned * WS * sizeof(float)));
                dim3 dimBlock(256);
                if (use_ellpack)
                {
                    dim3 dimGrid(getgriddim<int>(WS, dimBlock.x));
                    ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelMakeDenseVecWSKLocal<WS> << <dimGrid, dimBlock >> >(d_KCacheRemapIdx, ellpack_data_gpu, d_workingset, d_denseVec, dim_aligned)));
                    dimGrid = dim3(getgriddim<int>(WS, dimBlock.x), WS);
                    ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelCalcKLocalSparse<WS> << <dimGrid, dimBlock >> >(d_KLocal, d_K, d_KCacheRemapIdx, ellpack_data_gpu, d_x2, d_denseVec, d_workingset, gamma, num_vec, num_vec_aligned, dim, dim_aligned)));
                }
                else
                {
                    dim3 dimGrid(std::min(64, getgriddim<int>(dim, dimBlock.x)), WS);
                    ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelMakeDenseVecWSKLocal<WS> << <dimGrid, dimBlock >> >(d_KCacheRemapIdx, sparse_data_gpu, d_workingset, d_denseVec, dim_aligned)));
                    const int NUM_WARPS = 8;
                    dimBlock = dim3(32, NUM_WARPS);
                    dimGrid = dim3(WS / NUM_WARPS, WS);
                    ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelCalcKLocalSparse<WS, NUM_WARPS> << <dimGrid, dimBlock >> >(d_KLocal, d_K, d_KCacheRemapIdx, sparse_data_gpu, d_x2, d_denseVec, d_workingset, gamma, num_vec, num_vec_aligned, dim, dim_aligned)));
                }
            }
            else
            {
                const int BLOCK_SIZE = 16;
                dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
                dim3 dimGrid(WS / BLOCK_SIZE, WS / BLOCK_SIZE);
                //ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelCalcKLocalV2_NT<BLOCK_SIZE><<<dimGrid, dimBlock>>>(d_KLocal, d_x, d_xT, d_x2, d_workingset, gamma, num_vec, num_vec_aligned, dim, dim_aligned, WS)));
                ACCUMULATE_KERNEL_TIME(timer_calc_klocal, counter_calc_klocal, (kernelCalcKLocalV2_NN<WS, BLOCK_SIZE> << <dimGrid, dimBlock >> >(d_KLocal, d_K, d_KCacheRemapIdx, d_x, d_x2, d_y, d_workingset, gamma, num_vec_aligned, dim, dim_aligned)));
            }
#endif
            //export_cuda_buffer(d_KLocal, WS, WS, sizeof(float), "KLocal.dat");
#endif
            //for (int i = 0; i < WS; i++)
            //ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCache(sparse, d_workingset + i, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma)));
            //export_cuda_buffer(d_K, num_vec_aligned, cache_rows, sizeof(float), "K.dat");
            //export_cuda_buffer(d_KCacheRemapIdx, num_vec, 1, sizeof(int), "KRemapIdx.dat");
            //export_cuda_buffer(d_KCacheRowIdx, cache_rows, 1, sizeof(int), "KRowIdx.dat");
            //int ws[WS];
            //assert_cuda(cudaMemcpy(ws, d_workingset, WS * sizeof(int), cudaMemcpyDeviceToHost));
            //std::sort(ws, ws + WS);
            //std::cout << "WS:";
            //for (int i = 0; i < WS; i++)
            //    std::cout << " " << ws[i];
            //std::cout << std::endl;
            //std::ostringstream ss;
            //ss << iter;
            //std::string siter = ss.str();
            //export_cuda_buffer(d_workingset, WS, 1, sizeof(int), "ws" + siter + ".dat");
            //export_cuda_buffer(d_y, num_vec, 1, sizeof(float), "y" + siter + ".dat");
            //export_cuda_buffer(d_g, num_vec, 1, sizeof(float), "g" + siter + ".dat");
            //export_cuda_buffer(d_alpha, num_vec, 1, sizeof(float), "alpha" + siter + ".dat");
            //export_cuda_buffer(d_workingset, WS, 1, sizeof(int), "ws.dat");

#ifdef USE_DAIFLETCHER
            dim3 dimBlockKtoH(32, 8);
            dim3 dimGridKtoH(getgriddim(WS, dimBlockKtoH.x), getgriddim(WS, dimBlockKtoH.y));
            kernelKtoHLocal<WS><<<dimGridKtoH, dimBlockKtoH>>>(d_KLocal, d_y, d_workingset);
            kernelCalcE<WS><<<1, WS>>>(d_workingset, d_alpha, d_y, num_vec);
            //float e;
            //assert_cuda(cudaMemcpyFromSymbol(&e, d_df_e, sizeof(e)));
            //std::cout << "===\ne: " << e << std::endl;
            //std::cout << "===\nalpha: ";
            //print_cuda_buffer(d_alpha, WS, 1, std::cout);
            //std::cout << "===\ng: ";
            //print_cuda_buffer(d_g, WS, 1, std::cout);
            //std::cout << "===\ny: ";
            //print_cuda_buffer(d_y, WS, 1, std::cout);
            kernelUpdateGHLocal<WS><<<1, WS>>>(d_df_gh, d_g, d_y, d_alpha, d_workingset, d_KLocal);
            //std::cout << "===\ngh: ";
            //print_cuda_buffer(d_df_gh, WS, 1, std::cout);
            //std::cout << "===\K row 0: ";
            //print_cuda_buffer(d_KLocal, WS, 1, std::cout);
            float DELTAvpm = 1e-3,
                DELTAkin = 1;
            //ACCUMULATE_KERNEL_TIME(timer_local_solver, counter_local_solver, (kernelFletcherAlg2A<WS><<<1, WS>>>(WS, d_KLocal, d_df_gh, C, d_y, d_alpha, d_alphadiff, DELTAvpm*DELTAkin, d_df_g, d_df_y, d_df_tempv, d_df_d, d_df_Ad, d_df_t, d_df_xplus, d_df_tplus, d_df_sk, d_df_yk)));
            ACCUMULATE_KERNEL_TIME(timer_local_solver, counter_local_solver, (kernelFletcherAlg2A<WS><<<1, WS>>>(d_KLocal, d_df_gh, C, d_y, d_alpha, d_alphadiff, DELTAvpm*DELTAkin, d_workingset)));
#else
#if 0
            kernelSMO1Block<true, true, false><<<1, dimBlock32x8>>>(d_x, d_x2, d_y, d_g, d_alpha, d_workingset, gamma, C, eps, num_vec, num_vec_aligned, dim, dim_aligned, d_K, d_KCacheRemapIdx);
#else
            if (ELEM_PER_THREAD > 1)
                ACCUMULATE_KERNEL_TIME(timer_local_solver, counter_local_solver, (kernelSMO1BlockV2N<WS, ELEM_PER_THREAD> << <1, WS / ELEM_PER_THREAD >> >(d_y, d_g, d_alpha, d_alphadiff, d_workingset, gamma, C, eps, num_vec_aligned, d_KLocal, d_KCacheRemapIdx)));
            else
                ACCUMULATE_KERNEL_TIME(timer_local_solver, counter_local_solver, (kernelSMO1BlockV2<WS> << <1, WS >> >(d_y, d_g, d_alpha, d_alphadiff, d_workingset, gamma, C, eps, num_vec_aligned, d_KLocal, d_KCacheRemapIdx)));
#endif
            //print_cuda_buffer(d_g, 10, 1, std::cout);
#endif
            //std::cout << "===\nalpha: ";
            //print_cuda_buffer(d_alpha, WS, 1, std::cout);
            //std::cout << "===\nAdiff: ";
            //print_cuda_buffer(d_alphadiff, WS, 1, std::cout);
#ifdef CALC_KLOCAL
            if (sparse)
            {
                ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCacheKLocal<findCacheRowBlockSize, WS, ELEM_PER_THREAD>(sparse, d_workingset, d_x, d_x2, sparse_data_gpu, ellpack_data_gpu, jds_data_gpu, use_ellpack, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_alphadiff, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma)));
            }
            else
            {
#ifdef USE_CUBLAS
                ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCacheCublasKLocal<findCacheRowBlockSize, WS, ELEM_PER_THREAD>(sparse, d_workingset, d_x, d_xT, d_xTile, d_x2, sparse_data_gpu, d_K, d_KTile, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_alphadiff, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma, cublas)));
#else
                ACCUMULATE_KERNEL_TIME(timer_check_cache, counter_check_cache, (checkCacheKLocal<findCacheRowBlockSize, WS, ELEM_PER_THREAD>(sparse, d_workingset, d_x, d_x2, sparse_data_gpu, ellpack_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_alphadiff, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma)));
#endif
            }
#endif
            ACCUMULATE_KERNEL_TIME(timer_Gupdate, counter_Gupdate, (kernelUpdateG<WS> << <dimGrid, dimBlock >> >(d_y, d_g, d_alphadiff, d_workingset, d_K, d_KCacheRemapIdx, num_vec, num_vec_aligned)));

            //export_cuda_buffer(d_alpha, num_vec, 1, sizeof(float), "alpha.dat");
            //std::cout << "===\ng: ";
            //print_cuda_buffer(d_g, WS, 1, std::cout);

            if ((iter + 1) % 100 == 0)
            {
                float diff;
                assert_cuda(cudaMemcpyFromSymbol(&diff, d_diff, sizeof(float), 0));
                //if (diff == last_diff)
                //{
                //    int ws[WS];
                //    assert_cuda(cudaMemcpy(ws, d_workingset, WS * sizeof(int), cudaMemcpyDeviceToHost));
                //    std::sort(ws, ws + WS);
                //    std::cout << "WS:";
                //    for (int i = 0; i < WS; i++)
                //        std::cout << " " << ws[i];
                //    std::cout << std::endl;
                //    break;
                //}
                //last_diff = diff;

                size_t block_num_iter;
                assert_cuda(cudaMemcpyFromSymbol(&block_num_iter, d_block_num_iter, sizeof(block_num_iter), 0));
                assert_cuda(cudaMemcpyFromSymbol(&total_num_iter, d_total_num_iter, sizeof(total_num_iter), 0));
                std::cout << "Iter: " << total_num_iter << ", global iter: " << iter << ", diff: " << diff << std::endl;
                //std::cout << "Local iter: " << block_num_iter << std::endl;

                if (block_num_iter >= MAX_BLOCK_ITER)
                    std::cout << "Warning: Maximum number of iterations per block was reached" << std::endl;

                if (block_num_iter == 0)
                {
                    assert_cuda(cudaMemcpyFromSymbol(rho, d_rho, sizeof(float), 0));
                    std::cout << "Optimality reached after " << iter << " iterations, stopping loop. rho = " << *rho << std::endl;
                    break;
                }
            }
            if (iter >= MAX_GLOBAL_ITERS)
            {
                //kernelFindActiveSet<findActiveSetBlocksize><<<dimGridFindActiveSet, dimBlockFindActiveSet>>>(d_workingset, d_y, d_g, d_alpha, C, num_vec_shrunk);
                //kernelFindCacheRowV2<256><<<1, 256>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
                //int cache_rows_to_compute[WS];
                //int num_cache_rows_to_compute;
                //assert_cuda(cudaMemcpyFromSymbol(&num_cache_rows_to_compute, d_num_cache_rows_to_compute, sizeof(int), 0));
                //assert_cuda(cudaMemcpyFromSymbol(cache_rows_to_compute, d_cache_rows_to_compute, num_cache_rows_to_compute * sizeof(int), 0));
                //std::cout << "Num cache rows to compute: " << num_cache_rows_to_compute << "\n";
                //for (int i = 0; i < num_cache_rows_to_compute; i++)
                //    std::cout << " " << cache_rows_to_compute[i];
                //std::cout << std::endl;

                //std::cout << "WS: "; print_cuda_buffer(d_workingset, 10, 1);
                //std::cout << "K row 0: "; print_cuda_buffer(d_K, 10, 1);
                //std::cout << "K row 1: "; print_cuda_buffer(d_K + num_vec_aligned, 10, 1);
                //std::cout << "KCacheRowIdx: "; print_cuda_buffer(d_KCacheRowIdx, cache_rows, 1);

                //break; //LLLLLLLLLLLLLLLLLLLLLL
            }
        }

        assert_cuda(cudaMemcpyFromSymbol(&cacheUpdateCnt, d_cacheUpdateCnt, sizeof(int), 0));
        std::cout << "Cache row updates: " << cacheUpdateCnt << std::endl;
        assert_cuda(cudaMemcpyFromSymbol(updateGCnt, d_updateGCnt, sizeof(updateGCnt)));
        std::cout << "Update G: updated " << updateGCnt[0] << ", skipped " << updateGCnt[1] << std::endl;

        PRINT_KERNEL_TIME("CheckCache       ", timer_check_cache, counter_check_cache);
        PRINT_KERNEL_TIME("Find Active set  ", timer_find_active_set, counter_find_active_set);
        PRINT_KERNEL_TIME("Find N-best      ", timer_find_nbest, counter_find_nbest);
        PRINT_KERNEL_TIME("Fill working set ", timer_fill_ws, counter_fill_ws);
        PRINT_KERNEL_TIME("Calc/Copy KLocal ", timer_calc_klocal, counter_calc_klocal);
        PRINT_KERNEL_TIME("Local Solver     ", timer_local_solver, counter_local_solver);
        PRINT_KERNEL_TIME("G-update         ", timer_Gupdate, counter_Gupdate);
#ifdef USE_TIMERS
        std::cout << "Total device time       : " << (timer_check_cache
            + timer_find_active_set
            + timer_find_nbest
            + timer_fill_ws
            + timer_calc_klocal
            + timer_local_solver
            + timer_Gupdate) << " ms\n";
#endif

        assert_cuda(cudaMemcpy(alpha, d_alpha, num_vec * sizeof(float), cudaMemcpyDeviceToHost));

        if (sparse)
        {
            if (use_ellpack)
                freeCudaEllpack(ellpack_data_gpu);
            else
                freeCudaCsr(sparse_data_gpu);
            freeCudaJds(jds_data_gpu);
            assert_cuda(cudaFree(d_denseVec));
        }
        else
        {
            assert_cuda(cudaFree(d_x));
        }

        assert_cuda(cudaFree(d_x2));
        assert_cuda(cudaFree(d_K));
        assert_cuda(cudaFree(d_KLocal));
        assert_cuda(cudaFree(d_KDiag));
        assert_cuda(cudaFree(d_KCacheRemapIdx));
        assert_cuda(cudaFree(d_KCacheRowIdx));
        assert_cuda(cudaFree(d_KCacheRowPriority));
        assert_cuda(cudaFree(d_alpha));
        assert_cuda(cudaFree(d_alphadiff));
        assert_cuda(cudaFree(d_y));
        assert_cuda(cudaFree(d_g));
        assert_cuda(cudaFree(d_sortval));
        assert_cuda(cudaFree(d_sortidx));
	    assert_cuda(cudaFree(d_sortValues));
	    assert_cuda(cudaFree(d_sortIdxs));
        assert_cuda(cudaFree(d_workingset));
        assert_cuda(cudaFree(d_ws_priority));
#ifdef USE_CUBLAS
        assert_cuda(cudaFree(d_xTile));
        assert_cuda(cudaFree(d_KTile));
        assert_cuda(cudaFree(d_xT));
#endif
#ifdef USE_DAIFLETCHER
        assert_cuda(cudaFree(d_df_g));
        assert_cuda(cudaFree(d_df_gh));
        assert_cuda(cudaFree(d_df_y));
        assert_cuda(cudaFree(d_df_tempv));
        assert_cuda(cudaFree(d_df_d));
        assert_cuda(cudaFree(d_df_Ad));
        assert_cuda(cudaFree(d_df_t));
        assert_cuda(cudaFree(d_df_xplus));
        assert_cuda(cudaFree(d_df_tplus));
        assert_cuda(cudaFree(d_df_sk));
        assert_cuda(cudaFree(d_df_yk));
#endif
        assert_cublas(cublasDestroy(cublas));
    }
    catch (...)
    {
        if (sparse)
        {
            if (use_ellpack)
                freeCudaEllpack(ellpack_data_gpu);
            else
                freeCudaCsr(sparse_data_gpu);
            freeCudaJds(jds_data_gpu);
            cudaFree(d_denseVec);
        }
        else
        {
            cudaFree(d_x);
        }

        cudaFree(d_x2);
        cudaFree(d_K);
        cudaFree(d_KLocal);
        cudaFree(d_KDiag);
        cudaFree(d_KCacheRemapIdx);
        cudaFree(d_KCacheRowIdx);
        cudaFree(d_KCacheRowPriority);
        cudaFree(d_alpha);
        cudaFree(d_alphadiff);
        cudaFree(d_y);
        cudaFree(d_g);
        cudaFree(d_sortval);
        cudaFree(d_sortidx);
	    cudaFree(d_sortValues);
	    cudaFree(d_sortIdxs);
        cudaFree(d_workingset);
        cudaFree(d_ws_priority);
#ifdef USE_CUBLAS
        cudaFree(d_xTile);
        cudaFree(d_KTile);
        cudaFree(d_xT);
#endif
#ifdef USE_DAIFLETCHER
        cudaFree(d_df_g);
        cudaFree(d_df_gh);
        cudaFree(d_df_y);
        cudaFree(d_df_tempv);
        cudaFree(d_df_d);
        cudaFree(d_df_Ad);
        cudaFree(d_df_t);
        cudaFree(d_df_xplus);
        cudaFree(d_df_tplus);
        cudaFree(d_df_sk);
        cudaFree(d_df_yk);
#endif
        cublasDestroy(cublas);
        throw;
    }

}

void OrcusSvm1B::Train(float * alpha, float * rho, bool sparse, const Data & x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps, int ws_size)
{
    if (ws_size == 0)
    {
        if (num_vec >= 250000)
            ws_size = 2048;
        else
            ws_size = 1024;
/*#ifdef USE_ELLPACK
        if (sparse)
        {
            size_t free_mem, total_mem;
            assert_cuda(cudaFree(nullptr));  //force CUDA init
            assert_cuda(cuMemGetInfo(&free_mem, &total_mem));

            std::vector<int> rowLen(x.sparse->numRows);
            std::adjacent_difference(x.sparse->rowOffsets + 1, x.sparse->rowOffsets + x.sparse->numRows + 1, rowLen.begin());
            int maxRowLen = *std::max_element(rowLen.begin(), rowLen.end());
            size_t ellpack_size = 2 * x.sparse->numRows * maxRowLen * sizeof(float);

            if (ellpack_size >= free_mem * 0.5)
                ws_size = 512;
        }
#endif*/
    }
    ws_size = std::max(64, std::min(2048, ws_size));
    try
    {
        switch (ws_size)
        {
        case   64: train<  64>(alpha, rho, sparse, x, y, num_vec, num_vec_aligned, dim, dim_aligned, C, gamma, eps); break;
        case  128: train< 128>(alpha, rho, sparse, x, y, num_vec, num_vec_aligned, dim, dim_aligned, C, gamma, eps); break;
        case  256: train< 256>(alpha, rho, sparse, x, y, num_vec, num_vec_aligned, dim, dim_aligned, C, gamma, eps); break;
        case  512: train< 512>(alpha, rho, sparse, x, y, num_vec, num_vec_aligned, dim, dim_aligned, C, gamma, eps); break;
        case 1024: train<1024>(alpha, rho, sparse, x, y, num_vec, num_vec_aligned, dim, dim_aligned, C, gamma, eps); break;
        case 2048: train<2048>(alpha, rho, sparse, x, y, num_vec, num_vec_aligned, dim, dim_aligned, C, gamma, eps); break;
        default:
            std::cerr << "Unsupported working set size\n";
        }
    }
    catch (std::exception & e)
    {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        if (strstr(e.what(), "out of memory"))
            std::cerr << "Try lowering working set size\n";
    }
}
