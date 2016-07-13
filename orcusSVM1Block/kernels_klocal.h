#pragma once

template<unsigned int WS>
__global__ static void kernelMakeDenseVecWSKLocal(const int * KCacheRemapIdx, csr_gpu x, const int * ws, float * vec, int dim_aligned)
{
    int i = ws[blockIdx.y];
    if (KCacheRemapIdx[i] >= 0)
        return;

    int j = x.rowOffsets[i] + blockDim.x * blockIdx.x + threadIdx.x;
    while (j < x.rowOffsets[i + 1])
    {
        vec[dim_aligned * blockIdx.y + x.colInd[j]] = x.values[j];
        j += gridDim.x * blockDim.x;
    }
}

template<unsigned int WS>
__global__ static void kernelMakeDenseVecWSKLocal(const int * KCacheRemapIdx, ellpack_gpu x, const int * ws, float * vec, int dim_aligned)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int i = ws[k];
    if (KCacheRemapIdx[i] >= 0)
        return;

    int rowLen = x.rowLen[i];
    for (int d = 0; d < rowLen; d++)
    {
        vec[dim_aligned * k + x.colInd[x.numRows * d + i]] = x.values[x.numRows * d + i];
    }
}

//blockDim.x = warpSize
//blockDim.y = num warps
//gridDim.y = WS
template<unsigned int WS, unsigned int TILE_X, unsigned int NUM_WARPS>
__global__ static void kernelCalcKLocal(float * KLocal, const float * K, const int * KCacheRemapIdx, const int * ws, const float * x, const float * x2, float gamma, int num_vec_aligned, int dim, int dim_aligned)
{
    __shared__ int shWS[WS];
    __shared__ float shOut[TILE_X * NUM_WARPS];
    for (int k = blockDim.x * threadIdx.y + threadIdx.x; k < WS; k += blockDim.x * blockDim.y)
        shWS[k] = ws[k];
    __syncthreads();
    int rowWS = shWS[blockIdx.y];
    int block = TILE_X * blockDim.y * blockIdx.x;

    //while (block < WS) //probably not needed
    {
        int j = block + threadIdx.y * TILE_X;
#if 1
        int cache_row = KCacheRemapIdx[rowWS];
        if (cache_row >= 0)
        {
            for (int k = blockDim.x * threadIdx.y + threadIdx.x; k < WS; k += blockDim.x * blockDim.y)
            {
                KLocal[WS * blockIdx.y + k] = K[(size_t)num_vec_aligned * cache_row + shWS[k]];
            }
        }
        //else if ((cache_row = KCacheRemapIdx[colWS]) >= 0)
        //{
            //KLocal[WS * blockIdx.y + j] = K[num_vec_aligned * cache_row + rowWS];
        //}
        else
#endif
        {
            float sum[TILE_X] = {0};
            for (int d = threadIdx.x; d < dim; d += warpSize)
            {
#pragma unroll
                for (int ix = 0; ix < TILE_X; ix++)
                    sum[ix] += x[dim_aligned * rowWS + d] * x[dim_aligned * shWS[j + ix] + d];
            }
#pragma unroll
            for (int ix = 0; ix < TILE_X; ix++)
            {
                int colWS = shWS[j + ix];
                float s = warpReduceSum(sum[ix]);
                if (threadIdx.x == 0)
                {
                    s = x2[rowWS] + x2[colWS] - 2 * s;
                    shOut[TILE_X * threadIdx.y + ix] = expf(-gamma * s);
                }
            }
            __syncthreads();
            for (int k = blockDim.x * threadIdx.y + threadIdx.x; k < TILE_X * NUM_WARPS; k += blockDim.x * blockDim.y)
                KLocal[WS * blockIdx.y + block + k] = shOut[k];
        }

        //block += gridDim.x * blockDim.y;
    }
}

#if 0
//broken
template<int BLOCK_SIZE>
__global__ static void kernelCalcKLocalV2_NT(float * K, const float * x, const float * xT, const float * x2, const int * ws, float gamma, int num_vec, size_t num_vec_aligned, int dim, int dim_aligned, int num_rows)
{
    __shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int ws_idx = ws[row];
    float sum = 0;

    for (int block = 0; block < dim; block += BLOCK_SIZE)
    {
        if (block + threadIdx.x < dim)
            shA[threadIdx.y][threadIdx.x] = x[dim_aligned * ws_idx + block + threadIdx.x];
        else
            shA[threadIdx.y][threadIdx.x] = 0;
        if (block + threadIdx.y < dim)
            shB[threadIdx.y][threadIdx.x] = xT[num_vec_aligned * (block + threadIdx.y) + col];
        else
            shB[threadIdx.y][threadIdx.x] = 0;
        __syncthreads();

        for (int d = 0; d < BLOCK_SIZE; d++)
            sum += shA[threadIdx.y][d] * shB[d][threadIdx.x];
        __syncthreads();
    }

    if (row < num_rows && col < num_vec)
    {
        sum = x2[ws_idx] + x2[col] - 2 * sum;
        K[num_vec_aligned * row + col] = expf(-gamma * sum);
    }
}
#endif

template<unsigned int WS, int BLOCK_SIZE>
__global__ static void kernelCalcKLocalV2_NN(float * KLocal, const float * K, const int * KCacheRemapIdx, const float * x, const float * x2, const float * y, const int * ws, float gamma, size_t num_vec_aligned, int dim, int dim_aligned)
{
    __shared__ float shA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shB[BLOCK_SIZE][BLOCK_SIZE+1];
    __shared__ bool all_cached;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int colT = blockDim.x * blockIdx.x + threadIdx.y;
    int ws_idx = ws[row];
    int ws_idxJ = ws[colT];
    float sum = 0;
    if (threadIdx.x + threadIdx.y == 0)
        all_cached = true;
    __syncthreads();

    int cache_row = KCacheRemapIdx[ws_idx];
    if (cache_row < 0)
        all_cached = false;
    __syncthreads();

    if (!all_cached)
        for (int block = 0; block < dim; block += BLOCK_SIZE)
        {
            if (block + threadIdx.x < dim)
            {
                if (cache_row < 0)
                    shA[threadIdx.y][threadIdx.x] = x[dim_aligned * ws_idx + block + threadIdx.x];
                shB[threadIdx.x][threadIdx.y] = x[dim_aligned * ws_idxJ + block + threadIdx.x];
            }
            else
            {
                shA[threadIdx.y][threadIdx.x] = 0;
                shB[threadIdx.x][threadIdx.y] = 0;
            }
            __syncthreads();

            if (cache_row < 0)
                for (int d = 0; d < BLOCK_SIZE; d++)
                    sum += shA[threadIdx.y][d] * shB[d][threadIdx.x];
            __syncthreads();
        }

    if (row < WS && col < WS)
    {
        if (cache_row >= 0)
        {
            KLocal[WS * row + col] = K[(size_t)num_vec_aligned * cache_row + ws[col]];
        }
        else
        {
            sum = x2[ws_idx] + x2[ws[col]] - 2 * sum;
//#ifdef USE_DAIFLETCHER
            //KLocal[WS * row + col] = y[ws_idx] * y[ws[col]] * expf(-gamma * sum);
//#else
            KLocal[WS * row + col] = expf(-gamma * sum);
//#endif
        }
    }
}

template<int WS, int NUM_WARPS>
__global__ static void kernelCalcKLocalSparse(float * KLocal, const float * K, const int * KCacheRemapIdx, csr_gpu x, const float * d_x2, const float * vec, const int * ws, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int ws_idx = ws[blockIdx.y];
    int cache_row = KCacheRemapIdx[ws_idx];

    int block = NUM_WARPS * blockIdx.x;
    //while (block < WS)
    {
        int j = block + threadIdx.y;
        int ws_idxJ = ws[j];
        float sum = 0;
        if (cache_row < 0)
        {
            int end = x.rowOffsets[ws_idxJ] + x.rowLen[ws_idxJ];
            for (int d = x.rowOffsets[ws_idxJ] + threadIdx.x; d < end; d += warpSize)
            {
                sum += vec[dim_aligned * blockIdx.y + x.colInd[d]] * x.values[d];
            }
        }
        sum = warpReduceSum(sum);
        if (cache_row >= 0)
        {
            if (threadIdx.x < NUM_WARPS && threadIdx.y == 0)
                KLocal[WS * blockIdx.y + block + threadIdx.x] = K[(size_t)num_vec_aligned * cache_row + ws[block + threadIdx.x]];
        }
        else
        {
            if (threadIdx.x == 0)
            {
                sum = d_x2[ws_idx] + d_x2[ws_idxJ] - 2 * sum;
                //K[(size_t)num_vec_aligned * row[0] + ws_idxJ] = expf(-gamma * sum);
                KLocal[WS * blockIdx.y + j] = expf(-gamma * sum);
            }
        }
        //__syncthreads();

        //block += gridDim.x * blockDim.y;
    }
}

template<int WS>
__global__ static void kernelCalcKLocalSparse(float * KLocal, const float * K, const int * KCacheRemapIdx, ellpack_gpu x, const float * d_x2, const float * vec, const int * ws, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int ws_idx = ws[blockIdx.y];
    int cache_row = KCacheRemapIdx[ws_idx];

    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r < WS)
    {
        int ws_idxR = ws[r];
        if (cache_row >= 0)
        {
            KLocal[WS * blockIdx.y + r] = K[(size_t)num_vec_aligned * cache_row + ws_idxR];
        }
        else
        {
            float sum = 0;
            int rowLen = x.rowLen[ws_idxR];
            for (int d = 0; d < rowLen; d++)
            {
                sum += vec[dim_aligned * blockIdx.y + x.colInd[num_vec * d + ws_idxR]] * x.values[num_vec * d + ws_idxR];
            }
            sum = d_x2[ws_idx] + d_x2[ws_idxR] - 2 * sum;
            KLocal[WS * blockIdx.y + r] = expf(-gamma * sum);
        }
    }
}

template<unsigned int WS>
__global__ static void kernelCopyKToLocal(const int * ws, const float * K, float * KLocal, int * KCacheRemapIdx, int num_vec_aligned)
{
    int x = threadIdx.x;
    int y = blockIdx.x;
    KLocal[WS * y + x] = K[(size_t)num_vec_aligned * KCacheRemapIdx[ws[y]] + ws[x]];
}

