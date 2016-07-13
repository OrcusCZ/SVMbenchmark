#pragma once

//TODO: remove sparse arguments from cublas functions

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

template<unsigned int WS>
__global__ static void kernelMakeDenseVecWS(const int * KCacheRowIdx, csr_gpu x, float * vec, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y)
        return;
    int row = d_cache_rows_to_compute[blockIdx.y];
    int i = KCacheRowIdx[row];

    int j = x.rowOffsets[i] + blockDim.x * blockIdx.x + threadIdx.x;
    int end = x.rowOffsets[i + 1];
    while (j < end)
    {
        vec[dim_aligned * blockIdx.y + x.colInd[j]] = x.values[j];
        j += gridDim.x * blockDim.x;
    }
}

template<unsigned int WS>
__global__ static void kernelMakeDenseVecWS(const int * KCacheRowIdx, ellpack_gpu x, float * vec, int dim_aligned)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (d_num_cache_rows_to_compute <= k)
        return;
    int row = d_cache_rows_to_compute[k];
    int i = KCacheRowIdx[row];

    int rowLen = x.rowLen[i];
    for (int d = 0; d < rowLen; d++)
    {
        vec[dim_aligned * k + x.colInd[x.numRows * d + i]] = x.values[x.numRows * d + i];
    }
}

template<unsigned int WS, unsigned int TILE>
__global__ static void kernelCopyXTileT(float * xTile, const float * x, const int * KCacheRowIdx, size_t dim, size_t dim_aligned, size_t num_vec, size_t num_vec_aligned)
{
	__shared__ float tile[TILE][TILE + 1];
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x,
		yIndexO = blockDim.x * blockIdx.x + threadIdx.y,
        yIndex = blockDim.y * blockIdx.y + threadIdx.y,
        xIndexO = blockDim.y * blockIdx.y + threadIdx.x;
    int ws_size = d_num_cache_rows_to_compute;
    int row = d_cache_rows_to_compute[yIndex];
    if (xIndex < dim && yIndex < ws_size)
        tile[threadIdx.y][threadIdx.x] = x[dim_aligned * KCacheRowIdx[row] + xIndex];
    __syncthreads();
    if (xIndexO < ws_size && yIndexO < dim)
        xTile[WS * yIndexO + xIndexO] = tile[threadIdx.x][threadIdx.y];
    __syncthreads();
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

//block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS>
__global__ static void kernelFindCacheRowV2(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    //__shared__ int shNVal[WS];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    if (threadIdx.x < WS)
    {
        if (KCacheRemapIdx[ws[threadIdx.x]] < 0)
            orderN = atomicAdd(&num, 1);
        else
            KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
    }
    __syncthreads();

    for (int n = 0; n < num; n++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            //shNVal[n] = shVal[0];
            shNIdx[n] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    if (orderN >= 0)
    {
        int cache_row = shNIdx[orderN];
        d_cache_rows_to_compute[orderN] = cache_row;
        int irow = KCacheRowIdx[cache_row];
        if (irow >= 0)
            KCacheRemapIdx[irow] = -1;
        KCacheRowIdx[cache_row] = ws[threadIdx.x];
        KCacheRemapIdx[ws[threadIdx.x]] = cache_row;
        KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
    }
    //KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

//N * block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS, unsigned int N>
__global__ static void kernelFindCacheRowNV2(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    //__shared__ int shNVal[WS];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN[N];
    for (int n = 0; n < N; n++)
        orderN[n] = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    for (int i = threadIdx.x, n = 0; i < WS; i += blockDim.x, n++)
    {
        if (KCacheRemapIdx[ws[i]] < 0)
            orderN[n] = atomicAdd(&num, 1);
        else
            KCacheRowPriority[KCacheRemapIdx[ws[i]]] = d_cacheUpdateCnt;
    }
    __syncthreads();

    for (int m = 0; m < num; m++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            //shNVal[n] = shVal[0];
            shNIdx[m] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x, n = 0; n < N; i += blockDim.x, n++)
        if (orderN[n] >= 0)
        {
            int cache_row = shNIdx[orderN[n]];
            d_cache_rows_to_compute[orderN[n]] = cache_row;
            int irow = KCacheRowIdx[cache_row];
            if (irow >= 0)
                KCacheRemapIdx[irow] = -1;
            KCacheRowIdx[cache_row] = ws[i];
            KCacheRemapIdx[ws[i]] = cache_row;
            KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
        }
    //KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

//block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS>
__global__ static void kernelFindCacheRowKLocal(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows, const float * alphadiff)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    //__shared__ int shNVal[WS];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    if (threadIdx.x < WS)
    {
        if (alphadiff[threadIdx.x] != 0)
        {
            if (KCacheRemapIdx[ws[threadIdx.x]] < 0)
                orderN = atomicAdd(&num, 1);
            else
                KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
        }
    }
    __syncthreads();

    for (int n = 0; n < num; n++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            //shNVal[n] = shVal[0];
            shNIdx[n] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    if (orderN >= 0)
    {
        int cache_row = shNIdx[orderN];
        d_cache_rows_to_compute[orderN] = cache_row;
        int irow = KCacheRowIdx[cache_row];
        if (irow >= 0)
            KCacheRemapIdx[irow] = -1;
        KCacheRowIdx[cache_row] = ws[threadIdx.x];
        KCacheRemapIdx[ws[threadIdx.x]] = cache_row;
        KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
    }
    //KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
}

//N * block size must be larger or equal to WS
template<unsigned int blockSize, unsigned int WS, unsigned int N>
__global__ static void kernelFindCacheRowKLocalN(const int * ws, int * KCacheRemapIdx, int * KCacheRowIdx, volatile int * KCacheRowPriority, int cache_rows, const float * alphadiff)
{
    __shared__ int shVal[blockSize];
    __shared__ int shIdx[blockSize];
    //__shared__ int shNVal[WS];
    __shared__ int shNIdx[WS];
    __shared__ int num;
    int orderN[N];
    for (int n = 0; n < N; n++)
        orderN[n] = -1;

    if (threadIdx.x == 0)
        num = 0;
    __syncthreads();
    for (int i = threadIdx.x, n = 0; i < WS; i += blockDim.x, n++)
    {
        if (alphadiff[i] != 0)
        {
            if (KCacheRemapIdx[ws[i]] < 0)
                orderN[n] = atomicAdd(&num, 1);
            else
                KCacheRowPriority[KCacheRemapIdx[ws[i]]] = d_cacheUpdateCnt;
        }
    }
    __syncthreads();

    for (int m = 0; m < num; m++)
    {

        int v = INT_MAX;
        int i = -1;
        for (int k = threadIdx.x; k < cache_rows; k += blockDim.x)
        {
            if (KCacheRowPriority[k] < v)
            {
                v = KCacheRowPriority[k];
                i = k;
            }
        }
        shVal[threadIdx.x] = v;
        shIdx[threadIdx.x] = i;
        blockMinReduce2(shVal, shIdx);
        if (threadIdx.x == 0)
        {
            //shNVal[n] = shVal[0];
            shNIdx[m] = shIdx[0];
            KCacheRowPriority[shIdx[0]] = INT_MAX;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x, n = 0; n < N; i += blockDim.x, n++)
        if (orderN[n] >= 0)
        {
            int cache_row = shNIdx[orderN[n]];
            d_cache_rows_to_compute[orderN[n]] = cache_row;
            int irow = KCacheRowIdx[cache_row];
            if (irow >= 0)
                KCacheRemapIdx[irow] = -1;
            KCacheRowIdx[cache_row] = ws[i];
            KCacheRemapIdx[ws[i]] = cache_row;
            KCacheRowPriority[cache_row] = d_cacheUpdateCnt;
        }
    //KCacheRowPriority[KCacheRemapIdx[ws[threadIdx.x]]] = d_cacheUpdateCnt;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        d_num_cache_rows_to_compute = num;
        d_cacheUpdateCnt += num;
    }
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

__global__ static void kernelCublasFinalize(float * K, const float * KTile, const float * x2, const int * KCacheRowIdx, size_t num_vec, size_t num_vec_aligned, float gamma)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t k = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < num_vec && k < d_num_cache_rows_to_compute)
    {
        int row = d_cache_rows_to_compute[k];
        int j = KCacheRowIdx[row];
        size_t idx = num_vec_aligned * k + i;
        float s = KTile[idx];
        s = x2[j] + x2[i] - 2 * s;
        K[num_vec_aligned * row + i] = expf(-gamma * s);
    }
}

template<unsigned int TILE>
__global__ static void kernelCheckCachePriorityV2(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * x, const float * x2, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int last = d_cacheRow;
    if (last < 0)
        return;
    __shared__ float shsum[TILE][TILE+1];
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
            for (int d = threadIdx.x; d < dim; d += TILE)
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

template<unsigned int TILE>
__global__ static void kernelCheckCachePriorityV3(float * K, int * KCacheRowIdx, const float * x, const float * x2, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y)
        return;
    int row = d_cache_rows_to_compute[blockIdx.y];
    __shared__ float shsum[TILE][TILE+1];
    int block = blockDim.x * blockIdx.x;
    int i = KCacheRowIdx[row];
    //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
    //    printf("%d: i: %d, row: %d\n", blockIdx.y, i, row);

    //calculate cache matrix row [row], original index is [i]
    while (block < num_vec)
    {
        int j = block + threadIdx.y;
        int jout = block + threadIdx.x;
        float sum = 0;
        if (j < num_vec)
        {
            for (int d = threadIdx.x; d < dim; d += TILE)
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
            K[(size_t)num_vec_aligned * row + jout] = expf(-gamma * sum);
        }
        __syncthreads();
        block += gridDim.x * blockDim.x;
    }
}

//<4,4>
//block dim: 32 x number of warps
template<int TILE_X, int TILE_Y, int NUM_WARPS>
__global__ static void kernelCheckCachePriorityV4(float * d_K, int * d_KCacheRowIdx, const float * d_x, const float * d_x2, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y * TILE_Y)
        return;
    int num_y = TILE_Y;
    if (d_num_cache_rows_to_compute < (blockIdx.y + 1) * TILE_Y)
        num_y = d_num_cache_rows_to_compute % TILE_Y;
    __shared__ float shOut[TILE_X * TILE_Y * NUM_WARPS];
    int row[TILE_Y];
    int i[TILE_Y];
    for (int y = 0; y < TILE_Y; y++)
    {
        row[y] = d_cache_rows_to_compute[blockIdx.y * TILE_Y + y];
        i[y] = d_KCacheRowIdx[row[y]];
    }
    //if (threadIdx.x + threadIdx.y == 0 && blockIdx.x == 0)
    //{
    //    for (int y = 0; y < TILE_Y; y++)
    //        printf("[%d] %c row[%d] = %d, i[%d] = %d\n", blockIdx.y, y < num_y ? ' ' : '!', y, row[y], y, i[y]);
    //}
    //__shared__ float shsum[TILE][TILE+1];
    int block = NUM_WARPS * TILE_X * blockIdx.x;
    //if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
    //    printf("%d: i: %d, row: %d\n", blockIdx.y, i, row);

    //calculate cache matrix row [row], original index is [i]
    while (block < num_vec)
    {
        //int j = block + threadIdx.y;
        //int jout = block + threadIdx.x;
        int j = block + threadIdx.y * TILE_X;
        float sum[TILE_Y][TILE_X] = {0};
        if (j + TILE_X - 1 < num_vec)
        {
            for (int d = threadIdx.x; d < dim; d += warpSize)
            {
#pragma unroll
                for (int y = 0; y < TILE_Y; y++)
#pragma unroll
                    for (int x = 0; x < TILE_X; x++)
                        sum[y][x] += d_x[dim_aligned * i[y] + d] * d_x[dim_aligned * (j + x) + d];
            }
        }
        else
        {
            for (int d = threadIdx.x; d < dim; d += warpSize)
            {
#pragma unroll
                for (int x = 0; x < TILE_X; x++)
                    if (j + x < num_vec)
#pragma unroll
                        for (int y = 0; y < TILE_Y; y++)
                            sum[y][x] += d_x[dim_aligned * i[y] + d] * d_x[dim_aligned * (j + x) + d];
            }
        }
#pragma unroll
        for (int y = 0; y < TILE_Y; y++)
#pragma unroll
            for (int x = 0; x < TILE_X; x++)
            {
                float s = warpReduceSum(sum[y][x]);
                if (threadIdx.x == 0 && j + x < num_vec && y < num_y)
                {
                    s = d_x2[i[y]] + d_x2[j + x] - 2 * s;
                    //d_K[(size_t)num_vec_aligned * row[y] + j + x] = expf(-gamma * s);
                    shOut[NUM_WARPS * TILE_X * y + TILE_X * threadIdx.y + x] = expf(-gamma * s);
                }
            }
        __syncthreads();
        for (int x = threadIdx.x; x < NUM_WARPS * TILE_X && block + threadIdx.x < num_vec; x += blockDim.x)
        {
            for (int y = threadIdx.y; y < num_y; y += blockDim.y)
            {
                d_K[(size_t)num_vec_aligned * row[y] + block + x] = shOut[NUM_WARPS * TILE_X * y + x];
            }
        }
        __syncthreads();

        block += gridDim.x * blockDim.y * TILE_X;
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

template<unsigned int TILE>
__global__ static void kernelCheckCacheSparsePriorityV2(const int * i_ptr, float * K, int * KCacheRemapIdx, int * KCacheRowIdx, int * KCacheRowPriority, int cache_rows, const float * vec, const float * x2, csr_gpu x, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    int last = d_cacheRow;
    if (last < 0)
        return;
    __shared__ float shsum[TILE][TILE+1];
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
            for (int d = x.rowOffsets[j] + threadIdx.x; d < end; d += TILE)
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

template<int TILE_X, int TILE_Y, int NUM_WARPS>
__global__ static void kernelCheckCacheSparsePriorityV3(float * K, int * d_KCacheRowIdx, csr_gpu x, const float * d_x2, const float * vec, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y * TILE_Y)
        return;
    int num_y = TILE_Y;
    if (d_num_cache_rows_to_compute < (blockIdx.y + 1) * TILE_Y)
        num_y = d_num_cache_rows_to_compute % TILE_Y;
    __shared__ float shOut[TILE_X * TILE_Y * NUM_WARPS];
    int row[TILE_Y];
    int i[TILE_Y];
    for (int y = 0; y < TILE_Y; y++)
    {
        row[y] = d_cache_rows_to_compute[blockIdx.y * TILE_Y + y];
        i[y] = d_KCacheRowIdx[row[y]];
    }

    //int block = blockDim.x * blockIdx.x;
    int block = NUM_WARPS * TILE_X * blockIdx.x;
    while (block < num_vec)
    {
        //int j = block + threadIdx.y;
        int j = block + threadIdx.y * TILE_X;
        //int jout = block + threadIdx.x;
        float sum = 0;
        if (j < num_vec)
        {
            //int end = x.rowOffsets[j + 1];
            int end = x.rowOffsets[j] + x.rowLen[j];
            for (int d = x.rowOffsets[j] + threadIdx.x; d < end; d += warpSize)
            {
                sum += vec[dim_aligned * blockIdx.y + x.colInd[d]] * x.values[d];
            }
        }
        sum = warpReduceSum(sum);
        if (threadIdx.x == 0 && j < num_vec)
        {
            sum = d_x2[i[0]] + d_x2[j] - 2 * sum;
            //d_K[(size_t)num_vec_aligned * row[y] + j + x] = expf(-gamma * s);
            //shOut[NUM_WARPS * y + threadIdx.y + x] = expf(-gamma * s);
            K[(size_t)num_vec_aligned * row[0] + j] = expf(-gamma * sum);
        }
        /*if (threadIdx.y == 0 && jout < num_vec)
        {
            sum = x2[i[0]] + x2[jout] - 2 * shsum[threadIdx.x][0];
            K[(size_t)num_vec_aligned * last + jout] = expf(-gamma * sum);
        }*/
        __syncthreads();
        //block += gridDim.x * blockDim.x;
        block += gridDim.x * blockDim.y * TILE_X;
    }
}

__global__ static void kernelCheckCacheSparsePriorityV3(float * K, int * d_KCacheRowIdx, ellpack_gpu x, const float * d_x2, const float * vec, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y)
        return;
    //__shared__ float shOut[TILE_X * TILE_Y * NUM_WARPS];
    int row = d_cache_rows_to_compute[blockIdx.y];
    int i = d_KCacheRowIdx[row];
    float x2i = d_x2[i];

    int block = blockDim.x * blockIdx.x;
    while (block < num_vec)
    {
        //int j = block + threadIdx.y;
        int r = block + threadIdx.x;
        //int jout = block + threadIdx.x;
        if (r < num_vec)
        {
            float sum = 0;
            //int end = x.rowOffsets[j + 1];
            int rowLen = x.rowLen[r];
            for (int d = 0; d < rowLen; d++)
            {
                sum += vec[dim_aligned * blockIdx.y + x.colInd[num_vec * d + r]] * x.values[num_vec * d + r];
            }
            sum = x2i + d_x2[r] - 2 * sum;
            K[(size_t)num_vec_aligned * row + r] = expf(-gamma * sum);
        }
        /*sum = warpReduceSum(sum);
        if (threadIdx.x == 0 && j < num_vec)
        {
            sum = d_x2[i[0]] + d_x2[j] - 2 * sum;
            //d_K[(size_t)num_vec_aligned * row[y] + j + x] = expf(-gamma * s);
            //shOut[NUM_WARPS * y + threadIdx.y + x] = expf(-gamma * s);
            K[(size_t)num_vec_aligned * row[0] + j] = expf(-gamma * sum);
        }*/
        /*if (threadIdx.y == 0 && jout < num_vec)
        {
            sum = x2[i[0]] + x2[jout] - 2 * shsum[threadIdx.x][0];
            K[(size_t)num_vec_aligned * last + jout] = expf(-gamma * sum);
        }*/
        //__syncthreads();
        block += gridDim.x * blockDim.x;
        //block += gridDim.x * blockDim.y * TILE_X;
    }
}

__global__ static void kernelCheckCacheSparsePriorityV3(float * K, int * d_KCacheRowIdx, jds_gpu x, const float * d_x2, const float * vec, float gamma, int num_vec, int num_vec_aligned, int dim, int dim_aligned)
{
    if (d_num_cache_rows_to_compute <= blockIdx.y)
        return;
    int row = d_cache_rows_to_compute[blockIdx.y];
    int i = d_KCacheRowIdx[row];
    float x2i = d_x2[i];

    int block = blockDim.x * blockIdx.x;
    while (block < num_vec)
    {
        int r = block + threadIdx.x;
        if (r < num_vec)
        {
            float sum = 0;
            int rowLen = x.rowLen[r];
            for (int d = 0; d < rowLen; d++)
            {
                int i = x.colStart[d] + r;
                sum += vec[dim_aligned * blockIdx.y + x.colInd[i]] * x.values[i];
            }
            sum = x2i + d_x2[x.rowPerm[r]] - 2 * sum;
            K[(size_t)num_vec_aligned * row + x.rowPerm[r]] = expf(-gamma * sum);
        }
        block += gridDim.x * blockDim.x;
    }
}

//for DP
template<unsigned int findCacheRowBlockSize, unsigned int WS>
__global__ static void kernelCheckCacheCublasKLocal(int * d_workingset, float * d_x, float * d_xT, float * d_xTile, const float * d_x2, float * d_K, float * d_KTile, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, const float * d_alphadiff, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    kernelFindCacheRowKLocal<findCacheRowBlockSize><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    cudaDeviceSynchronize();
    //printf("d_num_cache_rows_to_compute = %d\n", d_num_cache_rows_to_compute);

    const int TILE = 16;
    dim3 dimBlock(TILE, TILE);
    dim3 dimGrid(getgriddim<size_t>(dim, dimBlock.x), getgriddim<size_t>(d_num_cache_rows_to_compute, dimBlock.y));
    kernelCopyXTileT<WS><<<dimGrid, dimBlock>>>(d_xTile, d_x, d_KCacheRowIdx, dim, dim_aligned, num_vec, num_vec_aligned);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, num_vec, d_num_cache_rows_to_compute, dim, &d_aux_one, d_xT, num_vec_aligned, d_xTile, WS, &d_aux_zero, d_KTile, num_vec_aligned);

    dimGrid.x = getgriddim<size_t>(num_vec, dimBlock.x);
    kernelCublasFinalize<<<dimGrid, dimBlock>>>(d_K, d_KTile, d_x2, d_KCacheRowIdx, num_vec, num_vec_aligned, gamma);

    cudaDeviceSynchronize();
    cublasDestroy(cublas);
}

template<unsigned int findCacheRowBlockSize, unsigned int WS>
void checkCache(bool sparse, const int * d_i, float * d_x, const float * d_x2, const csr_gpu & sparse_data_gpu, float * d_K, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    dim3 dimBlockCache(256);
    dim3 dimGridCache(getgriddim<int>(num_vec, dimBlockCache.x));
    size_t kernelCheckCacheSMSize = dimBlockCache.x * sizeof(int2);
    const int TILE = 16;
    if (sparse)
    {
        assert_cuda(cudaMemset(d_denseVec, 0, dim * sizeof(float)));
        dim3 dimBlock(256);
        dim3 dimGrid(std::min(64, getgriddim<int>(dim, dimBlock.x)));
        kernelMakeDenseVec<<<dimGrid, dimBlock>>>(d_i, d_KCacheRemapIdx, sparse_data_gpu, d_denseVec);
        kernelFindCacheRow<<<1, 256, 256 * sizeof(int2)>>>(d_i, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
        dimBlock = dim3(TILE, TILE);
        dimGrid = dim3(std::min(256, getgriddim<int>(num_vec, dimBlock.y)));
        kernelCheckCacheSparsePriorityV2<TILE><<<dimGrid, dimBlock>>>(d_i, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_denseVec, d_x2, sparse_data_gpu, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
    else
    {
        kernelCheckCacheSMSize = std::max(kernelCheckCacheSMSize, dim * sizeof(float));
        kernelFindCacheRow<<<1, 256, 256 * sizeof(int2)>>>(d_i, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
        dim3 dimBlock(TILE, TILE);
        dim3 dimGrid(std::min(256, getgriddim<int>(num_vec, dimBlock.y)));
        kernelCheckCachePriorityV2<TILE><<<dimGrid, dimBlock>>>(d_i, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_x, d_x2, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
    kernelCheckCacheFinalize<<<1, 1>>>(d_KCacheRemapIdx, d_KCacheRowPriority);
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheV2(bool sparse, const int * d_workingset, float * d_x, const float * d_x2, const csr_gpu & csr_data_gpu, const ellpack_gpu & ellpack_data_gpu, const jds_gpu & jds_data_gpu, bool use_ellpack, float * d_K, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowNV2<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    else
        kernelFindCacheRowV2<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    if (sparse)
    {
        assert_cuda(cudaMemset(d_denseVec, 0, dim_aligned * WS * sizeof(float)));
        dim3 dimBlock(256);
        if (use_ellpack)
        {
            dim3 dimGrid(getgriddim<int>(WS, dimBlock.x));
            kernelMakeDenseVecWS<WS> << <dimGrid, dimBlock >> >(d_KCacheRowIdx, ellpack_data_gpu, d_denseVec, dim_aligned);
            dimGrid = dim3(std::min(64, getgriddim<int>(num_vec, dimBlock.x)), WS);
            kernelCheckCacheSparsePriorityV3 << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, ellpack_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
        }
        else
        {
            dim3 dimGrid(std::min(64, getgriddim<int>(dim, dimBlock.x)), WS);
            kernelMakeDenseVecWS<WS> << <dimGrid, dimBlock >> >(d_KCacheRowIdx, csr_data_gpu, d_denseVec, dim_aligned);
            //const int TILE_X = 1;
            //const int TILE_Y = 1;
            //const int NUM_WARPS = 8;
            //dimBlock = dim3(32, NUM_WARPS);
            //dimGrid = dim3(std::min(256, getgriddim<int>(num_vec, NUM_WARPS * TILE_X)), WS / TILE_Y);
            //kernelCheckCacheSparsePriorityV3<TILE_X, TILE_Y, NUM_WARPS> << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, c_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
            dimGrid = dim3(std::min(64, getgriddim<int>(num_vec, dimBlock.x)), WS);
            kernelCheckCacheSparsePriorityV3 << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, jds_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
        }
    }
    else
    {
        const int TILE_X = 4;
        const int TILE_Y = 8;
        const int NUM_WARPS = 4;
        dim3 dimBlock(32, NUM_WARPS);
        dim3 dimGrid(std::min(256, getgriddim<int>(num_vec, NUM_WARPS * TILE_X)), WS / TILE_Y);
        kernelCheckCachePriorityV4<TILE_X, TILE_Y, NUM_WARPS><<<dimGrid, dimBlock>>>(d_K, d_KCacheRowIdx, d_x, d_x2, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheKLocal(bool sparse, const int * d_workingset, float * d_x, const float * d_x2, const csr_gpu & csr_data_gpu, const ellpack_gpu & ellpack_data_gpu, const jds_gpu & jds_data_gpu, bool use_ellpack, float * d_K, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_alphadiff, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowKLocalN<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    else
        kernelFindCacheRowKLocal<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    if (sparse)
    {
        assert_cuda(cudaMemset(d_denseVec, 0, dim_aligned * WS * sizeof(float)));
        dim3 dimBlock(256);
        if (use_ellpack)
        {
            dim3 dimGrid(getgriddim<int>(WS, dimBlock.x));
            kernelMakeDenseVecWS<WS> << <dimGrid, dimBlock >> >(d_KCacheRowIdx, ellpack_data_gpu, d_denseVec, dim_aligned);
            dimGrid = dim3(std::min(64, getgriddim<int>(num_vec, dimBlock.x)), WS);
            kernelCheckCacheSparsePriorityV3 << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, ellpack_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
        }
        else
        {
            dim3 dimGrid(std::min(64, getgriddim<int>(dim, dimBlock.x)), WS);
            kernelMakeDenseVecWS<WS> << <dimGrid, dimBlock >> >(d_KCacheRowIdx, csr_data_gpu, d_denseVec, dim_aligned);
            //const int TILE_X = 1;
            //const int TILE_Y = 1;
            //const int NUM_WARPS = 8;
            //dimBlock = dim3(32, NUM_WARPS);
            //dimGrid = dim3(std::min(256, getgriddim<int>(num_vec, NUM_WARPS * TILE_X)), WS / TILE_Y);
            //kernelCheckCacheSparsePriorityV3<TILE_X, TILE_Y, NUM_WARPS> << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, csr_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
            dimGrid = dim3(std::min(64, getgriddim<int>(num_vec, dimBlock.x)), WS);
            kernelCheckCacheSparsePriorityV3 << <dimGrid, dimBlock >> >(d_K, d_KCacheRowIdx, jds_data_gpu, d_x2, d_denseVec, gamma, num_vec, num_vec_aligned, dim, dim_aligned);

        }
    }
    else
    {
        const int TILE_X = 4;
        const int TILE_Y = 8;
        const int NUM_WARPS = 4;
        dim3 dimBlock(32, NUM_WARPS);
        dim3 dimGrid(std::min(256, getgriddim<int>(num_vec, NUM_WARPS * TILE_X)), WS / TILE_Y);
        kernelCheckCachePriorityV4<TILE_X, TILE_Y, NUM_WARPS><<<dimGrid, dimBlock>>>(d_K, d_KCacheRowIdx, d_x, d_x2, gamma, num_vec, num_vec_aligned, dim, dim_aligned);
    }
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheCublas(bool sparse, const int * d_workingset, float * d_x, float * d_xT, float * d_xTile, const float * d_x2, const csr_gpu & sparse_data_gpu, float * d_K, float * d_KTile, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma, cublasHandle_t cublas)
{
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowNV2<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    else
        kernelFindCacheRowV2<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows);
    //import_cuda_buffer(d_workingset, WS, 1, sizeof(int), "working_set.dat");
    //import_cuda_buffer(d_KCacheRemapIdx, num_vec, 1, sizeof(int), "KCacheRemapIdx.dat");
    //import_cuda_buffer(d_KCacheRowIdx, cache_rows, 1, sizeof(int), "KCacheRowIdx.dat");
    //import_cuda_buffer(d_KCacheRowPriority, cache_rows, 1, sizeof(int), "KCacheRowPriority.dat");

    const int TILE = 16;
    dim3 dimBlock(TILE, TILE);
    dim3 dimGrid(getgriddim<size_t>(dim, dimBlock.x), getgriddim<size_t>(WS, dimBlock.y));
    kernelCopyXTileT<WS, TILE><<<dimGrid, dimBlock>>>(d_xTile, d_x, d_KCacheRowIdx, dim, dim_aligned, num_vec, num_vec_aligned);

    float alpha = 1,
          beta = 0;
    assert_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, num_vec, WS, dim, &alpha, d_xT, num_vec_aligned, d_xTile, WS, &beta, d_KTile, num_vec_aligned));

    dimGrid.x = getgriddim<size_t>(num_vec, dimBlock.x);
    kernelCublasFinalize<<<dimGrid, dimBlock>>>(d_K, d_KTile, d_x2, d_KCacheRowIdx, num_vec, num_vec_aligned, gamma);
}

template<unsigned int findCacheRowBlockSize, unsigned int WS, unsigned int ELEM_PER_THREAD>
void checkCacheCublasKLocal(bool sparse, const int * d_workingset, float * d_x, float * d_xT, float * d_xTile, const float * d_x2, const csr_gpu & sparse_data_gpu, float * d_K, float * d_KTile, int * d_KCacheRemapIdx, int * d_KCacheRowIdx, int * d_KCacheRowPriority, const float * d_alphadiff, float * d_denseVec, int num_vec, int num_vec_aligned, int dim, int dim_aligned, int cache_rows, float gamma, cublasHandle_t cublas)
{
    //kernelCheckCacheCublasKLocal<findCacheRowBlockSize><<<1, 1>>>(d_workingset, d_x, d_xT, d_xTile, d_x2, d_K, d_KTile, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_alphadiff, num_vec, num_vec_aligned, dim, dim_aligned, cache_rows, gamma);
    if (ELEM_PER_THREAD > 1)
        kernelFindCacheRowKLocalN<findCacheRowBlockSize, WS, ELEM_PER_THREAD><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    else
        kernelFindCacheRowKLocal<findCacheRowBlockSize, WS><<<1, findCacheRowBlockSize>>>(d_workingset, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, cache_rows, d_alphadiff);
    int num_cache_rows_to_compute;
    assert_cuda(cudaMemcpyFromSymbol(&num_cache_rows_to_compute, d_num_cache_rows_to_compute, sizeof(int)));
    if (num_cache_rows_to_compute <= 0)
        return;

    const int TILE = 16;
    dim3 dimBlock(TILE, TILE);
    dim3 dimGrid(getgriddim<size_t>(dim, dimBlock.x), getgriddim<size_t>(num_cache_rows_to_compute, dimBlock.y));
    kernelCopyXTileT<WS, TILE><<<dimGrid, dimBlock>>>(d_xTile, d_x, d_KCacheRowIdx, dim, dim_aligned, num_vec, num_vec_aligned);

    float alpha = 1,
          beta = 0;
    assert_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, num_vec, num_cache_rows_to_compute, dim, &alpha, d_xT, num_vec_aligned, d_xTile, WS, &beta, d_KTile, num_vec_aligned));

    dimGrid.x = getgriddim<size_t>(num_vec, dimBlock.x);
    kernelCublasFinalize<<<dimGrid, dimBlock>>>(d_K, d_KTile, d_x2, d_KCacheRowIdx, num_vec, num_vec_aligned, gamma);
}
