#pragma once

__global__ static void kernelPrepareSortI(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
            v = y_ * g[k];
        else
            v = -FLT_MAX;
        valbuf[k] = v;
        idxbuf[k] = k;
    }
}

__global__ static void kernelPrepareSortJ(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
            v = -y_ * g[k];
        else
            v = -FLT_MAX;
        valbuf[k] = v;
        idxbuf[k] = k;
    }
}

__global__ static void kernelPrepareSortJSecondOrder(float * valbuf, int * idxbuf, const float * y, const float * g, const float * alpha, float C, int num_vec, int num_vec_aligned, const int * i_ptr, const float * K, const float * KDiag, const int * KCacheRemapIdx)
{
    int i = *i_ptr;
    int cache_row = KCacheRemapIdx[i];
    float th = y[i] * g[i];

    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        if (((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C)) && th > y[k] * g[k])
        {
            float den = KDiag[i] + KDiag[k] - 2 * K[(size_t)num_vec_aligned * cache_row + k];
            float val = th - y[k] * g[k];
            v = val * val / den;
        }
        else
            v = -FLT_MAX;
        valbuf[k] = v;
        idxbuf[k] = k;
    }
}

#if 0
template<unsigned int blockSize, unsigned int NC>
__global__ static void kernelFindNBest(int * ws, const float * y, const float * g, const float * alpha, float C, int num_vec, int * ws_priority, float * aux_val, int * aux_idx, SYNC_BUFFER_DEF)
{
    __shared__ float shValI[blockSize * NC];
    __shared__ float shValJ[blockSize * NC];
    __shared__ int shIdxI[blockSize * NC];
    __shared__ int shIdxJ[blockSize * NC];
    float vmaxI = -FLT_MAX,
          vmaxJ = -FLT_MAX;
    int imaxI = 0,
        imaxJ = 0;
    for (int i = 0; i < NC; i++)
    {
        shValI[threadIdx.x + blockSize * i] = -FLT_MAX;
        shValJ[threadIdx.x + blockSize * i] = -FLT_MAX;
        shIdxI[threadIdx.x + blockSize * i] = 0;
        shIdxJ[threadIdx.x + blockSize * i] = 0;
    }
    __syncthreads();

    //main loop
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        float g_ = g[k];
        if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
            v = y_ * g_;
        else
            v = -FLT_MAX;
        int ik = k;

        for (int i = 0; i < NC; i++)
        {
            float vI = shValI[blockSize * i + threadIdx.x];
            int iI = shIdxI[blockSize * i + threadIdx.x];
            if (v > vI)
            {
                swap_dev(v, vI);
                swap_dev(ik, iI);
            }
            shValI[blockSize * i + threadIdx.x] = vI;
            shIdxI[blockSize * i + threadIdx.x] = iI;
        }
        if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
            v = -y_ * g_;
        else
            v = -FLT_MAX;
        int jk = k;

        for (int i = 0; i < NC; i++)
        {
            float vJ = shValJ[blockSize * i + threadIdx.x];
            int iJ = shIdxJ[blockSize * i + threadIdx.x];
            if (v > vJ)
            {
                swap_dev(v, vJ);
                swap_dev(jk, iJ);
            }
            shValJ[blockSize * i + threadIdx.x] = vJ;
            shIdxJ[blockSize * i + threadIdx.x] = iJ;
        }
        __syncthreads();
    }

    blockBitonicSortN<true>(shIdxI, shValI, blockSize * NC);
    blockBitonicSortN<true>(shIdxJ, shValJ, blockSize * NC);

    for (int i = threadIdx.x; i < NC; i += blockDim.x)
    {
        int gi = i + NC * blockIdx.x;
        int gj = gi + NC * gridDim.x;
        aux_idx[gi] = shIdxI[i];
        aux_val[gi] = shValI[i];
        aux_idx[gj] = shIdxJ[i];
        aux_val[gj] = shValJ[i];
    }

    WAIT_FOR_THE_FINAL_BLOCK;

    for (int i = threadIdx.x; i < NC * gridDim.x; i += blockDim.x)
    {
        shIdxI[i] = aux_idx[i];
        shValI[i] = aux_val[i];
        shIdxJ[i] = aux_idx[i + NC * gridDim.x];
        shValJ[i] = aux_val[i + NC * gridDim.x];
    }
    __syncthreads();

	blockBitonicSortN<true>(shIdxI, shValI, gridDim.x * NC);
	blockBitonicSortN<true>(shIdxJ, shValJ, gridDim.x * NC);

    for (int i = threadIdx.x; i < NC; i += blockDim.x)
    {
        //int wsi = ws[i] = shIdxI[i];
        //int wsj = ws[i + NC] = shIdxJ[i];
        int wsi = aux_idx[i] = shIdxI[i];
        int wsj = aux_idx[i + NC] = shIdxJ[i];
        ws_priority[wsi] = INT_MAX;
        ws_priority[wsj] = INT_MAX;
    }
}

#else

//blockSize >= NC, numSortBlocks * NC > NC + blockSize
template<unsigned int blockSize, unsigned int NC, unsigned int numSortBlocks>
__global__ static void kernelFindNBest(int * ws, const float * y, const float * g, const float * alpha, float C, int num_vec, int * ws_priority, float * aux_val, int * aux_idx, SYNC_BUFFER_DEF)
{
	//__shared__ float shValI[NC + blockSize];
	//__shared__ float shValJ[NC + blockSize];
	//__shared__ int shIdxI[NC + blockSize];
	//__shared__ int shIdxJ[NC + blockSize];
	__shared__ float shValI[NC * numSortBlocks];
	__shared__ float shValJ[NC * numSortBlocks];
	__shared__ int shIdxI[NC * numSortBlocks];
	__shared__ int shIdxJ[NC * numSortBlocks];
	__shared__ int shTmpNum;

	float vmaxI = -FLT_MAX,
		vmaxJ = -FLT_MAX;
	int imaxI = 0,
		imaxJ = 0;
	//for (int i = threadIdx.x; i < NC; i+=blockDim.x)
	//{
	//	shValI[i] = -FLT_MAX;
	//	shValJ[i] = -FLT_MAX;
	//	shIdxI[i] = 0;
	//	shIdxJ[i] = 0;
	//}
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	float v, y_, a_, g_;
    if (k < num_vec)
    {
        float y_ = y[k];
        float a_ = alpha[k];
        float g_ = g[k];

        if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
            shValI[threadIdx.x] = y_ * g_;
        else
            shValI[threadIdx.x] = -FLT_MAX;
        shIdxI[threadIdx.x] = k;

        if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
            shValJ[threadIdx.x] = -y_ * g_;
        else
            shValJ[threadIdx.x] = -FLT_MAX;
        shIdxJ[threadIdx.x] = k;

        if(threadIdx.x == 0) shTmpNum = 0;
    }
    else
    {
        shValI[threadIdx.x] = -FLT_MAX;
        shValJ[threadIdx.x] = -FLT_MAX;
    }
    __syncthreads();
    blockBitonicSort<true>(shIdxI, shValI);
    blockBitonicSort<true>(shIdxJ, shValJ);

	//main loop
	for (int koffset = gridDim.x * blockDim.x + blockDim.x * blockIdx.x; koffset < num_vec; koffset += gridDim.x * blockDim.x)
	{
        k = koffset + threadIdx.x;
        if (k < num_vec)
        {
            y_ = y[k];
            a_ = alpha[k];
            g_ = g[k];

            /////////////////////////
            //I part
            if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
                v = y_ * g_;
            else
                v = -FLT_MAX;

            //atomicInc and add new value
            if (v > shValI[NC - 1]) {
                int id = atomicAdd(&shTmpNum, 1);
                shValI[NC+id] = v;
                shIdxI[NC+id] = k;
            }
        }
		__syncthreads();


		//sort new-ones
		if (shTmpNum > 0) {
            int sortSize = NC * 2;
            while (shTmpNum + NC > sortSize)
                sortSize *= 2;
            if (threadIdx.x < sortSize - (shTmpNum + NC))
                shValI[shTmpNum + NC + threadIdx.x] = -FLT_MAX;
            __syncthreads();
			blockBitonicSortN<true>(shIdxI, shValI, sortSize);
			//blockBitonicSortN<true>(shIdxI, shValI, shTmpNum + NC);
			if (threadIdx.x == 0) shTmpNum = 0;
		}
		__syncthreads();
		
        if (k < num_vec)
        {
            /////////////////////////
            //J part
            if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
                v = -y_ * g_;
            else
                v = -FLT_MAX;

            //atomicInc and add new value
            if (v > shValJ[NC - 1]) {
                int id = atomicAdd(&shTmpNum, 1);
                shValJ[NC + id] = v;
                shIdxJ[NC + id] = k;
            }
        }
		__syncthreads();

		//sort new-ones
		if (shTmpNum > 0) {
            int sortSize = NC * 2;
            while (shTmpNum + NC > sortSize)
                sortSize *= 2;
            if (threadIdx.x < sortSize - (shTmpNum + NC))
                shValJ[shTmpNum + NC + threadIdx.x] = -FLT_MAX;
            __syncthreads();
			blockBitonicSortN<true>(shIdxJ, shValJ, sortSize);
			//blockBitonicSortN<true>(shIdxJ, shValJ, shTmpNum + NC);
			if (threadIdx.x == 0) shTmpNum = 0;
		}
		__syncthreads();
	}

	for (int i = threadIdx.x; i < NC; i += blockDim.x)
	{
		int gi = i + NC * blockIdx.x;
		int gj = gi + NC * gridDim.x;
		aux_idx[gi] = shIdxI[i];
		aux_val[gi] = shValI[i];
		aux_idx[gj] = shIdxJ[i];
		aux_val[gj] = shValJ[i];
	}
	
	WAIT_FOR_THE_FINAL_BLOCK;

	for (int i = threadIdx.x; i < NC * gridDim.x; i += blockDim.x)
	{
		shIdxI[i] = aux_idx[i];
		shValI[i] = aux_val[i];
		shIdxJ[i] = aux_idx[i + NC * gridDim.x];
		shValJ[i] = aux_val[i + NC * gridDim.x];
	}
	__syncthreads();

	blockBitonicSortN<true>(shIdxI, shValI, gridDim.x * NC);
	blockBitonicSortN<true>(shIdxJ, shValJ, gridDim.x * NC);

	for (int i = threadIdx.x; i < NC; i += blockDim.x)
	{
		//int wsi = ws[i] = shIdxI[i];
		//int wsj = ws[i + NC] = shIdxJ[i];
		int wsi = aux_idx[i] = shIdxI[i];
		int wsj = aux_idx[i + NC] = shIdxJ[i];
		ws_priority[wsi] = INT_MAX; //illegal __global__ write of size 4, thread (15,0,0), block (14,0,0), probably fixed
		ws_priority[wsj] = INT_MAX;
	}
}

#endif

template<unsigned int WS, unsigned int NC>
__global__ static void kernelFillWorkingSet(int * ws, const float * alpha, float C, int * ws_priority, int * new_ws)
{
    __shared__ int shWS[WS]; //old working set
    __shared__ int shNewWS[WS]; //new working set
    __shared__ int shWSPriority[WS];
    __shared__ int shNewWSPriority[WS];
    __shared__ float shAlpha[WS];

    for (int k = threadIdx.x; k < WS; k += blockDim.x)
    {
        int i = shWS[k] = ws[k];
        shWSPriority[k] = ws_priority[i];
        shAlpha[k] = alpha[i];
    }
    for (int k = threadIdx.x; k < NC * 2; k += blockDim.x)
    {
        shNewWS[k] = new_ws[k];
        shNewWSPriority[k] = 0;
    }
    __syncthreads();
    
    blockBitonicSort<false>(shWS, shWSPriority);

    int n = NC * 2;
    if (threadIdx.x == 0)
    {
        //free
        for (int i = 0; i < WS && n < WS; i++)
        {
            if (shAlpha[i] > 0 && shAlpha[i] < C && shWSPriority[i] < INT_MAX)
            {
                shNewWSPriority[n] = shWSPriority[i] + 1;
                shNewWS[n++] = shWS[i];
            }
        }
        //lower bound
        for (int i = 0; i < WS && n < WS; i++)
        {
            if (shAlpha[i] <= 0 && shWSPriority[i] < INT_MAX)
            {
                shNewWSPriority[n] = shWSPriority[i] + 1;
                shNewWS[n++] = shWS[i];
            }
        }
        //upper bound
        for (int i = 0; i < WS && n < WS; i++)
        {
            if (shAlpha[i] >= C && shWSPriority[i] < INT_MAX)
            {
                shNewWSPriority[n] = shWSPriority[i] + 1;
                shNewWS[n++] = shWS[i];
            }
        }
        if (n < WS)
            printf("[Error] Not enough elements to fill working set, this should never happen\n");
    }
    __syncthreads();

    for (int k = threadIdx.x; k < WS; k += blockDim.x)
    {
        int i = shNewWS[k];
        ws[k] = i;
        ws_priority[i] = shNewWSPriority[k];
    }
}

template<unsigned int blockSize, unsigned int WS>
__global__ static void kernelFindActiveSet(int * ws, const float * y, const float * g, const float * alpha, float C, int num_vec)
{
    __shared__ float shValI[blockSize];
    __shared__ float shValJ[blockSize];
    __shared__ int shIdxI[blockSize];
    __shared__ int shIdxJ[blockSize];
    float vmaxI = -FLT_MAX,
          vmaxJ = -FLT_MAX;
    int imaxI = 0,
        imaxJ = 0;
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
    {
        float v;
        float y_ = y[k];
        float a_ = alpha[k];
        float g_ = g[k];
        if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
            v = y_ * g_;
        else
            v = -FLT_MAX;
        if (v > vmaxI)
        {
            vmaxI = v;
            imaxI = k;
        }
        if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
            v = -y_ * g_;
        else
            v = -FLT_MAX;
        if (v > vmaxJ)
        {
            vmaxJ = v;
            imaxJ = k;
        }
    }
    shValI[threadIdx.x] = vmaxI;
    shIdxI[threadIdx.x] = imaxI;
    shValJ[threadIdx.x] = vmaxJ;
    shIdxJ[threadIdx.x] = imaxJ;
    __syncthreads();
    //blockBitonicSort<true>(shIdxI, shValI, blockSize);
    //blockBitonicSort<true>(shIdxJ, shValJ, blockSize);
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            if (shValI[threadIdx.x + s] > shValI[threadIdx.x])
            {
                //shValI[threadIdx.x] = shValI[threadIdx.x + s];
                //shIdxI[threadIdx.x] = shIdxI[threadIdx.x + s];
                swap_dev(shValI[threadIdx.x], shValI[threadIdx.x + s]);
                swap_dev(shIdxI[threadIdx.x], shIdxI[threadIdx.x + s]);
            }
            if (shValJ[threadIdx.x + s] > shValJ[threadIdx.x])
            {
                //shValJ[threadIdx.x] = shValJ[threadIdx.x + s];
                //shIdxJ[threadIdx.x] = shIdxJ[threadIdx.x + s];
                swap_dev(shValJ[threadIdx.x], shValJ[threadIdx.x + s]);
                swap_dev(shIdxJ[threadIdx.x], shIdxJ[threadIdx.x + s]);
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        int oidx = blockIdx.x;
        int shidx = 0;
        while (oidx < WS / 2)
        {
            ws[oidx] = shIdxI[shidx];
            ws[oidx + WS / 2] = shIdxJ[shidx];
            oidx += gridDim.x;
            ++shidx;
        }
        //printf("Block %d output loops %d\n", blockIdx.x, shidx);
    }
}

//WS*numSortBlocks / 2  must be greather than blockSize
template<unsigned int blockSize, unsigned int WS, unsigned int numSortBlocks>
__global__ static void kernelFindActiveSetV2(int * ws, const float * y, const float * g, const float * alpha, float C, int num_vec, volatile float *sortValues, volatile int *sortIdxs, SYNC_BUFFER_DEF)
{
	__shared__ float shValI[WS*numSortBlocks / 2];
	__shared__ float shValJ[WS*numSortBlocks / 2];
	__shared__ int shIdxI[WS*numSortBlocks / 2];
	__shared__ int shIdxJ[WS*numSortBlocks / 2];
	float vmaxI = -FLT_MAX,
		vmaxJ = -FLT_MAX;
	int imaxI = 0,
		imaxJ = 0;
	for (int k = blockDim.x * blockIdx.x + threadIdx.x; k < num_vec; k += gridDim.x * blockDim.x)
	{
		float v;
		float y_ = y[k];
		float a_ = alpha[k];
		float g_ = g[k];
		if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
			v = y_ * g_;
		else
			v = -FLT_MAX;
		if (v > vmaxI)
		{
			vmaxI = v;
			imaxI = k;
		}
		if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
			v = -y_ * g_;
		else
			v = -FLT_MAX;
		if (v > vmaxJ)
		{
			vmaxJ = v;
			imaxJ = k;
		}
	}
	shValI[threadIdx.x] = vmaxI;
	shIdxI[threadIdx.x] = imaxI;
	shValJ[threadIdx.x] = vmaxJ;
	shIdxJ[threadIdx.x] = imaxJ;
	__syncthreads();
	//blockBitonicSort<true>(shIdxI, shValI, blockSize);
	//blockBitonicSort<true>(shIdxJ, shValJ, blockSize);
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			if (shValI[threadIdx.x + s] > shValI[threadIdx.x])
			{
				swap_dev(shValI[threadIdx.x], shValI[threadIdx.x + s]);
				swap_dev(shIdxI[threadIdx.x], shIdxI[threadIdx.x + s]);
			}
			if (shValJ[threadIdx.x + s] > shValJ[threadIdx.x])
			{
				swap_dev(shValJ[threadIdx.x], shValJ[threadIdx.x + s]);
				swap_dev(shIdxJ[threadIdx.x], shIdxJ[threadIdx.x + s]);
			}
		}
		__syncthreads();
	}
	if (threadIdx.x < WS/2)
	{
		sortValues[blockIdx.x * WS / 2 + threadIdx.x] = shValI[threadIdx.x];
		sortValues[(gridDim.x + blockIdx.x) * WS / 2 + threadIdx.x] = shValJ[threadIdx.x];
		sortIdxs[blockIdx.x * WS / 2 + threadIdx.x] = shIdxI[threadIdx.x];
		sortIdxs[(gridDim.x + blockIdx.x) * WS / 2 + threadIdx.x] = shIdxJ[threadIdx.x];
	}
	
	WAIT_FOR_THE_FINAL_BLOCK; //global synchronization, last block continue, others return
	
	//copy data from all the blocks
	for (int k = threadIdx.x; k < numSortBlocks * WS / 2; k += blockSize) {
		shValI[k] = sortValues[k];
		shValJ[k] = sortValues[numSortBlocks * WS / 2 + k];
		shIdxI[k] = sortIdxs[k];
		shIdxJ[k] = sortIdxs[numSortBlocks * WS / 2 + k];
	}
	__syncthreads();

	//cross-block sort:
	for (int s = numSortBlocks / 2; s > 0; s >>= 1)
	{
		for (int k = threadIdx.x; k < WS/2 * s; k += blockSize)
		{
			if (shValI[k + s*WS/2] > shValI[k])
			{
				swap_dev(shValI[k], shValI[k + s * WS / 2]);
				swap_dev(shIdxI[k], shIdxI[k + s * WS / 2]);
			}
			if (shValJ[k + s*WS / 2] > shValJ[k])
			{
				swap_dev(shValJ[k], shValJ[k + s * WS / 2]);
				swap_dev(shIdxJ[k], shIdxJ[k + s * WS / 2]);
			}
		}
		__syncthreads();
	}

	//inter-block sort:
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		int ix = threadIdx.x % s;
		int iy = threadIdx.x / s;
		int offset = iy * WS / 2 + ix;
		for (int k = iy; k < numSortBlocks; k += blockSize / s)
		{
			
			if (shValI[offset + s] > shValI[offset])
			{
				swap_dev(shValI[offset], shValI[offset + s]);
				swap_dev(shIdxI[offset], shIdxI[offset + s]);
			}
			if (shValJ[offset + s] > shValJ[offset])
			{
				swap_dev(shValJ[offset], shValJ[offset + s]);
				swap_dev(shIdxJ[offset], shIdxJ[offset + s]);
			}
		}
		__syncthreads();
	}

	//store N-bests
	for (int k = threadIdx.x; k < WS / 2; k += blockSize) {
		int ix = threadIdx.x % numSortBlocks;
		int iy = threadIdx.x / numSortBlocks;
		ws[k] = shIdxI[ix * WS / 2 + iy];
		ws[k + WS / 2] = shIdxJ[ix * WS / 2 + iy];
	}
	
}
