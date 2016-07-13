#pragma once

//blockSize >= NC
template<unsigned int blockSize, unsigned int NC, unsigned int numSortBlocks>
__global__ static void kernelFindNBestV2(int * ws, const float * y, const float * g, const float * alpha, float C, int num_vec, int * ws_priority, float * aux_val, int * aux_idx, SYNC_BUFFER_DEF)
{
    //const int sharedSize = NC + blockSize;
    const int sharedSize = 2 * blockSize;
    __shared__ float shValI[sharedSize];
    __shared__ float shValJ[sharedSize];
    __shared__ int shIdxI[sharedSize];
    __shared__ int shIdxJ[sharedSize];
	//__shared__ float shValI[NC * numSortBlocks];
	//__shared__ float shValJ[NC * numSortBlocks];
	//__shared__ int shIdxI[NC * numSortBlocks];
	//__shared__ int shIdxJ[NC * numSortBlocks];
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

#ifdef USE_DAIFLETCHER
        if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
            shValI[threadIdx.x] = y_ - g_;
        else
            shValI[threadIdx.x] = -FLT_MAX;
        shIdxI[threadIdx.x] = k;

        if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
            shValJ[threadIdx.x] = y_ - g_;
        else
            shValJ[threadIdx.x] = -FLT_MAX;
        shIdxJ[threadIdx.x] = k;
#else
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
#endif

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
#ifdef USE_DAIFLETCHER
            if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
                v = y_ - g_;
            else
                v = -FLT_MAX;
#else
            if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
                v = y_ * g_;
            else
                v = -FLT_MAX;
#endif

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
#ifdef USE_DAIFLETCHER
            if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
                v = y_ - g_;
            else
                v = -FLT_MAX;
#else
            if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
                v = -y_ * g_;
            else
                v = -FLT_MAX;
#endif

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

    for (int i = threadIdx.x; i < NC; i += blockDim.x)
    {
        shIdxI[i] = aux_idx[i];
        shValI[i] = aux_val[i];
        shIdxJ[i] = aux_idx[i + NC * gridDim.x];
        shValJ[i] = aux_val[i + NC * gridDim.x];
    }
    for (int k = 1; k < gridDim.x; k++)
    {
        for (int i = threadIdx.x; i < NC; i += blockDim.x)
        {
            shIdxI[i + NC] = aux_idx[i + NC * k];
            shValI[i + NC] = aux_val[i + NC * k];
            shIdxJ[i + NC] = aux_idx[i + NC * (k + gridDim.x)];
            shValJ[i + NC] = aux_val[i + NC * (k + gridDim.x)];
        }
        __syncthreads();

        blockBitonicSortN<true>(shIdxI, shValI, 2 * NC);
        blockBitonicSortN<true>(shIdxJ, shValJ, 2 * NC);
    }

    if (shValI[NC - 1] <= -FLT_MAX || shValJ[NC - 1] <= -FLT_MAX)
        printf("[Error] Not enough elements found in FindNBest\n");

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

//blockSize >= NC
template<unsigned int blockSize, unsigned int NC, unsigned int numSortBlocks>
__global__ static void kernelFindNBestV3(int * ws, const float * y, const float * g, const float * alpha, float C, int num_vec, int * ws_priority, float * aux_val, int * aux_idx, SYNC_BUFFER_DEF)
{
    __shared__ float shValI[NC + blockSize];
    __shared__ float shValJ[NC + blockSize];
    __shared__ int shIdxI[NC + blockSize];
    __shared__ int shIdxJ[NC + blockSize];
	//__shared__ float shValI[NC * numSortBlocks];
	//__shared__ float shValJ[NC * numSortBlocks];
	//__shared__ int shIdxI[NC * numSortBlocks];
	//__shared__ int shIdxJ[NC * numSortBlocks];
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

    for (int i = threadIdx.x; i < NC; i += blockDim.x)
    {
        shIdxI[i] = aux_idx[i];
        shValI[i] = aux_val[i];
        shIdxJ[i] = aux_idx[i + NC * gridDim.x];
        shValJ[i] = aux_val[i + NC * gridDim.x];
    }
    for (int k = 1; k < gridDim.x; k++)
    {
        for (int i = threadIdx.x; i < NC; i += blockDim.x)
        {
            shIdxI[i + NC] = aux_idx[i + NC * k];
            shValI[i + NC] = aux_val[i + NC * k];
            shIdxJ[i + NC] = aux_idx[i + NC * (k + gridDim.x)];
            shValJ[i + NC] = aux_val[i + NC * (k + gridDim.x)];
        }
        __syncthreads();

        blockBitonicSortN<true>(shIdxI, shValI, 2 * NC);
        blockBitonicSortN<true>(shIdxJ, shValJ, 2 * NC);
    }

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
