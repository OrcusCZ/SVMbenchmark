#pragma once

//dimBlock(32, 8)
template<unsigned int WS, bool secondOrder, bool updateAllG, bool calcK>
__global__ static void kernelSMO1Block(const float * x, const float * x2, const float * y, float * g, float * alpha, const int * ws, float gamma, float C, float eps, int num_vec, int num_vec_aligned, int dim, int dim_aligned, float * K, int * KCacheRemapIdx)
{
    __shared__ float shK[WS][WS];
    __shared__ float shX[WS][32+1];
    __shared__ float shX2[WS][32+1];
    __shared__ int shWS[WS];
    __shared__ float shY[WS];
    __shared__ float shG[WS];
    __shared__ float shAlpha[WS];
    __shared__ float shVal[32];
    __shared__ float shIdx[32];
    __shared__ float shLambda;
    __shared__ float shGUpdate[WS];
    //float eps2;

    //clear K
    for (int i = threadIdx.x; i < WS; i += blockDim.x)
        for (int j = threadIdx.y; j < WS; j += blockDim.y)
            shK[j][i] = 0;
    //copy buffers to shared memory
    if (threadIdx.y == 0)
        for (int i = threadIdx.x; i < WS; i += blockDim.x)
        {
            int wsi = shWS[i] = ws[i];
            shY[i] = y[wsi];
            shG[i] = g[wsi];
            shAlpha[i] = alpha[wsi];
            shGUpdate[i] = 0;
        }
    __syncthreads();

    //calculate K
    if (calcK)
    {
        for (int b = 0; b < dim; b += 32)
        {
            for (int i = threadIdx.y; i < WS; i += blockDim.y)
                if (b + threadIdx.x < dim)
                    shX[i][threadIdx.x] = x[dim_aligned * ws[i] + b + threadIdx.x];
            __syncthreads();
            for (int j = threadIdx.y; j < WS; j += blockDim.y)
                for (int i = threadIdx.x; i < WS; i += blockDim.x)
                {
                    float sum = 0;
                    for (int k = 0; k < 32 && b + k < dim; k++)
                        sum += shX[i][k] * shX[j][k];
                    shK[j][i] += sum;
                }
        }
        __syncthreads();
        for (int j = threadIdx.y; j < WS; j += blockDim.y)
            for (int i = threadIdx.x; i < WS; i += blockDim.x)
                shK[j][i] = exp(-gamma * (x2[ws[i]] + x2[ws[j]] - 2 * shK[j][i]));
    }
    else
    {
        for (int j = threadIdx.y; j < WS; j += blockDim.y)
            for (int i = threadIdx.x; i < WS; i += blockDim.x)
                shK[j][i] = K[(size_t)num_vec_aligned * KCacheRemapIdx[shWS[j]] + shWS[i]];
    }
    __syncthreads();

    //optimization loop
    for (int iter = 0;; iter++)
    {
        //find I,J for SMO
        float max_valI = -FLT_MAX,
            max_valJ = -FLT_MAX;
        int max_idxI = 0,
            max_idxJ = 0;
        if (threadIdx.y == 0)  //only the first warp selects I,J
        {
            for (int i = threadIdx.x; i < WS; i += blockDim.x)
            {
                float v;
                float y_ = shY[i];
                float a_ = shAlpha[i];
                float g_ = shG[i];

                if ((y_ > 0 && a_ < C) || (y_ < 0 && a_ > 0))
                    v = y_ * g_;
                else
                    v = -FLT_MAX;
                if (v > max_valI)
                {
                    max_valI = v;
                    max_idxI = i;
                }

                //if (!secondOrder)
                {
                    if ((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C))
                        v = -y_ * g_;
                    else
                        v = -FLT_MAX;
                    if (v > max_valJ)
                    {
                        max_valJ = v;
                        max_idxJ = i;
                    }
                }
            }
            shVal[threadIdx.x] = max_valI;
            shIdx[threadIdx.x] = max_idxI;
        }
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.y == 0 && threadIdx.x < s)
                if (shVal[threadIdx.x + s] > shVal[threadIdx.x])
                {
                    shVal[threadIdx.x] = shVal[threadIdx.x + s];
                    shIdx[threadIdx.x] = shIdx[threadIdx.x + s];
                }
            __syncthreads();
        }
        int wsI = shIdx[0];  //found I
        int wsJ;
        int wsJ1;
        __syncthreads();

        //if (secondOrder)
        {
            if (threadIdx.y == 0)
            {
                shVal[threadIdx.x] = max_valJ;
                shIdx[threadIdx.x] = max_idxJ;
            }
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (threadIdx.y == 0 && threadIdx.x < s)
                    if (shVal[threadIdx.x + s] > shVal[threadIdx.x])
                    {
                        shVal[threadIdx.x] = shVal[threadIdx.x + s];
                        shIdx[threadIdx.x] = shIdx[threadIdx.x + s];
                    }
                __syncthreads();
            }
            wsJ1 = shIdx[0];
            __syncthreads();
        }
        if (secondOrder)
        {
            if (threadIdx.y == 0)
            {
                max_valJ = -FLT_MAX;
                max_idxJ = 0;
                for (int j = threadIdx.x; j < WS; j += blockDim.x)
                {
                    float v;
                    float y_ = shY[j];
                    float a_ = shAlpha[j];
                    float th = shY[wsI] * shG[wsI];
                    if (((y_ > 0 && a_ > 0) || (y_ < 0 && a_ < C)) && th > y_ * shG[j])
                    {
                        float den = 1 + 1 - 2 * shK[wsI][j];
                        float val = th - y_ * shG[j];
                        v = val * val / den;
                    }
                    else
                        v = -FLT_MAX;
                    if (v > max_valJ)
                    {
                        max_valJ = v;
                        max_idxJ = j;
                    }
                }

                shVal[threadIdx.x] = max_valJ;
                shIdx[threadIdx.x] = max_idxJ;
            }
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (threadIdx.y == 0 && threadIdx.x < s)
                    if (shVal[threadIdx.x + s] > shVal[threadIdx.x])
                    {
                        shVal[threadIdx.x] = shVal[threadIdx.x + s];
                        shIdx[threadIdx.x] = shIdx[threadIdx.x + s];
                    }
                __syncthreads();
            }
            wsJ = shIdx[0];
            __syncthreads();
        }
        else
            wsJ = wsJ1;

        float diff = shY[wsI] * shG[wsI] - shY[wsJ1] * shG[wsJ1];
        if (iter == 0)
        {
            if (threadIdx.x == 0 && threadIdx.y == 0)
            {
                d_diff = diff;
                //shLambda = min(0.5, max(eps/2, 0.1 * diff));
                //printf("diff: %f \teps2: %f\n", diff, shLambda);
            }
            /*__syncthreads();
            eps2 = shLambda;
            __syncthreads();*/
        }
        if (threadIdx.x == 0 && threadIdx.y == 0)
            printf("Block iter %d diff %f ws %d %d\n", iter, diff, wsI, wsJ);
        if (diff < eps || iter >= MAX_BLOCK_ITER)
        {
            //copy modified g and alpha back to global memory
            if (threadIdx.y == 0)
            {
                for (int i = threadIdx.x; i < WS; i += blockDim.x)
                {
                    int wsi = shWS[i];
                    alpha[wsi] = shAlpha[i];
                    if (!updateAllG)
                        g[wsi] = shG[i];
                }
                if (threadIdx.x == 0)
                {
                    //printf("diff: %f\n", diff);
                    d_block_num_iter = iter;
                    d_rho = -(shY[wsI] * shG[wsI] + shY[wsJ1] * shG[wsJ1]) / 2;
                }
            }
            break;  //exit kernel if optimal
        }

        //update alpha
        if (threadIdx.y == 0 && threadIdx.x == 0)
        {
            float l1 = shY[wsI] > 0 ? C - shAlpha[wsI] : shAlpha[wsI];
            float l2 = shY[wsJ] > 0 ? shAlpha[wsJ] : C - shAlpha[wsJ];
            float l3 = (shY[wsI] * shG[wsI] - shY[wsJ] * shG[wsJ]) / (1 + 1 - 2 * shK[wsI][wsJ]);  //1 is KDiag for RBF kernel
            float l = min(l1, min(l2, l3));

            shAlpha[wsI] += l * shY[wsI];
            shAlpha[wsJ] -= l * shY[wsJ];
            shLambda = l;
        }
        __syncthreads();
        //update g
        if (threadIdx.y == 0)
        {
            float l = shLambda;
            if (updateAllG)
            {
                shGUpdate[wsJ] += l;
                shGUpdate[wsI] -= l;
            }
            for (int k = threadIdx.x; k < WS; k += blockDim.x)
                shG[k] += l * shY[k] * (shK[wsJ][k] - shK[wsI][k]);
        }
        __syncthreads();
    }

    if (updateAllG)
    {
        for (int gk_offset = 0; gk_offset < num_vec; gk_offset += blockDim.x)
        {
            int gk = gk_offset + threadIdx.x;
            if (threadIdx.y == 0)
            {
                float update = 0;
                for (int i = 0; i < WS; i++)
                    update += shGUpdate[i] * K[(size_t)num_vec_aligned * KCacheRemapIdx[shWS[i]] + gk];
                g[gk] += y[gk] * update;
            }
        }
    }
}

#if 0
//storing whole K matrix in shared memory
//num threads = WS, e.g. 64
template<unsigned int WS>
__global__ static void kernelSMO1BlockV2(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
	__shared__ float shK[WS][WS];
    __shared__ int shWS[WS];
    __shared__ float shAlpha[WS];
    __shared__ float shY[WS];
    __shared__ int shIdx[WS];
    __shared__ float shLambda;
    __shared__ float shGUpdate[WS];
    __shared__ float shI[WS];
    __shared__ float shJ[WS];
    //float eps2;

    //copy buffers to shared memory
    int wsi = shWS[threadIdx.x] = d_ws[threadIdx.x];
    float y = shY[threadIdx.x] = d_y[wsi];
    float g = d_g[wsi];
    float a = shAlpha[threadIdx.x] = d_alpha[wsi];
    float aold = a;
    //shGUpdate[i] = 0;
    __syncthreads();
	
	float eps2;

	//copy K values
	#pragma unroll
	for (int j = 0; j < WS; j++)
		shK[j][threadIdx.x] = d_K[(size_t)num_vec_aligned * d_KCacheRemapIdx[shWS[j]] + shWS[threadIdx.x]]; //read KCacheRemapIdx first into shared?
	__syncthreads();

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            shI[threadIdx.x] = y * g;
        else
            shI[threadIdx.x] = -FLT_MAX;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            shJ[threadIdx.x] = -y * g;
        else
            shJ[threadIdx.x] = -FLT_MAX;
        int wsI = blockMaxReduce(shI, shIdx);
        int wsJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[wsI];

        float diff = vI + shJ[wsJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
            d_alpha[wsi] = shAlpha[threadIdx.x];
            d_alphadiff[threadIdx.x] = -(a - aold) * y;
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[wsJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        if (((y > 0 && a > 0) || (y < 0 && a < C)) && vI > y * g)
        {
            float den = 1 + 1 - 2 * shK[wsI][threadIdx.x];
            float val = vI - y * g;
            shI[threadIdx.x] = val * val / den;
        }
        else
            shI[threadIdx.x] = -FLT_MAX;
        wsJ = blockMaxReduce(shI, shIdx);

        //update alpha
        float l1 = shY[wsI] > 0 ? C - shAlpha[wsI] : shAlpha[wsI];
        float l2 = shY[wsJ] > 0 ? shAlpha[wsJ] : C - shAlpha[wsJ];
        float l3 = (vI + shJ[wsJ]) / (1 + 1 - 2 * shK[wsI][wsJ]);  //1 is KDiag for RBF kernel
        float l = min(l1, min(l2, l3));
        
        __syncthreads();
        if (threadIdx.x == wsI)
            shAlpha[threadIdx.x] = a += l * y;
        if (threadIdx.x == wsJ)
            shAlpha[threadIdx.x] = a -= l * y;
        __syncthreads();

        //update g
        g += l * y * (shK[wsJ][threadIdx.x] - shK[wsI][threadIdx.x]);
	} //main loop
} //kernelSMO1Block

#elif 0
//storing only bottom triangle of the K matrix in shared memory

__device__ int tri_idx(int x, int y)
{
    if (x > y)
    {
        int tmp = x;
        x = y;
        y = tmp;
    }
    return x + (y + 1) * y / 2;
}

//num threads = WS, e.g. 64
template<unsigned int WS>
__global__ static void kernelSMO1BlockV2_Tri(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
	__shared__ float shK[(WS * WS + WS) / 2];
    __shared__ int shWS[WS];
    __shared__ float shAlpha[WS];
    __shared__ float shY[WS];
    __shared__ int shIdx[WS];
    __shared__ float shLambda;
    __shared__ float shGUpdate[WS];
    __shared__ float shI[WS];
    __shared__ float shJ[WS];
    //float eps2;

    //copy buffers to shared memory
    int wsi = shWS[threadIdx.x] = d_ws[threadIdx.x];
    float y = shY[threadIdx.x] = d_y[wsi];
    float g = d_g[wsi];
    float a = shAlpha[threadIdx.x] = d_alpha[wsi];
    float aold = a;
    //shGUpdate[i] = 0;
    __syncthreads();
	
	float eps2;

	//copy K values
	#pragma unroll
	//for (int j = 0; j < WS; j++)
		//shK[j][threadIdx.x] = d_K[(size_t)num_vec_aligned * d_KCacheRemapIdx[shWS[j]] + shWS[threadIdx.x]]; //read KCacheRemapIdx first into shared?
    for (int j = 0; j < WS; j++)
        if (threadIdx.x <= j)
            shK[tri_idx(j, threadIdx.x)] = d_K[(size_t)num_vec_aligned * d_KCacheRemapIdx[shWS[j]] + shWS[threadIdx.x]];
	__syncthreads();

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            shI[threadIdx.x] = y * g;
        else
            shI[threadIdx.x] = -FLT_MAX;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            shJ[threadIdx.x] = -y * g;
        else
            shJ[threadIdx.x] = -FLT_MAX;
        int wsI = blockMaxReduce(shI, shIdx);
        int wsJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[wsI];

        float diff = vI + shJ[wsJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
            d_alpha[wsi] = shAlpha[threadIdx.x];
            d_alphadiff[threadIdx.x] = -(a - aold) * y;
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[wsJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        if (((y > 0 && a > 0) || (y < 0 && a < C)) && vI > y * g)
        {
            float den = 1 + 1 - 2 * shK[tri_idx(wsI, threadIdx.x)];
            float val = vI - y * g;
            shI[threadIdx.x] = val * val / den;
        }
        else
            shI[threadIdx.x] = -FLT_MAX;
        wsJ = blockMaxReduce(shI, shIdx);

        //update alpha
        float l1 = shY[wsI] > 0 ? C - shAlpha[wsI] : shAlpha[wsI];
        float l2 = shY[wsJ] > 0 ? shAlpha[wsJ] : C - shAlpha[wsJ];
        float l3 = (vI + shJ[wsJ]) / (1 + 1 - 2 * shK[tri_idx(wsI, wsJ)]);  //1 is KDiag for RBF kernel
        float l = min(l1, min(l2, l3));
        
        __syncthreads();
        if (threadIdx.x == wsI)
            shAlpha[threadIdx.x] = a += l * y;
        if (threadIdx.x == wsJ)
            shAlpha[threadIdx.x] = a -= l * y;
        __syncthreads();

        //update g
        g += l * y * (shK[tri_idx(wsJ, threadIdx.x)] - shK[tri_idx(wsI, threadIdx.x)]);
	} //main loop
} //kernelSMO1Block

#elif 0
//storing only Ith row of K matrix in shared memory
//num threads = WS, e.g. 64
template<unsigned int WS>
__global__ static void kernelSMO1BlockV2(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
	__shared__ float shKI[WS];
    __shared__ int shWS[WS];
    __shared__ float shAlpha[WS];
    __shared__ float shY[WS];
    __shared__ int shIdx[WS];
    __shared__ float shLambda;
    __shared__ float shGUpdate[WS];
    __shared__ float shI[WS];
    __shared__ float shJ[WS];
    //__shared__ int shCacheRow[WS];
    //float eps2;

    //copy buffers to shared memory
    int wsi = shWS[threadIdx.x] = d_ws[threadIdx.x];
    float y = shY[threadIdx.x] = d_y[wsi];
    float g = d_g[wsi];
    float a = shAlpha[threadIdx.x] = d_alpha[wsi];
    float aold = a;
    //shGUpdate[i] = 0;
    __syncthreads();
    //shCacheRow[threadIdx.x] = d_KCacheRemapIdx[shWS[threadIdx.x]];
    //__syncthreads();
	
	float eps2;

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            shI[threadIdx.x] = y * g;
        else
            shI[threadIdx.x] = -FLT_MAX;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            shJ[threadIdx.x] = -y * g;
        else
            shJ[threadIdx.x] = -FLT_MAX;
        int wsI = blockMaxReduce(shI, shIdx);
        //shKI[threadIdx.x] = d_K[(size_t)num_vec_aligned * shCacheRow[wsI] + shWS[threadIdx.x]];
        shKI[threadIdx.x] = d_K[WS * wsI + threadIdx.x];
        int wsJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[wsI];

        float diff = vI + shJ[wsJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
            d_alpha[wsi] = shAlpha[threadIdx.x];
            d_alphadiff[threadIdx.x] = -(a - aold) * y;
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[wsJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        if (((y > 0 && a > 0) || (y < 0 && a < C)) && vI > y * g)
        {
            float den = 1 + 1 - 2 * shKI[threadIdx.x];
            float val = vI - y * g;
            shI[threadIdx.x] = val * val / den;
        }
        else
            shI[threadIdx.x] = -FLT_MAX;
        wsJ = blockMaxReduce(shI, shIdx);
        float KJ = d_K[WS * wsJ + threadIdx.x];

        //update alpha
        float l1 = shY[wsI] > 0 ? C - shAlpha[wsI] : shAlpha[wsI];
        float l2 = shY[wsJ] > 0 ? shAlpha[wsJ] : C - shAlpha[wsJ];
        float l3 = (vI + shJ[wsJ]) / (1 + 1 - 2 * shKI[wsJ]);  //1 is KDiag for RBF kernel
        float l = min(l1, min(l2, l3));
        
        __syncthreads();
        if (threadIdx.x == wsI)
            shAlpha[threadIdx.x] = a += l * y;
        if (threadIdx.x == wsJ)
            shAlpha[threadIdx.x] = a -= l * y;
        __syncthreads();

        //update g
        //g += l * y * (d_K[(size_t)num_vec_aligned * shCacheRow[wsJ] + shWS[threadIdx.x]] - shKI[threadIdx.x]);
        g += l * y * (KJ - shKI[threadIdx.x]);
	} //main loop
} //kernelSMO1Block

#else
//storing only Ith row of K matrix in shared memory
//num threads = WS, e.g. 64
template<unsigned int WS>
__global__ static void kernelSMO1BlockV2(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
    __shared__ int shIdx[WS];
    __shared__ float shI[WS];
    __shared__ float shJ[WS];
    __shared__ float shLambda1, shLambda2;

    //copy buffers to shared memory
    int wsi = d_ws[threadIdx.x];
    float y = d_y[wsi];
    float g = d_g[wsi];
    float a = d_alpha[wsi];
    float aold = a;
    __syncthreads();
	
	float eps2;

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        if ((y > 0 && a < C) || (y < 0 && a > 0))
            shI[threadIdx.x] = y * g;
        else
            shI[threadIdx.x] = -FLT_MAX;
        if ((y > 0 && a > 0) || (y < 0 && a < C))
            shJ[threadIdx.x] = -y * g;
        else
            shJ[threadIdx.x] = -FLT_MAX;
        int wsI = blockMaxReduce(shI, shIdx);
        __syncthreads();
        float KI = d_K[WS * wsI + threadIdx.x];
        int wsJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[wsI];

        float diff = vI + shJ[wsJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
            d_alpha[wsi] = a;
            d_alphadiff[threadIdx.x] = -(a - aold) * y;
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[wsJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        if (((y > 0 && a > 0) || (y < 0 && a < C)) && vI > y * g)
        {
            float den = 1 + 1 - 2 * KI;
            float val = vI - y * g;
            shI[threadIdx.x] = val * val / den;
        }
        else
            shI[threadIdx.x] = -FLT_MAX;
        wsJ = blockMaxReduce(shI, shIdx);
        float KJ = d_K[WS * wsJ + threadIdx.x];

        //update alpha
        //float l1 = shY[wsI] > 0 ? C - shAlpha[wsI] : shAlpha[wsI];
        //float l2 = shY[wsJ] > 0 ? shAlpha[wsJ] : C - shAlpha[wsJ];
        //float l3 = (vI + shJ[wsJ]) / (1 + 1 - 2 * shKI[wsJ]);  //1 is KDiag for RBF kernel
        //float l = min(l1, min(l2, l3));
        if (threadIdx.x == wsI)
            shLambda1 = y > 0 ? C - a : a;
        if (threadIdx.x == wsJ)
            shLambda2 = min(y > 0 ? a : C - a, (vI + shJ[wsJ]) / (1 + 1 - 2 * KI));
        __syncthreads();
        float l = min(shLambda1, shLambda2);
        
        if (threadIdx.x == wsI)
            a += l * y;
        if (threadIdx.x == wsJ)
            a -= l * y;

        //update g
        g += l * y * (KJ - KI);
	} //main loop
} //kernelSMO1Block

#endif
//storing only Ith row of K matrix in shared memory
//each thread processes several elements
//num threads = WS / N
template<unsigned int WS, unsigned int N>
__global__ static void kernelSMO1BlockV2N(const float * d_y, float * d_g, float * d_alpha, float * d_alphadiff, const int * d_ws, float gamma, float C, float eps, int num_vec_aligned, float * d_K, int * d_KCacheRemapIdx)
{
	__shared__ float shKI[WS];
    __shared__ int shIdx[WS/N];
    __shared__ float shI[WS/N];
    __shared__ float shJ[WS/N];
    __shared__ int shIdxThread[WS/N];
    __shared__ float shLambda;

    //copy buffers to shared memory
    int wsi[N];
    float y[N],
          g[N],
          a[N],
          aold[N];
#pragma unroll
    for (int n = 0; n < N; n++)
    {
        wsi[n] = d_ws[blockDim.x * n + threadIdx.x];
        y[n] = d_y[wsi[n]];
        g[n] = d_g[wsi[n]];
        aold[n] = a[n] = d_alpha[wsi[n]];
    }
    __syncthreads();
	
	float eps2;

	//optimization loop
	for (int iter = 0;; iter++)
	{
        //select I,J first order
        float maxI = -FLT_MAX;
        float maxJ = -FLT_MAX;
        int maxIidx;
#pragma unroll
        for (int n = 0; n < N; n++)
        {
            if ((y[n] > 0 && a[n] < C) || (y[n] < 0 && a[n] > 0))
            {
                float v = y[n] * g[n];
                if (v > maxI)
                {
                    maxI = v;
                    maxIidx = n;
                }
            }
            if ((y[n] > 0 && a[n] > 0) || (y[n] < 0 && a[n] < C))
            {
                float v = -y[n] * g[n];
                if (v > maxJ)
                    maxJ = v;
            }
        }
        shI[threadIdx.x] = maxI;
        shJ[threadIdx.x] = maxJ;
        shIdxThread[threadIdx.x] = maxIidx;
        int thI = blockMaxReduce(shI, shIdx);
        int wsI = blockDim.x * shIdxThread[thI] + thI;
#pragma unroll
        for (int n = 0; n < N; n++)
            shKI[blockDim.x * n + threadIdx.x] = d_K[WS * wsI + blockDim.x * n + threadIdx.x];
        int thJ = blockMaxReduce(shJ, shIdx);
        float vI = shI[thI];

        float diff = vI + shJ[thJ];
        if (iter == 0)
        {
            if (threadIdx.x == 0)
                d_diff = diff;
            eps2 = min(0.5, max(eps / 2.0, 0.1 * diff));
        }

        //TODO: should "diff < eps" be there? prolly not
        if (diff < eps2 || iter >= MAX_BLOCK_ITER || diff < eps)
        {
#pragma unroll
            for (int n = 0; n < N; n++)
            {
                d_alpha[wsi[n]] = a[n];
                d_alphadiff[blockDim.x * n + threadIdx.x] = -(a[n] - aold[n]) * y[n];
            }
			
            if (threadIdx.x == 0)
            {
                d_block_num_iter = iter;
                d_total_num_iter += iter;
                d_rho = -(vI - shJ[thJ]) / 2;
            }
            break;
        }
        __syncthreads();

        //select J, second order
        //using shI
        maxJ = -FLT_MAX;
        int maxJidx;
        for (int n = 0; n < N; n++)
        {
            if (((y[n] > 0 && a[n] > 0) || (y[n] < 0 && a[n] < C)) && vI > y[n] * g[n])
            {
                float den = 1 + 1 - 2 * shKI[blockDim.x * n + threadIdx.x];
                float val = vI - y[n] * g[n];
                val = val * val / den;
                if (val > maxJ)
                {
                    maxJ = val;
                    maxJidx = n;
                }
            }
        }
        shI[threadIdx.x] = maxJ;
        shIdxThread[threadIdx.x] = maxJidx;
        thJ = blockMaxReduce(shI, shIdx);
        int wsJ = blockDim.x * shIdxThread[thJ] + thJ;
        float KJ[N];
#pragma unroll
        for (int n = 0; n < N; n++)
            KJ[n] = d_K[WS * wsJ + blockDim.x * n + threadIdx.x];

        //update alpha
        if (threadIdx.x == thI)
            shLambda = y[maxIidx] > 0 ? C - a[maxIidx] : a[maxIidx];
        __syncthreads();
        if (threadIdx.x == thJ)
        {
            float l = min(shLambda, y[maxJidx] > 0 ? a[maxJidx] : C - a[maxJidx]);
            shLambda = min(l, (vI - y[maxJidx] * g[maxJidx]) / (1 + 1 - 2 * shKI[wsJ]));
        }
        
        __syncthreads();
        if (threadIdx.x == thI)
            a[maxIidx] += shLambda * y[maxIidx];
        if (threadIdx.x == thJ)
            a[maxJidx] -= shLambda * y[maxJidx];

        //update g
#pragma unroll
        for (int n = 0; n < N; n++)
            g[n] += shLambda * y[n] * (KJ[n] - shKI[blockDim.x * n + threadIdx.x]);
	} //main loop
}
