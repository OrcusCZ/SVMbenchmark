#define NUM_THREADS 256 //must match with cpp file
#define DENSE_TILE_SIZE 16 //must match with cpp file
#define SPARSE_TILE_SIZE 16 //must match with cpp file

//#define FLT_MAX         3.402823466e+38F   //float.h copy
#ifndef INT_MAX
#define INT_MAX         2147483647 //limit.h copy
#endif

#define gidx get_global_id(0)
#define gidy get_global_id(1)
#define tidx get_local_id(0)
#define tidy get_local_id(1)
#define bidx get_group_id(0)
#define bidy get_group_id(1)

__kernel void kernelMemsetInt( __global int* buf, int value, unsigned int num ){
       unsigned int  x = get_global_id(0);
       if(x < num) buf[x] = value;
}

__kernel void kernelMemsetFloat( __global float* buf, float value, unsigned int num ){
       unsigned int x = get_global_id(0);
       if(x < num) buf[x] = value;
}

__kernel void kernelCalcRowLen(__global unsigned int * rowLen, __global __read_only unsigned int * rowOffsets, unsigned int numRows)
{
    int k = gidx;

    if (k < numRows)
    {
        rowLen[k] = rowOffsets[k + 1] - rowOffsets[k];
    }
}

#define LOCAL_SUM(sh_val, tid, num_threads) { \
    if (num_threads >= 1024) { \
        if (tid < 512) { \
            sh_val[tid] += sh_val[tid+512]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (num_threads >= 512) { \
        if (tid < 256) { \
            sh_val[tid] += sh_val[tid+256]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (num_threads >= 256) { \
        if (tid < 128) { \
            sh_val[tid] += sh_val[tid+128]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    if (num_threads >= 128) { \
        if (tid < 64) { \
            sh_val[tid] += sh_val[tid+64]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
    for(int i=32; i>=1; i/=2) { \
        if (tid < 64) { \
            sh_val[tid] += sh_val[tid+i]; \
        } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } \
} //LOCAL_SUM


//<<< num_vects, NUM_THREADS >>>
__kernel void kernelSumPow2(__global float *result, __global __read_only float *data, unsigned int dim, unsigned int dim_aligned)
{
    local float sbuff[NUM_THREADS];

    unsigned int offset = dim_aligned * get_group_id(0);

    float sum = 0;
    for(int i=tidx; i < dim; i += NUM_THREADS) {
        float val = data[offset + i];
        sum += val*val;
    }
    sbuff[tidx] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    LOCAL_SUM(sbuff, tidx, NUM_THREADS);

    if(tidx == 0) result[bidx] = sbuff[0];

} //kernelSumPow2

__kernel void kernelPow2SumSparse(__global __read_only float * x_values, __global __read_only unsigned int * x_rowOffsets, unsigned int x_numRows, __global float * x2)
{
    unsigned int id = gidx;
    
    while (id < x_numRows)
    {
        float sum = 0;
        int end = x_rowOffsets[id + 1];
        for (int i = x_rowOffsets[id]; i < end; i++)
        {
            float v = x_values[i];
            sum += v * v;
        }
        x2[id] = sum;

        id += get_num_groups(0) * get_local_size(0);
    }
} //kernelPow2SumSparse

__kernel void kernelSelectI(__global float * valbuf, __global int * idxbuf, __global __read_only float * y, __global __read_only float * g, __global __read_only float * alpha, float C, unsigned int num_vec)
{

    local float sval[NUM_THREADS];
    local int sidx[NUM_THREADS];

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for(int i = gidx; i < num_vec; i += get_global_size(0))
    {
		float v;
		float y_ = y[i];
		float a_ = alpha[i];
		if ((y_ == 1 && a_ < C) || (y_ == -1 && a_ > 0))
			v = y_ * g[i];
		else
			v = -FLT_MAX;
		if (v > max_val)
		{
			max_val = v;
			max_idx = i;
		}
    }

   	//local reduction
    sval[tidx] = max_val;
    sidx[tidx] = max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);
	
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
	{
        if (tidx < s)
        {
            if (sval[tidx + s] > sval[tidx])
            {
                sval[tidx] = sval[tidx + s];
                sidx[tidx] = sidx[tidx + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	if (tidx == 0)
    {
		valbuf[bidx] = sval[0];
		idxbuf[bidx] = sidx[0];
    }

} //kernelSelectI

//second order search with cached K
__kernel void kernelSelectJCached(__global float * valbuf, __global int * idxbuf, __global __read_only float * y, __global __read_only float * g, __global __read_only float * alpha, float C, unsigned int num_vec, unsigned int num_vec_aligned, __global __read_only int * i_ptr, __global __read_only float * K, __global __read_only float * KDiag, __global __read_only int * KCacheRemapIdx)
{

    local float sval[NUM_THREADS];
    local int sidx[NUM_THREADS];

    float max_val = -FLT_MAX;
    int max_idx = 0;

    int i = i_ptr[0];
    int cache_row = KCacheRemapIdx[i];
    float th = y[i] * g[i];
	
    for(int k = gidx; k < num_vec; k += get_global_size(0))
    {
		float vv;
        float y_ = y[k];
        float a_ = alpha[k];
				
        if (((y_ == 1 && a_ > 0) || (y_ == -1 && a_ < C)) && th > y[k] * g[k])
        {
            float den = KDiag[i] + KDiag[k] - 2 * K[num_vec_aligned * cache_row + k];
            float v = th - y[k] * g[k];
            vv = v * v / den;
        }
        else 
            vv = -FLT_MAX;
			
		if (vv > max_val)
		{
			max_val = vv;
			max_idx = k;
		}
    }

   	//local reduction
    sval[tidx] = max_val;
    sidx[tidx] = max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);
	
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
	{
        if (tidx < s)
        {
            if (sval[tidx + s] > sval[tidx])
            {
                sval[tidx] = sval[tidx + s];
                sidx[tidx] = sidx[tidx + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	if (tidx == 0)
    {
		valbuf[bidx] = sval[0];
		idxbuf[bidx] = sidx[0];
    }
	
} //kernelSelectJCached


__kernel void kernelReduceMaxIdx(__global __read_only float * val, __global __read_only int * idx, __global float * val_out, __global int * idx_out, int res_offset, unsigned int len)
{
    local float sval[NUM_THREADS];
    local int sidx[NUM_THREADS];

    int frame = NUM_THREADS * bidx;
    float max_val = -FLT_MAX;
    int max_idx = 0;

    for(int i = gidx; i < len; i += get_global_size(0))
    {
		float v = val[i];
		if (v > max_val)
		{
			max_val = v;
			max_idx = idx[i];
		}
   }

    sval[tidx] = max_val;
    sidx[tidx] = max_idx;
    barrier(CLK_LOCAL_MEM_FENCE);
	
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
	{
        if (tidx < s)
        {
            if (sval[tidx + s] > sval[tidx])
            {
                sval[tidx] = sval[tidx + s];
                sidx[tidx] = sidx[tidx + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	if (tidx == 0)
    {
		if(res_offset < 0) {
			val_out[bidx] = sval[0];
			idx_out[bidx] = sidx[0];
		} else {
			idx_out[bidx+res_offset] = sidx[0];
		}
    }
	
} //kernelReduceMaxIdxDebug

__kernel void kernelUpdateAlphaAndLambdaCached(__global float * alpha, __global float * lambda, __global __read_only float * y, __global __read_only float * g, __global __read_only float * K, float C, __global __read_only int * ws, unsigned int num_vec_aligned, __global __read_only float * KDiag, __global __read_only int * KCacheRemapIdx)
{
    int i = ws[0];
    int j = ws[1];
    int cache_row = KCacheRemapIdx[i];
    float l1 = y[i] > 0 ? C - alpha[i] : alpha[i];
    float l2 = y[j] > 0 ? alpha[j] : C - alpha[j];
    float l3 = (y[i] * g[i] - y[j] * g[j]) / (KDiag[i] + KDiag[j] - 2 * K[num_vec_aligned * cache_row + j]);
    float l = min(l1, min(l2, l3));

    lambda[0] = l;
    alpha[i] += l * y[i];
    alpha[j] -= l * y[j];
} //kernelUpdateAlphaAndLambdaCached

__kernel void kernelUpdategCached(__global float * g, __global __read_only float * lambda, __global __read_only float * y, __global __read_only float * K, __global __read_only int * ws, unsigned int num_vec, unsigned int num_vec_aligned, __global __read_only int * KCacheRemapIdx)
{
    int i = ws[0];
    int j = ws[1];
    int i_cache_row = KCacheRemapIdx[i];
    int j_cache_row = KCacheRemapIdx[j];
    unsigned int k = gidx;
    if (k < num_vec)
    {
        g[k] += lambda[0] * y[k] * (K[num_vec_aligned * j_cache_row + k] - K[num_vec_aligned * i_cache_row + k]);
    }
} //kernelUpdategCached


__kernel void kernelCheckCacheFinalize(__global __read_only int * KCacheRemapIdx, __global __read_only int * KCacheRowPriority, __global int2 *KCacheChanges)
{
    int2 c;
#pragma unroll
    for (int i = 0; i < 2; i++)
    {
        c = KCacheChanges[i];
        if (c.x >= 0)
        {
            KCacheRemapIdx[c.x] = c.y;
            KCacheChanges[i].x = -1;
        }
    }
    c = KCacheChanges[2];
    if (c.x >= 0)
    {
        KCacheRowPriority[c.x] = c.y;
        KCacheChanges[2].x = -1;
    }
} //kernelCheckCacheFinalize

__kernel void kernelCheckCachePriorityV2(__global __read_only int * i_ptr, unsigned int offset, __global float * K, __global int * KCacheRemapIdx, __global int * KCacheRowIdx, __global int * KCacheRowPriority, int cache_rows, __global __read_only float * x, __global __read_only float * x2, float gamma, unsigned int num_vec, unsigned int num_vec_aligned, unsigned int dim, unsigned int dim_aligned, __global int *d_cacheRow)
{
    int last = d_cacheRow[0];
    if (last < 0)
        return;
    local float shsum[DENSE_TILE_SIZE][DENSE_TILE_SIZE+1];
    int block = get_local_size(0) * bidx;
    int i = i_ptr[offset];

    //calculate cache matrix row [last], original index is [i]
    while (block < num_vec)
    {
        int j = block + tidy;
        int jout = block + tidx;
        float sum = 0;
        if (j < num_vec)
        {
            for (int d = tidx; d < dim; d += DENSE_TILE_SIZE)
            {
                sum += x[dim_aligned * i + d] * x[dim_aligned * j + d];
            }
        }
        shsum[tidy][tidx] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
        {
            if (tidx < s)
                shsum[tidy][tidx] += shsum[tidy][tidx + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tidy == 0 && jout < num_vec)
        {
            sum = x2[i] + x2[jout] - 2 * shsum[tidx][0];
            K[num_vec_aligned * last + jout] = native_exp(-gamma * sum);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        block += get_num_groups(0) * get_local_size(0);
    }
} //kernelCheckCachePriorityV2

__kernel void kernelFindCacheRow(__global __read_only int * i_ptr, unsigned int offset, __global int * KCacheRemapIdx, __global int * KCacheRowIdx, __global int * KCacheRowPriority, int cache_rows, __global int2 * d_KCacheChanges, __global int * d_cacheUpdateCnt, __global int * d_cacheRow)
{
    local int2 spriority[NUM_THREADS];
    int j = gidx;
    int i = i_ptr[offset];
    if (KCacheRemapIdx[i] >= 0)
    {
        if (j == 0)
        {
            KCacheRowPriority[KCacheRemapIdx[i]] = d_cacheUpdateCnt[0];  // refresh priority
            d_cacheRow[0] = -2;
        }
        return;  //item already in cache
    }
    int2 minpriority = (int2)(INT_MAX, 0);
    for (int k = 0; k < cache_rows; k += get_local_size(0))
    {
        int idx = k + tidx;
        if (idx < cache_rows)
        {
            int v = KCacheRowPriority[idx];
            if (v < minpriority.x)
                minpriority = (int2)(v, idx);
        }
    }
    spriority[tidx] = minpriority;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
    {
        if (tidx < s)
        {
            if (spriority[tidx + s].x < spriority[tidx].x)
                spriority[tidx] = spriority[tidx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0)
    {
        int last = spriority[0].y;
        int del_i = KCacheRowIdx[last];
        if (del_i >= 0)
            d_KCacheChanges[1] = (int2)(del_i, -1);  //cache row for vector [del_i] will be overwritten, remove it from RemapIdx array
        //set correct indices
        d_KCacheChanges[0] = (int2)(i, last);
        KCacheRowIdx[last] = i;
        d_KCacheChanges[2] = (int2)(last, ++d_cacheUpdateCnt[0]);
        d_cacheRow[0] = last;
    }
} //kernelFindCacheRow

__kernel void kernelMakeDenseVec(__global __read_only int * i_ptr, unsigned int offset, __global __read_only int * KCacheRemapIdx, __global __read_only float * x_values, __global __read_only int * x_rowOffsets, __global __read_only int * x_colInd, __global float * vec)
{
	int i = i_ptr[offset];
	if (KCacheRemapIdx[i] >= 0)  //if [i] is already in cache, exit. we won't need any dense vector
		return;
	int j = x_rowOffsets[i] + gidx;
	while (j < x_rowOffsets[i + 1])
	{
		vec[x_colInd[j]] = x_values[j];
		j += get_num_groups(0) * get_local_size(0);
	}
} //kernelMakeDenseVec

__kernel void kernelCheckCacheSparsePriorityV2(__global __read_only int * i_ptr, unsigned int offset, __global float * K, __global int * KCacheRemapIdx, __global int * KCacheRowIdx, __global int * KCacheRowPriority, int cache_rows, __global __read_only int * d_cacheRow, __global __read_only float * vec, __global __read_only float * x2, __global __read_only float * x_values, __global __read_only int * x_rowOffsets, __global __read_only int * x_colInd, __global __read_only int * x_rowLen, float gamma, unsigned int num_vec, unsigned int num_vec_aligned, unsigned int dim, unsigned int dim_aligned)
{
    int last = d_cacheRow[0];
    if (last < 0)
        return;
    local float shsum[SPARSE_TILE_SIZE][SPARSE_TILE_SIZE+1];
    int block = get_local_size(0) * bidx;
    int i = i_ptr[offset];

    //calculate cache matrix row [last], original index is [i]
    while (block < num_vec)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        int j = block + tidy;
        int jout = block + tidx;
        float sum = 0;
        if (j < num_vec)
        {
            int end = x_rowOffsets[j] + x_rowLen[j];
            for (int d = x_rowOffsets[j] + tidx; d < end; d += SPARSE_TILE_SIZE)
            {
                sum += vec[x_colInd[d]] * x_values[d];
            }
        }
        shsum[tidy][tidx] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = get_local_size(0) / 2; s > 0; s >>= 1)
        {
            if (tidx < s)
                shsum[tidy][tidx] += shsum[tidy][tidx + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tidy == 0 && jout < num_vec)
        {
            sum = x2[i] + x2[jout] - 2 * shsum[tidx][0];
            K[num_vec_aligned * last + jout] = native_exp(-gamma * sum);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        block += get_num_groups(0) * get_local_size(0);
    }
} //kernelCheckCacheSparsePriorityV2

