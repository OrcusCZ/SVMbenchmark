#pragma once

__device__ size_t d_block_num_iter;
__device__ size_t d_total_num_iter;
__device__ float d_rho;
__device__ float d_diff;
__device__ int d_cache_rows_to_compute[MAX_WORKING_SET]; //must be >= WORKING_SET
__device__ int d_num_cache_rows_to_compute;
__device__ int d_updateGCnt[2];

__device__ int d_cacheUpdateCnt;
//contains changes to KCacheRemapIdx buffer, which should be written after kernelCheckCache ends
//each change to buffer is contained in int2 variable (x,y) such that
//KCacheRemapIdx[x] = y
//pair at index [2] is for KCacheRowPriority
__device__ int2 d_KCacheChanges[3];

__device__ int d_cacheRow;

__device__ float d_aux_one, d_aux_zero;

#ifdef USE_DAIFLETCHER
__device__ float d_df_lam_ext;
__device__ float d_df_e;
__device__ int d_df_ls;
__device__ int d_df_proj;
#endif
DEFINE_SYNC_BUFFERS(1); //for inter-blocks synchronization
