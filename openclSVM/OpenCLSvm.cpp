#define NOMINMAX
#include <iostream>
#include <fstream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <cfloat>

#include "simpleCL.h"
#include "my_stopwatch.h"
#include "OpenCLSvm.h"



#ifndef HAVE_NULLPTR
#define nullptr NULL
#endif

#define ALIGN_UP(x, align) (align * ((x + align - 1) / align))
#define DIV_ALIGNED(x, y) ((x)/(y) + ((x)%(y)>0))

#define NUM_THREADS 256 //must match with kernels.cl file
#define DENSE_TILE_SIZE 16 //must match with kernels.cl file
#define SPARSE_TILE_SIZE 16 //must match with kernels.cl file
#define OCL_CACHE_SIZE_MB 800

extern int g_cache_size;

//if 0 - OK, else wrong
#define assert_opencl_mem_void(allocation) if((allocation) == nullptr) {printf("OpenCL GPU mem allocation Error (file:%s, line:%d)\n", __FILE__, __LINE__); return;}
#define assert_opencl_mem(allocation) if((allocation) == nullptr) {printf("OpenCL GPU mem allocation Error (file:%s, line:%d)\n", __FILE__, __LINE__); return -1;}

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
	cl_mem values;
	cl_mem colInd;
	cl_mem rowOffsets;
    cl_mem rowLen;
};

//make a GPU deep copy of a CPU csr matrix
int make_opencl_csr(sclHard gpu_hardware, csr_gpu &x_gpu, const csr &x_cpu) {
	x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;
	
	assert_opencl_mem(x_gpu.values = sclMallocWrite(gpu_hardware, CL_MEM_READ_WRITE, x_gpu.nnz * sizeof(float), x_cpu.values));
	assert_opencl_mem(x_gpu.colInd = sclMallocWrite(gpu_hardware, CL_MEM_READ_WRITE, x_gpu.nnz * sizeof(int), x_cpu.colInd));
	assert_opencl_mem(x_gpu.rowOffsets = sclMallocWrite(gpu_hardware, CL_MEM_READ_WRITE, (x_gpu.numRows+1) * sizeof(int), x_cpu.rowOffsets));
	assert_opencl_mem(x_gpu.rowLen = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, x_gpu.numRows * sizeof(int)));

	sclSoft software;
	software = sclGetCLSoftware("kernels.cl", "kernelCalcRowLen", gpu_hardware );
	size_t global_size[2], local_size[2];
    global_size[0] = ALIGN_UP(x_gpu.numRows, NUM_THREADS); global_size[1] = 1;
	local_size[0] = NUM_THREADS; local_size[1] = 1;

	sclManageArgsLaunchKernel( gpu_hardware, software, global_size, local_size, " %a %a %a ",
		sizeof(cl_mem), &x_gpu.rowLen,
		sizeof(cl_mem), &x_gpu.rowOffsets,
		sizeof(int), &x_gpu.numRows
	);

	sclReleaseClSoft( software );

	return 0;
} //make_opencl_csr

void openclCsrFree(csr_gpu &x_gpu) {
	sclReleaseMemObject(x_gpu.values);
	sclReleaseMemObject(x_gpu.colInd);
	sclReleaseMemObject(x_gpu.rowOffsets);
    sclReleaseMemObject(x_gpu.rowLen);
	x_gpu.values = nullptr;
	x_gpu.colInd = nullptr;
	x_gpu.rowOffsets = nullptr;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
} //openclCsrFree

void openclMemsetFloat(sclHard & gpu_hardware, cl_mem & buffer, float value, unsigned int num) {
	static sclSoft software = sclGetCLSoftware("kernels.cl", "kernelMemsetFloat", gpu_hardware );
	size_t global_size[2], local_size[2];
    global_size[0] = ALIGN_UP(num, NUM_THREADS); global_size[1] = 1;
	local_size[0] = NUM_THREADS; local_size[1] = 1;

	sclSetArgsEnqueueKernel( gpu_hardware, software, global_size, local_size, " %a %a %a ",
		sizeof(cl_mem), &buffer,
		sizeof(float), &value,
		sizeof(int), &num
	);

	//sclReleaseClSoft( software );

} //openclMemsetFloat

void openclMemsetInt(sclHard & gpu_hardware, cl_mem & buffer, int value, unsigned int num) {
	static sclSoft software = sclGetCLSoftware("kernels.cl", "kernelMemsetInt", gpu_hardware );
	size_t global_size[2], local_size[2];
    global_size[0] = ALIGN_UP(num, NUM_THREADS); global_size[1] = 1;
	local_size[0] = NUM_THREADS; local_size[1] = 1;

	sclSetArgsEnqueueKernel( gpu_hardware, software, global_size, local_size, " %a %a %a ",
		sizeof(cl_mem), &buffer,
		sizeof(int), &value,
		sizeof(int), &num
	);

	//sclReleaseClSoft( software );
} //openclMemsetInt



void computeX2Dense(sclHard & gpu_hardware, cl_mem & d_x, cl_mem & d_x2sum, unsigned int & num_vec, unsigned int & num_vec_aligned, unsigned int & dim, unsigned int & dim_aligned) {
 	sclSoft software;
	software = sclGetCLSoftware("kernels.cl", "kernelSumPow2", gpu_hardware );
	size_t global_size[2], local_size[2];
    global_size[0] = num_vec * NUM_THREADS; global_size[1] = 1;
	local_size[0] = NUM_THREADS; local_size[1] = 1;

	sclSetArgsEnqueueKernel( gpu_hardware, software, global_size, local_size, " %a %a %a %a ",
		sizeof(cl_mem), &d_x2sum,
		sizeof(cl_mem), &d_x,
		sizeof(int), &dim,
		sizeof(int), &dim_aligned
	);

	sclReleaseClSoft( software );
} //computeX2Dense

void computeX2Sparse(sclHard & gpu_hardware, csr_gpu & x, cl_mem & d_x2)
{
 	sclSoft software;
	software = sclGetCLSoftware("kernels.cl", "kernelPow2SumSparse", gpu_hardware );
	size_t global_size[2], local_size[2];
	size_t numBlocks = std::min<unsigned int>(NUM_THREADS, DIV_ALIGNED(x.numRows, NUM_THREADS) );
    global_size[0] = numBlocks * NUM_THREADS; global_size[1] = 1;
	local_size[0] = NUM_THREADS; local_size[1] = 1;

	sclSetArgsEnqueueKernel( gpu_hardware, software, global_size, local_size, " %a %a %a %a ",
		sizeof(cl_mem), &x.values,
		sizeof(cl_mem), &x.rowOffsets,
		sizeof(int), &x.numRows,
		sizeof(cl_mem), &d_x2
	);

	sclReleaseClSoft( software );
} //computeX2Sparse

void computeKDiag(sclHard & gpu_hardware, cl_mem & d_KDiag, unsigned int & num_vec)
{
    //K[i,i] is always 1 for RBF kernel, let's just use memset here
	openclMemsetFloat(gpu_hardware, d_KDiag, 1, num_vec);

} //computeKDiag

void reduceMaxIdx(sclHard & gpu_hardware, cl_mem & d_reduceval, cl_mem & d_reduceidx, unsigned int & num_vec, cl_mem & d_result, int result_offset)
{

	static sclSoft kernelReduceMax = sclGetCLSoftware("kernels.cl", "kernelReduceMaxIdx", gpu_hardware );
	
	static int offset = 0;
	static unsigned int num_active_blocks = DIV_ALIGNED(num_vec, NUM_THREADS);

	size_t global_size[2], local_size[2];
	global_size[0] = NUM_THREADS; global_size[1] = 1;
	local_size[0] = NUM_THREADS; local_size[1] = 1;

	offset = result_offset;
	sclSetArgsEnqueueKernel( gpu_hardware, kernelReduceMax, global_size, local_size, " %a %a %a %a %a %a ",
		sizeof(cl_mem), &d_reduceval,
		sizeof(cl_mem), &d_reduceidx,
		sizeof(cl_mem), &d_reduceval,
		sizeof(cl_mem), &d_result,
		sizeof(int), &offset,
		sizeof(int), &num_active_blocks
	);

} //reduceMaxIdx

void checkCache(sclHard &gpu_hardware, bool sparse, cl_mem & d_i, unsigned int _offset, cl_mem & d_x, cl_mem & d_x2, const csr_gpu & sparse_data_gpu, cl_mem & d_K, cl_mem & d_KCacheRemapIdx, cl_mem & d_KCacheRowIdx, cl_mem & d_KCacheRowPriority, cl_mem & d_denseVec, unsigned int & num_vec, unsigned int & num_vec_aligned, unsigned int & dim, unsigned int & dim_aligned, unsigned int & cache_rows, float & gamma, cl_mem & d_KCacheChanges, cl_mem & d_cacheUpdateCnt, cl_mem & d_cacheRow)
{
	static bool firstRun = true;
	static sclSoft kernelFindCacheRow, kernelCheckCachePriorityV2, kernelCheckCacheFinalize, kernelMakeDenseVec, kernelCheckCacheSparsePriorityV2;
	static size_t global_oneBlock[2], local_oneBlock[2];
	static size_t global_sparse[2], local_sparse[2];
	static size_t global_dense[2], local_dense[2];
	static size_t global_makedense[2], local_makedense[2];
	static size_t global_one[2], local_one[2];
	static unsigned int offset=_offset;


	offset = _offset;
	if(firstRun) {
		firstRun = false;

		kernelFindCacheRow = sclGetCLSoftware("kernels.cl", "kernelFindCacheRow", gpu_hardware );
		global_oneBlock[0] = NUM_THREADS; global_oneBlock[1] = 1;
		local_oneBlock[0] = NUM_THREADS; local_oneBlock[1] = 1;
		sclSetKernelArgs( kernelFindCacheRow, " %a %a %a %a %a %a %a %a %a ", 
			sizeof(cl_mem), &d_i,
			sizeof(int), &offset,
			sizeof(cl_mem), &d_KCacheRemapIdx,
			sizeof(cl_mem), &d_KCacheRowIdx,
			sizeof(cl_mem), &d_KCacheRowPriority,
			sizeof(int), &cache_rows,
			sizeof(cl_mem), &d_KCacheChanges,
			sizeof(cl_mem), &d_cacheUpdateCnt,
			sizeof(cl_mem), &d_cacheRow
		);

		kernelCheckCacheFinalize = sclGetCLSoftware("kernels.cl", "kernelCheckCacheFinalize", gpu_hardware );
		global_one[0] = 1; global_one[1] = 1;
		local_one[0] = 1; local_one[1] = 1;
		sclSetKernelArgs( kernelCheckCacheFinalize, " %a %a %a ", 
			sizeof(cl_mem), &d_KCacheRemapIdx,
			sizeof(cl_mem), &d_KCacheRowPriority,
			sizeof(cl_mem), &d_KCacheChanges
		);

		if(sparse) {
			kernelMakeDenseVec = sclGetCLSoftware("kernels.cl", "kernelMakeDenseVec", gpu_hardware );
			global_makedense[0] = NUM_THREADS * std::min<int>(64, DIV_ALIGNED(dim, NUM_THREADS)); global_makedense[1] = 1;
			local_makedense[0] = NUM_THREADS; local_makedense[1] = 1;
			sclSetKernelArgs( kernelMakeDenseVec, " %a %a %a %a %a %a %a ", 
				sizeof(cl_mem), &d_i,
				sizeof(int), &offset,
				sizeof(cl_mem), &d_KCacheRemapIdx,
				sizeof(cl_mem), &sparse_data_gpu.values,
				sizeof(cl_mem), &sparse_data_gpu.rowOffsets,
				sizeof(cl_mem), &sparse_data_gpu.colInd,
				sizeof(cl_mem), &d_denseVec
			);

			kernelCheckCacheSparsePriorityV2 = sclGetCLSoftware("kernels.cl", "kernelCheckCacheSparsePriorityV2", gpu_hardware );
			global_sparse[0] = 0; global_sparse[1] = SPARSE_TILE_SIZE;
			local_sparse[0] = SPARSE_TILE_SIZE; local_sparse[1] = SPARSE_TILE_SIZE;
			sclSetKernelArgs( kernelCheckCacheSparsePriorityV2, " %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a ", 
				sizeof(cl_mem), &d_i,
				sizeof(int), &offset,
				sizeof(cl_mem), &d_K,
				sizeof(cl_mem), &d_KCacheRemapIdx,
				sizeof(cl_mem), &d_KCacheRowIdx,
				sizeof(cl_mem), &d_KCacheRowPriority,
				sizeof(int), &cache_rows,
				sizeof(cl_mem), &d_cacheRow,
				sizeof(cl_mem), &d_denseVec,
				sizeof(cl_mem), &d_x2,
				sizeof(cl_mem), &sparse_data_gpu.values,
				sizeof(cl_mem), &sparse_data_gpu.rowOffsets,
				sizeof(cl_mem), &sparse_data_gpu.colInd,
				sizeof(cl_mem), &sparse_data_gpu.rowLen,
				sizeof(float), &gamma,
				sizeof(int), &num_vec,
				sizeof(int), &num_vec_aligned,
				sizeof(int), &dim,
				sizeof(int), &dim_aligned
			);
 
		} else {

			kernelCheckCachePriorityV2 = sclGetCLSoftware("kernels.cl", "kernelCheckCachePriorityV2", gpu_hardware );
			global_dense[0] = 0; global_dense[1] = DENSE_TILE_SIZE;
			local_dense[0] = DENSE_TILE_SIZE; local_dense[1] = DENSE_TILE_SIZE;
			sclSetKernelArgs( kernelCheckCachePriorityV2, " %a %a %a %a %a %a %a %a %a %a %a %a %a %a %a ", 
				sizeof(cl_mem), &d_i,
				sizeof(int), &offset,
				sizeof(cl_mem), &d_K,
				sizeof(cl_mem), &d_KCacheRemapIdx,
				sizeof(cl_mem), &d_KCacheRowIdx,
				sizeof(cl_mem), &d_KCacheRowPriority,
				sizeof(int), &cache_rows,
				sizeof(cl_mem), &d_x,
				sizeof(cl_mem), &d_x2,
				sizeof(float), &gamma,
				sizeof(int), &num_vec,
				sizeof(int), &num_vec_aligned,
				sizeof(int), &dim,
				sizeof(int), &dim_aligned,
				sizeof(cl_mem), &d_cacheRow
			);
	
		} //else sparse
	}

	if(sparse) {
		if(!firstRun) {
			sclSetKernelArg( kernelMakeDenseVec, 1, sizeof(int), &offset );
			sclSetKernelArg( kernelFindCacheRow, 1, sizeof(int), &offset );
			sclSetKernelArg( kernelCheckCacheSparsePriorityV2, 1, sizeof(int), &offset );
		}

		openclMemsetFloat( gpu_hardware, d_denseVec, 0, dim);
		sclEnqueueKernel( gpu_hardware, kernelMakeDenseVec, global_makedense, local_makedense );
		sclEnqueueKernel( gpu_hardware, kernelFindCacheRow, global_oneBlock, local_oneBlock );
		global_sparse[0] = SPARSE_TILE_SIZE * std::min<int>(256, DIV_ALIGNED(num_vec, SPARSE_TILE_SIZE));
		sclEnqueueKernel( gpu_hardware, kernelCheckCacheSparsePriorityV2, global_sparse, local_sparse );
		
	} else {

		if(!firstRun) {
			sclSetKernelArg( kernelFindCacheRow, 1, sizeof(int), &offset );
			sclSetKernelArg( kernelCheckCachePriorityV2, 1, sizeof(int), &offset );
		}

		//sclPrintDeviceArrayInt(gpu_hardware, d_KCacheRemapIdx, 2); //LLLLLLLLLLL
		sclEnqueueKernel( gpu_hardware, kernelFindCacheRow, global_oneBlock, local_oneBlock );
		//sclFinish(gpu_hardware); //LLLLLLLLLLLL
		//sclPrintDeviceArrayInt(gpu_hardware, d_cacheRow, 1); //LLLLLLLLLLL

		global_dense[0] = DENSE_TILE_SIZE * std::min<int>(256, DIV_ALIGNED(num_vec, DENSE_TILE_SIZE));
		sclEnqueueKernel( gpu_hardware, kernelCheckCachePriorityV2, global_dense, local_dense );
		//sclFinish(gpu_hardware); //LLLLLLLLLLLL
		//sclPrintDeviceArrayInt(gpu_hardware, d_KCacheChanges, 6); //LLLLLLLLLLL

	} //else sparse
	sclEnqueueKernel( gpu_hardware, kernelCheckCacheFinalize, global_one, local_one );
	//sclFinish(gpu_hardware); //LLLLLLLLLLLL

} //checkCache


void reduceMaxIdxDebug(sclHard &gpu_hardware, cl_mem d_reduceval, cl_mem d_reduceidx, unsigned int num_vec) {
	float *v = new float[num_vec];
	int *id = new int[num_vec];
	sclRead(gpu_hardware, num_vec * sizeof(float), d_reduceval, v);
	sclRead(gpu_hardware, num_vec * sizeof(int), d_reduceidx, id);
	float max_v = v[0];
	int max_id = id[0];
	for(unsigned int i=0; i < num_vec; i++) {
		if(v[i] > max_v) {
			max_v = v[i];
			max_id = id[i];
		}
	}
	sclWrite(gpu_hardware, 1 * sizeof(int), d_reduceidx, &max_id);
	delete v;
	delete id;
} //reduceMaxIdxDebug

void OpenCLSvmTrain(float * alpha, float * rho, bool sparse, const float * x, const float * y, unsigned int num_vec, unsigned int num_vec_aligned, unsigned int dim, unsigned int dim_aligned, float C, float gamma, float eps) {
	int nDevice = 0; //should be parameter in future

    cl_mem d_alpha = nullptr,
        d_x = nullptr,
        d_y = nullptr,
        d_g = nullptr,
        d_K = nullptr,
        d_KDiag = nullptr,
        d_reduceval = nullptr,
        d_reduceval2 = nullptr,
        d_reduceidx = nullptr,
        d_reduceidx2 = nullptr,
		d_lambda = nullptr,
		d_workingset = nullptr,
        d_workingset2 = nullptr,
		d_KCacheRemapIdx = nullptr,
        d_KCacheRowIdx = nullptr,  // items at index [cache_rows] and [cache_rows+1] are indices of last inserted item
        d_KCacheRowPriority = nullptr,  // the higher the priority is, the later was the item added
		d_shrinkIdx = nullptr,
		d_shrinkOrigIdx = nullptr,
		d_doShrink = nullptr,
		d_denseVec = nullptr,  //dense vector used to calculate K cache row for sparse data
		d_x2 = nullptr,
		d_KCacheChanges = nullptr,
		d_cacheUpdateCnt = nullptr,
		d_cacheRow = nullptr;

    bool usePriorityCache = true;
    bool useShrinking = false;
    size_t reduce_block_size = NUM_THREADS;
    size_t reduce_buff_size = ALIGN_UP(num_vec, reduce_block_size);
    size_t ones_size = std::max(num_vec_aligned, dim_aligned);
    size_t cache_size_mb = g_cache_size;
	int cacheUpdateCnt = 0;

	MyStopWatch cl;
	cl.start();

	int gpu_found;
	sclHard gpu_hardware = sclGetGPUHardware( nDevice, &gpu_found );
	if(gpu_found <= 0) {
		std::cerr << "OpenCL device index " << nDevice << " out of range: " << std::endl;
		return;
	}
	sclPrintDeviceNamePlatforms(&gpu_hardware, 1);

    if (cache_size_mb == 0)
    {
		cache_size_mb = OCL_CACHE_SIZE_MB; //~ to fit in 1GB, OpenCL doesn't have gpu free memory functions
    }
    unsigned int cache_rows = (unsigned int) (cache_size_mb * 1024 * 1024 / (num_vec_aligned * sizeof(float)));
	cache_rows = std::min(cache_rows, num_vec);

    std::cout << "Training data: " << (sparse ? "sparse" : "dense") << std::endl;
    std::cout << "Data size: " << num_vec << "\nDimension: " << dim << std::endl;
    std::cout << "Cache size: " << cache_rows << " rows (" << (100.f * cache_rows / (float)num_vec) << " % of data set)" << std::endl;

    const csr * sparse_data = (const csr *)x;
    csr_gpu sparse_data_gpu;
    assert_opencl_mem_void(d_x2 = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec * sizeof(float)));
    if (sparse)
    {
        make_opencl_csr(gpu_hardware, sparse_data_gpu, *sparse_data);
        assert_opencl_mem_void(d_denseVec = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, dim * sizeof(float)));
        std::cout << "Precalculating X2" << std::endl;
        computeX2Sparse(gpu_hardware, sparse_data_gpu, d_x2);
    }
    else
    {
		assert_opencl_mem_void(d_x = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec_aligned * dim_aligned * sizeof(float)));
    }

	assert_opencl_mem_void(d_alpha = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec_aligned * sizeof(float)));
	assert_opencl_mem_void(d_y = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec_aligned * sizeof(float)));
	assert_opencl_mem_void(d_g = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec_aligned * sizeof(float)));
	assert_opencl_mem_void(d_reduceval = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, reduce_buff_size / reduce_block_size * sizeof(float)));
	assert_opencl_mem_void(d_reduceidx = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, reduce_buff_size / reduce_block_size * sizeof(int)));
	assert_opencl_mem_void(d_lambda = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, sizeof(float)));
	assert_opencl_mem_void(d_workingset = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, 2 * sizeof(int)));
	assert_opencl_mem_void(d_KCacheRemapIdx = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec * sizeof(int)));
	assert_opencl_mem_void(d_KCacheRowIdx = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, (cache_rows + 2) * sizeof(int)));
	assert_opencl_mem_void(d_KCacheRowPriority = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, cache_rows * sizeof(int)));
	assert_opencl_mem_void(d_KDiag = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, num_vec * sizeof(float)));
	assert_opencl_mem_void(d_K = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, cache_rows * num_vec_aligned * sizeof(float)));
	assert_opencl_mem_void(d_KCacheChanges = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, 6 * sizeof(int)));
	assert_opencl_mem_void(d_cacheUpdateCnt = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, sizeof(int)));
	assert_opencl_mem_void(d_cacheRow = sclMalloc(gpu_hardware, CL_MEM_READ_WRITE, sizeof(int)));

    openclMemsetFloat(gpu_hardware, d_alpha, 0, num_vec_aligned);

	if (!sparse)
    {
        sclWrite(gpu_hardware, num_vec_aligned * dim_aligned * sizeof(float), d_x, (void *)x);
        std::cout << "Precalculating X2" << std::endl;
        computeX2Dense(gpu_hardware, d_x, d_x2, num_vec, num_vec_aligned, dim, dim_aligned);
    }
	sclWrite(gpu_hardware, num_vec_aligned * sizeof(float), d_y, (void *)y);

    int KCacheChanges[6];
    for (int i = 0; i < 6; i++)
        KCacheChanges[i] = -1;
    sclWrite(gpu_hardware, sizeof(int)*6, d_KCacheChanges, KCacheChanges);

    openclMemsetFloat(gpu_hardware, d_g, 1, num_vec_aligned);
    openclMemsetInt(gpu_hardware, d_KCacheRemapIdx, -1, num_vec);
    openclMemsetInt(gpu_hardware, d_KCacheRowIdx, -1, cache_rows + 2);
    openclMemsetInt(gpu_hardware, d_KCacheRowPriority, -1, cache_rows);
	sclWrite(gpu_hardware, sizeof(int), d_cacheUpdateCnt, &cacheUpdateCnt);

    std::cout << "Precalculating KDiag" << std::endl;
    computeKDiag(gpu_hardware, d_KDiag, num_vec);

    unsigned int num_vec_shrunk = num_vec;
    
	int cacheLastPtrIdx = 0;

	sclSoft kernelSelectI = sclGetCLSoftware("kernels.cl", "kernelSelectI", gpu_hardware );
	sclSoft kernelSelectJCached = sclGetCLSoftware("kernels.cl", "kernelSelectJCached", gpu_hardware );
	sclSoft kernelUpdateAlphaAndLambdaCached = sclGetCLSoftware("kernels.cl", "kernelUpdateAlphaAndLambdaCached", gpu_hardware );
	sclSoft kernelUpdategCached = sclGetCLSoftware("kernels.cl", "kernelUpdategCached", gpu_hardware );

	size_t global_size[2], local_size[2];
	size_t global_one[2], local_one[2];
    global_size[0] = ALIGN_UP(num_vec, reduce_block_size); global_size[1] = 1;
	local_size[0] = reduce_block_size; local_size[1] = 1;
    global_one[0] = 1; global_one[1] = 1;
	local_one[0] = 1; local_one[1] = 1;
	
	if ( sclFinish(gpu_hardware) != CL_SUCCESS ) {
		return;
	}

    std::cout << "Starting iterations" << std::endl;
	//for (int iter = 0; iter < 20; iter++)
    for (int iter = 0; /*iter < 100*/; iter++)
    {
		sclSetArgsEnqueueKernel( gpu_hardware, kernelSelectI, global_size, local_size, " %a %a %a %a %a %a %a ", 
			sizeof(cl_mem), &d_reduceval,
			sizeof(cl_mem), &d_reduceidx,
			sizeof(cl_mem), &d_y,
			sizeof(cl_mem), &d_g,
			sizeof(cl_mem), &d_alpha,
			sizeof(float), &C,
			sizeof(int), &num_vec_shrunk
		);
		//sclFinish(gpu_hardware); //LLLLLLLLLLLL

		//sclPrintDeviceArrayFloat(gpu_hardware, d_reduceval, 10); //LLLLLLLLLLL

		reduceMaxIdx(gpu_hardware, d_reduceval, d_reduceidx, num_vec_shrunk, d_workingset, 0);
		//reduceMaxIdxDebug(gpu_hardware, d_reduceval, d_reduceidx, num_vec_shrunk);

		//sclPrintDeviceArrayInt(gpu_hardware, d_cacheRow, 1); //LLLLLLLLLLLL

        //check if I is cached
        checkCache(gpu_hardware, sparse, d_workingset, 0, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma, d_KCacheChanges, d_cacheUpdateCnt, d_cacheRow);

		sclSetArgsEnqueueKernel( gpu_hardware, kernelSelectJCached, global_size, local_size, " %a %a %a %a %a %a %a %a %a %a %a %a ", 
			sizeof(cl_mem), &d_reduceval,
			sizeof(cl_mem), &d_reduceidx,
			sizeof(cl_mem), &d_y,
			sizeof(cl_mem), &d_g,
			sizeof(cl_mem), &d_alpha,
			sizeof(float), &C,
			sizeof(int), &num_vec_shrunk,
			sizeof(int), &num_vec_aligned,
			sizeof(cl_mem), &d_workingset,
			sizeof(cl_mem), &d_K,
			sizeof(cl_mem), &d_KDiag,
			sizeof(cl_mem), &d_KCacheRemapIdx
		);
		//sclPrintDeviceArrayInt(gpu_hardware, d_KCacheRowIdx, 3); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_K, 10); //LLLLLLLLLLL
		

		//sclPrintDeviceArrayFloat(gpu_hardware, d_reduceval, 4, 1096); //LLLLLLLLLLL
		//sclPrintDeviceArrayInt(gpu_hardware, d_reduceidx, 4, 1096); //LLLLLLLLLLL

		//sclDumpDeviceArrayFloat(gpu_hardware, "debug_values.txt", d_reduceval, num_vec_shrunk); //LLLLLLLLLLL

        reduceMaxIdx(gpu_hardware, d_reduceval, d_reduceidx, num_vec_shrunk, d_workingset, 1);
		//reduceMaxIdxDebug(gpu_hardware, d_reduceval, d_reduceidx, num_vec_shrunk);
		
		//sclPrintDeviceArrayFloat(gpu_hardware, d_reduceval2, 16); //LLLLLLLLLLL
		//sclPrintDeviceArrayInt(gpu_hardware, d_reduceidx2, 16); //LLLLLLLLLLL

		//sclCopy(gpu_hardware, sizeof(int), d_workingset, d_reduceidx, 1*sizeof(int), 0);
		//sclFinish(gpu_hardware); //LLLLLLLLLLLL
		//sclPrintDeviceArrayInt(gpu_hardware, d_workingset, 2); //LLLLLLLLLLL

		//sclPrintDeviceArrayInt(gpu_hardware, d_KCacheRemapIdx, 20); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_denseVec, 2); //LLLLLLLLLLL

        //check if J is cached
		checkCache(gpu_hardware, sparse, d_workingset, 1, d_x, d_x2, sparse_data_gpu, d_K, d_KCacheRemapIdx, d_KCacheRowIdx, d_KCacheRowPriority, d_denseVec, num_vec_shrunk, num_vec_aligned, dim, dim_aligned, cache_rows, gamma, d_KCacheChanges, d_cacheUpdateCnt, d_cacheRow);
		//sclPrintDeviceArrayFloat(gpu_hardware, d_denseVec, 2); //LLLLLLLLLLL
		//sclPrintDeviceArrayInt(gpu_hardware, d_KCacheRemapIdx, num_vec_shrunk); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_K, 4); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_K, 4, num_vec_aligned*256); //LLLLLLLLLLL

        if (iter > 0 && iter % 10000 == 0)
        {
            int ws[2];
            float yi, yj, gi, gj;
			sclRead(gpu_hardware, 2 * sizeof(int), d_workingset, ws, true);
			sclRead(gpu_hardware, sizeof(float), d_y, &yi, false, ws[0]*sizeof(float));
			sclRead(gpu_hardware, sizeof(float), d_y, &yj, false, ws[1]*sizeof(float));
			sclRead(gpu_hardware, sizeof(float), d_g, &gi, false, ws[0]*sizeof(float));
			sclRead(gpu_hardware, sizeof(float), d_g, &gj, true , ws[1]*sizeof(float));

            float diff = yi * gi - yj * gj;
            std::cout << "Iter " << iter << ": " << diff << " [" << ws[0] << "," << ws[1] << "]" << std::endl;

            if (diff < eps)
            {
                *rho = -(yi * gi + yj * gj) / 2;
                std::cout << "Optimality reached, stopping loop. rho = " << *rho << std::endl;
                break;
            }
        }

        //update all data here, not only num_vec_shrunk
		sclSetArgsEnqueueKernel( gpu_hardware, kernelUpdateAlphaAndLambdaCached, global_one, local_one, " %a %a %a %a %a %a %a %a %a %a ", 
			sizeof(cl_mem), &d_alpha,
			sizeof(cl_mem), &d_lambda,
			sizeof(cl_mem), &d_y,
			sizeof(cl_mem), &d_g,
			sizeof(cl_mem), &d_K,
			sizeof(float), &C,
			sizeof(cl_mem), &d_workingset,
			sizeof(int), &num_vec_aligned,
			sizeof(cl_mem), &d_KDiag,
			sizeof(cl_mem), &d_KCacheRemapIdx
		);
		//sclFinish(gpu_hardware); //LLLLLLLLLLLL

		//sclPrintDeviceArrayInt(gpu_hardware, d_KCacheRemapIdx, 20); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_lambda, 1); //LLLLLLLLLLL

		sclSetArgsEnqueueKernel( gpu_hardware, kernelUpdategCached, global_size, local_size, " %a %a %a %a %a %a %a %a ", 
			sizeof(cl_mem), &d_g,
			sizeof(cl_mem), &d_lambda,
			sizeof(cl_mem), &d_y,
			sizeof(cl_mem), &d_K,
			sizeof(cl_mem), &d_workingset,
			sizeof(int), &num_vec_shrunk,
			sizeof(int), &num_vec_aligned,
			sizeof(cl_mem), &d_KCacheRemapIdx
		);
		//sclFinish(gpu_hardware); //LLLLLLLLLLLL

		//sclPrintDeviceArrayFloat(gpu_hardware, d_g, 20); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_alpha, 20); //LLLLLLLLLLL
		//sclPrintDeviceArrayFloat(gpu_hardware, d_K, 20); //LLLLLLLLLLL
		
		if(iter == 0) cl.stop(); //measure openCL initialization time
    } //for iterations

	sclFinish(gpu_hardware); //LLLLLLLLLLLL

	printf("Elapsed time to initialize OpenCL and compile kenrels: %f \n", cl.getTime());

	sclRead(gpu_hardware, sizeof(int), d_cacheUpdateCnt, &cacheUpdateCnt);
    std::cout << "Cache row updates: " << cacheUpdateCnt << std::endl;

    sclRead(gpu_hardware, num_vec * sizeof(float), d_alpha, alpha);

    if (useShrinking)
    {
        sclReleaseMemObject(d_shrinkIdx);
        sclReleaseMemObject(d_shrinkOrigIdx);
        sclReleaseMemObject(d_doShrink);
    }
    if (sparse)
    {
        openclCsrFree(sparse_data_gpu);
        sclReleaseMemObject(d_denseVec);
    }
    else
    {
        sclReleaseMemObject(d_x);
    }

    sclReleaseMemObject(d_x2);
    sclReleaseMemObject(d_K);
    sclReleaseMemObject(d_KDiag);
    sclReleaseMemObject(d_KCacheRemapIdx);
    sclReleaseMemObject(d_KCacheRowIdx);
    sclReleaseMemObject(d_KCacheRowPriority);
    sclReleaseMemObject(d_alpha);
    sclReleaseMemObject(d_y);
    sclReleaseMemObject(d_g);
    sclReleaseMemObject(d_reduceval);
    sclReleaseMemObject(d_reduceidx);
    sclReleaseMemObject(d_lambda);
	sclReleaseMemObject(d_KCacheChanges);
	sclReleaseMemObject(d_cacheUpdateCnt);
	sclReleaseMemObject(d_cacheRow);

	sclReleaseClSoft( kernelSelectI );
	sclReleaseClSoft( kernelSelectJCached );
	sclReleaseClSoft( kernelUpdateAlphaAndLambdaCached );
	sclReleaseClSoft( kernelUpdategCached );

	sclReleaseClHard( gpu_hardware );

}
