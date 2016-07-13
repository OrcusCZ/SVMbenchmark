#include "csr.h"
#include "../cudaerror.h"
#include "../cuda_utils.h"
#include <vector>
#include <numeric>

static __global__ void kernelCalcRowLen(unsigned int * rowLen, const unsigned int * rowOffsets, int numRows)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < numRows)
    {
        rowLen[k] = rowOffsets[k + 1] - rowOffsets[k];
    }
}

//make a GPU deep copy of a CPU csr matrix
void OrcusSVM1B::makeCudaCsr(csr_gpu & x_gpu, const csr & x_cpu)
{
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
}

void OrcusSVM1B::freeCudaCsr(csr_gpu & x_gpu)
{
	cudaFree(x_gpu.values);
	cudaFree(x_gpu.colInd);
	cudaFree(x_gpu.rowOffsets);
    cudaFree(x_gpu.rowLen);
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
	x_gpu.rowOffsets = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
}

void OrcusSVM1B::makeCudaEllpack(ellpack_gpu & x_gpu, const csr & x_cpu)
{
    x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;

    std::vector<int> rowLen(x_cpu.numRows);
    std::adjacent_difference(x_cpu.rowOffsets + 1, x_cpu.rowOffsets + x_cpu.numRows + 1, rowLen.begin());
    x_gpu.maxRowLen = *std::max_element(rowLen.begin(), rowLen.end());

    assert_cuda(cudaMalloc((void **)&(x_gpu.values), x_gpu.numRows * x_gpu.maxRowLen * sizeof(float)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.colInd), x_gpu.numRows * x_gpu.maxRowLen * sizeof(int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowLen), x_gpu.numRows * sizeof(int)));

    assert_cuda(cudaMemcpy(x_gpu.rowLen, &rowLen[0], x_gpu.numRows * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> values_ellpack(x_gpu.numRows * x_gpu.maxRowLen);
    std::vector<int> colInd_ellpack(x_gpu.numRows * x_gpu.maxRowLen);
    for (int row = 0; row < x_gpu.numRows; row++)
    {
        int end = x_cpu.rowOffsets[row + 1];
        for (int i = x_cpu.rowOffsets[row], j = 0; i < end; i++, j++)
        {
            values_ellpack[x_cpu.numRows * j + row] = x_cpu.values[i];
            colInd_ellpack[x_cpu.numRows * j + row] = x_cpu.colInd[i];
        }
    }

    assert_cuda(cudaMemcpy(x_gpu.values, &values_ellpack[0], x_gpu.numRows * x_gpu.maxRowLen * sizeof(float), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.colInd, &colInd_ellpack[0], x_gpu.numRows * x_gpu.maxRowLen * sizeof(int), cudaMemcpyHostToDevice));
}

void OrcusSVM1B::freeCudaEllpack(ellpack_gpu & x_gpu)
{
	cudaFree(x_gpu.values);
	cudaFree(x_gpu.colInd);
    cudaFree(x_gpu.rowLen);
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
    x_gpu.rowLen = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
    x_gpu.maxRowLen = 0;
}

template<typename T>
class IdxComparator
{
    T * v;
public:
    IdxComparator(T * v) : v(v) {}
    bool operator()(int i1, int i2)
    {
        return (*v)[i1] > (*v)[i2];
    }
};

void OrcusSVM1B::makeCudaJds(jds_gpu & x_gpu, const csr & x_cpu)
{
    x_gpu.nnz = x_cpu.nnz;
	x_gpu.numCols = x_cpu.numCols;
	x_gpu.numRows = x_cpu.numRows;

    std::vector<int> rowLen(x_cpu.numRows);
    std::adjacent_difference(x_cpu.rowOffsets + 1, x_cpu.rowOffsets + x_cpu.numRows + 1, rowLen.begin());
    x_gpu.maxRowLen = *std::max_element(rowLen.begin(), rowLen.end());

    std::vector<int> rowPerm(rowLen.size());
    std::iota(rowPerm.begin(), rowPerm.end(), 0);
    std::sort(rowPerm.begin(), rowPerm.end(), IdxComparator<std::vector<int>>(&rowLen));
    std::vector<int> rowLenSorted(rowLen.size());
    for (int i = 0; i < rowPerm.size(); i++)
        rowLenSorted[i] = rowLen[rowPerm[i]];

    assert_cuda(cudaMalloc((void **)&(x_gpu.values), x_gpu.nnz * sizeof(float)));
	assert_cuda(cudaMalloc((void **)&(x_gpu.colInd), x_gpu.nnz * sizeof(int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowLen), x_gpu.numRows * sizeof(int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.rowPerm), x_gpu.numRows * sizeof(int)));
    assert_cuda(cudaMalloc((void **)&(x_gpu.colStart), x_gpu.numCols * sizeof(int)));

    assert_cuda(cudaMemcpy(x_gpu.rowLen, &rowLenSorted[0], x_gpu.numRows * sizeof(int), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.rowPerm, &rowPerm[0], x_gpu.numRows * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> values_jds(x_gpu.nnz);
    std::vector<int> colInd_jds(x_gpu.nnz);
    std::vector<int> colStart(x_gpu.maxRowLen);
    int out_idx = 0;
    for (int col = 0; col < x_gpu.maxRowLen; col++)
    {
        colStart[col] = out_idx;
        for (int row = 0; row < x_gpu.numRows; row++)
        {
            if (rowLenSorted[row] <= col)
                continue;
            int i = x_cpu.rowOffsets[rowPerm[row]] + col;
            values_jds[out_idx] = x_cpu.values[i];
            colInd_jds[out_idx] = x_cpu.colInd[i];
            out_idx++;
        }
    }

    assert_cuda(cudaMemcpy(x_gpu.colStart, &colStart[0], x_gpu.maxRowLen * sizeof(int), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.values, &values_jds[0], x_gpu.nnz * sizeof(float), cudaMemcpyHostToDevice));
    assert_cuda(cudaMemcpy(x_gpu.colInd, &colInd_jds[0], x_gpu.nnz * sizeof(int), cudaMemcpyHostToDevice));
}

void OrcusSVM1B::freeCudaJds(jds_gpu & x_gpu)
{
	cudaFree(x_gpu.values);
	cudaFree(x_gpu.colInd);
    cudaFree(x_gpu.rowLen);
    cudaFree(x_gpu.rowPerm);
    cudaFree(x_gpu.colStart);
	x_gpu.values = NULL;
	x_gpu.colInd = NULL;
    x_gpu.rowLen = NULL;
    x_gpu.rowPerm = NULL;
    x_gpu.colStart = NULL;
	x_gpu.nnz = 0;
	x_gpu.numRows = 0;
	x_gpu.numCols = 0;
    x_gpu.maxRowLen = 0;
}