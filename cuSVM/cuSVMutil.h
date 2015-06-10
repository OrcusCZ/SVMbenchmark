#ifndef _CUSVMUTIL_H_
#define _CUSVMUTIL_H_


/* Macros from "cuSVMutil.h". */
#define MBtoLeave         (200)

#define CUBIC_ROOT_MAX_OPS         (2000)

#define SAXPY_CTAS_MAX           (80)
#define SAXPY_THREAD_MIN         (32)
#define SAXPY_THREAD_MAX         (128)
#define TRANS_BLOCK_DIM             (16)

__global__ void transpose(float *odata, float *idata, int width, int height)
{
	__shared__ float block[TRANS_BLOCK_DIM][TRANS_BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * TRANS_BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TRANS_BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TRANS_BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * TRANS_BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
}

void VectorSplay (int n, int tMin, int tMax, int gridW, int *nbrCtas, 
                        int *elemsPerCta, int *threadsPerCta)
{
    if (n < tMin) {
        *nbrCtas = 1;
        *elemsPerCta = n;
        *threadsPerCta = tMin;
    } else if (n < (gridW * tMin)) {
        *nbrCtas = ((n + tMin - 1) / tMin);
        *threadsPerCta = tMin;
        *elemsPerCta = *threadsPerCta;
    } else if (n < (gridW * tMax)) {
        int grp;
        *nbrCtas = gridW;
        grp = ((n + tMin - 1) / tMin);
        *threadsPerCta = (((grp + gridW -1) / gridW) * tMin);
        *elemsPerCta = *threadsPerCta;
    } else {
        int grp;
        *nbrCtas = gridW;
        *threadsPerCta = tMax;
        grp = ((n + tMin - 1) / tMin);
        grp = ((grp + gridW - 1) / gridW);
        *elemsPerCta = grp * tMin;
    }
}


#endif /* _CUSVMUTIL_H_ */
