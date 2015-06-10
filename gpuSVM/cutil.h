#ifndef _CUTIL_H_
#define _CUTIL_H_

#include <helper_cuda.h>

// Interface btwen cuda 5.0 and older versions using cuda utility library
#define cutilDeviceSynchronize cudaDeviceSynchronize
#define cutilSafeCall(x) checkCudaErrors(x)
#define CUDA_SAFE_CALL(x) checkCudaErrors(x)
#define cutilCheckMsg(x) getLastCudaError(x)

#endif /* _CUTIL_H_ */
