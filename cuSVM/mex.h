#ifndef _MEX_H_
#define _MEX_H_

/* Some macros redefining (unincluded) Matlab mex functions. */
#define mxArray float
#define mxCreateNumericMatrix(X,Y,CLASS,TYPE) (float *) malloc((X) * (Y) * sizeof(float))
#define mxGetData(X) (X)
#define mxCUDA_SAFE_CALL(X) { \
	cudaError err = X; \
	if (err != cudaSuccess) { \
		printf("Unable to call cuda function: %s!\n", cudaGetErrorString(err)); \
		exit(err); \
	} \
}
#define mexPutVariable(BASE,NAME,FROM) free(FROM)

#endif /* _MEX_H_ */