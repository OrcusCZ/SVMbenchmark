#ifndef FRAMEWORK
#define FRAMEWORK

#define VERSION 0.2

#define BLOCKSIZE 256
#define IMUL(a, b) __mul24(a, b)
#define MAX_PITCH 262144
#define MAX_POINTS (MAX_PITCH/sizeof(float) - 2)

#define intDivideRoundUp(a, b) (a%b!=0)?(a/b+1):(a/b)

//added instead of SDK
#define CUDA_SAFE_CALL(call) {cudaError_t err = call; if (cudaSuccess != err) { fprintf (stderr, "Cuda error in file '%s' in line %i : %s.",__FILE__, __LINE__, cudaGetErrorString(err) ); exit(EXIT_FAILURE);}}


#ifdef __DEVICE_EMULATION__
#define SYNC __syncthreads()
#else
#define SYNC 
#endif

#define REDUCE0  0x00000001
#define REDUCE1  0x00000002
#define REDUCE2  0x00000004
#define REDUCE3  0x00000008
#define REDUCE4  0x00000010
#define REDUCE5  0x00000020
#define REDUCE6  0x00000040
#define REDUCE7  0x00000080
#define REDUCE8  0x00000100
#define REDUCE9  0x00000200
#define REDUCE10 0x00000400
#define REDUCE11 0x00000800
#define REDUCE12 0x00001000
#define REDUCE13 0x00002000
#define REDUCE14 0x00004000
#define REDUCE15 0x00008000
#define REDUCE16 0x00010000
#define REDUCE17 0x00020000
#define REDUCE18 0x00040000
#define REDUCE19 0x00080000
#define REDUCE20 0x00100000
#define REDUCE21 0x00200000
#define REDUCE22 0x00400000
#define REDUCE23 0x00800000
#define REDUCE24 0x01000000
#define REDUCE25 0x02000000
#define REDUCE26 0x04000000
#define REDUCE27 0x08000000
#define REDUCE28 0x10000000
#define REDUCE29 0x20000000
#define REDUCE30 0x40000000
#define REDUCE31 0x80000000
#define NOREDUCE 0x00000000
#define INFTY  __int_as_float(0x7f000000)
#define NINFTY __int_as_float(0xff000000)


#endif
