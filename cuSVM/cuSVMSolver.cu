

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <ctype.h>
#include <float.h>
#include <algorithm>
#include <math.h>
#include "cublas.h"  
#include "mex.h" 
#include "cuda.h" 
#include "cuSVMutil.h"
#include <vector>

__constant__ float C;
__constant__ float taumin;
__constant__ float kernelwidth;




template <unsigned int blockSize>
__global__ void FindBJ(float *d_F, float* d_y,float* d_alpha,float* d_KernelCol,float *g_odata,int* g_index,float BIValue, unsigned int n)
{

	__shared__ float sdata[blockSize];
	__shared__ int ind[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid]=-FLT_MAX;
	ind[tid]=0;

	float temp;
	float globaltemp;

	float LocalCloseY;
	float LocalFarY;
	float maxtemp;
	float denomclose;
	float denomfar=1.f;


	while (i < n) 
	{ 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0.f;
		denomclose=(2.f-2.f*d_KernelCol[i]);
		if(i+blockSize<n){denomfar=(2.f-2.f*d_KernelCol[i+blockSize]);}


		denomclose=denomclose<taumin?taumin:denomclose;
		denomfar=denomfar<taumin?taumin:denomfar;


		maxtemp=
			fmaxf(
			globaltemp=
			(LocalCloseY*d_alpha[i])>(LocalCloseY==1?0:-C) ?
			__fdividef(__powf(BIValue+LocalCloseY*d_F[i],2.f),denomclose)
			:-FLT_MAX, 
			i+blockSize<n ? 
			((LocalFarY*d_alpha[i+blockSize])>(LocalFarY==1?0:-C)?      
			__fdividef(__powf(BIValue+LocalFarY*d_F[i+blockSize],2.f),denomfar)
			:-FLT_MAX)
			:-FLT_MAX);

		sdata[tid]=fmaxf(temp=sdata[tid],maxtemp);

		if (sdata[tid]!=temp)
		{
			sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
		}

		i += gridSize; 
	}


	__syncthreads();

	if (tid < 128){ if (sdata[tid] < sdata[tid + 128]){ ind[tid]=ind[tid+128];sdata[tid]=sdata[tid+128];  }} __syncthreads(); 

	if (tid < 64){ if (sdata[tid] < sdata[tid + 64]){ ind[tid]=ind[tid+64];sdata[tid]=sdata[tid+64];  }} __syncthreads();   

	if (tid < 32) 
	{
		if (sdata[tid] <sdata[tid + 32]) {ind[tid]=ind[tid+32];sdata[tid]=sdata[tid+32];} __syncthreads();
		if (sdata[tid] <sdata[tid + 16]) {ind[tid]=ind[tid+16];sdata[tid]=sdata[tid+16];} __syncthreads();
		if (sdata[tid] <sdata[tid + 8]) {ind[tid]=ind[tid+8];sdata[tid]=sdata[tid+8];} __syncthreads();
		if (sdata[tid] <sdata[tid + 4]) {ind[tid]=ind[tid+4];sdata[tid]=sdata[tid+4];} __syncthreads();
		if (sdata[tid] <sdata[tid + 2]) {ind[tid]=ind[tid+2];sdata[tid]=sdata[tid+2];} __syncthreads();
		if (sdata[tid] <sdata[tid + 1]) {ind[tid]=ind[tid+1];sdata[tid]=sdata[tid+1];} __syncthreads();
	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0) g_index[blockIdx.x] = ind[0];
}

template <unsigned int blockSize>
__global__ void FindBI(float *d_F, float* d_y,float* d_alpha,float *g_odata,int* g_index,unsigned int n)
{

	__shared__ float sdata[blockSize];
	__shared__ int ind[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid]=-FLT_MAX;
	ind[tid]=0;



	float temp;
	float globaltemp;

	float LocalCloseY;
	float LocalFarY;
	float maxtemp;


	while (i < n) 
	{ 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

		maxtemp=
			fmaxf(
			globaltemp= 
			(LocalCloseY*d_alpha[i])<(LocalCloseY==1?C:0) ?  
			-(d_F[i]*LocalCloseY)  
			:-FLT_MAX, 
			i+blockSize<n ? 
			((LocalFarY*d_alpha[i+blockSize])<(LocalFarY==1?C:0) ?  
			-(d_F[i+blockSize]*LocalFarY)  
			:-FLT_MAX)
			:-FLT_MAX);

		sdata[tid]=fmaxf(temp=sdata[tid],maxtemp);

		if (sdata[tid]!=temp)
		{
			sdata[tid]== globaltemp ? ind[tid]=i : ind[tid]=i+blockSize;
		}

		i += gridSize; 
	}


	__syncthreads();

	if (tid < 128){ if (sdata[tid] < sdata[tid + 128]){ ind[tid]=ind[tid+128];sdata[tid]=sdata[tid+128];  }} __syncthreads(); 

	if (tid < 64){ if (sdata[tid] < sdata[tid + 64]){ ind[tid]=ind[tid+64];sdata[tid]=sdata[tid+64];  }} __syncthreads(); 

	if (tid < 32) 
	{
		if (sdata[tid] <sdata[tid + 32]) {ind[tid]=ind[tid+32];sdata[tid]=sdata[tid+32];} __syncthreads();
		if (sdata[tid] <sdata[tid + 16]) {ind[tid]=ind[tid+16];sdata[tid]=sdata[tid+16];} __syncthreads();
		if (sdata[tid] <sdata[tid + 8]) {ind[tid]=ind[tid+8];sdata[tid]=sdata[tid+8];} __syncthreads();
		if (sdata[tid] <sdata[tid + 4]) {ind[tid]=ind[tid+4];sdata[tid]=sdata[tid+4];} __syncthreads();
		if (sdata[tid] <sdata[tid + 2]) {ind[tid]=ind[tid+2];sdata[tid]=sdata[tid+2];} __syncthreads();
		if (sdata[tid] <sdata[tid + 1]) {ind[tid]=ind[tid+1];sdata[tid]=sdata[tid+1];} __syncthreads();

	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	if (tid == 0) g_index[blockIdx.x] = ind[0];
}






template <unsigned int blockSize>
__global__ void FindStoppingJ(float *d_F, float* d_y,float* d_alpha,float *g_odata,unsigned int n)
{

	__shared__ float sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid]=FLT_MAX;


	float LocalCloseY;
	float LocalFarY;


	while (i < n) 
	{ 
		LocalCloseY=d_y[i];
		LocalFarY=(i+blockSize)<n ? d_y[i+blockSize]:0;

		sdata[tid]=
			fminf(
			sdata[tid],
			fminf( 
			(LocalCloseY*d_alpha[i])>(LocalCloseY==1?0:-C) ?  
			-(d_F[i]*LocalCloseY)  
			:FLT_MAX, 
			i+blockSize<n ? 
			((LocalFarY*d_alpha[i+blockSize])>(LocalFarY==1?0:-C)?  
			-(d_F[i+blockSize]*LocalFarY)  
			:FLT_MAX)
			:FLT_MAX));

		i += gridSize; 
	}   


	__syncthreads();

	if (tid < 128){ sdata[tid]=fminf(sdata[tid],sdata[tid+128]);} __syncthreads(); 

	if (tid < 64){ sdata[tid]=fminf(sdata[tid],sdata[tid+64]);} __syncthreads(); 

	if (tid < 32) {
		sdata[tid]=fminf(sdata[tid],sdata[tid+32]); __syncthreads();
		sdata[tid]=fminf(sdata[tid],sdata[tid+16]); __syncthreads();
		sdata[tid]=fminf(sdata[tid],sdata[tid+8]); __syncthreads();
		sdata[tid]=fminf(sdata[tid],sdata[tid+4]); __syncthreads();
		sdata[tid]=fminf(sdata[tid],sdata[tid+2]); __syncthreads();
		sdata[tid]=fminf(sdata[tid],sdata[tid+1]); __syncthreads();


	}
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}




__global__ void UpdateF(float * F,float *KernelColI,float* KernelColJ, float* d_y,float deltaalphai,float deltaalphaj,float yi,float yj,int n)
{

	int totalThreads,ctaStart,tid;
	totalThreads = gridDim.x*blockDim.x;
	ctaStart = blockDim.x*blockIdx.x;
	tid = threadIdx.x;
	int i;

	for (i = ctaStart + tid; i < n; i += totalThreads) 
	{  
		F[i] = F[i] + yi*d_y[i]*deltaalphai*KernelColI[i]+yj*d_y[i]*deltaalphaj*KernelColJ[i];
	}


}

__global__ void RBFFinish(float *KernelCol, const float * KernelDotProd,const float* DotProd,const float* DotProdRow,const int n)
{

	int totalThreads,ctaStart,tid;
	totalThreads = gridDim.x*blockDim.x;
	ctaStart = blockDim.x*blockIdx.x;
	tid = threadIdx.x;
	int i;
	float temp;

	for (i = ctaStart + tid; i < n; i += totalThreads) 
	{



		KernelCol[i] = expf(kernelwidth*(DotProd[i]+*DotProdRow-KernelDotProd[i]*2.f));
	}


}






void RBFKernel(float *d_KernelJ,const int BJIndex,const float *d_x,const float * d_Kernel_InterRow,float *d_KernelDotProd, float *d_SelfDotProd,const int& m,const int& n,const int &nbrCtas,const int& threadsPerCta)
{

	cublasSgemv ('n', m, n, 1,d_x, m, d_Kernel_InterRow, 1, 0, d_KernelDotProd, 1);

	RBFFinish<<<nbrCtas,threadsPerCta>>>(d_KernelJ, d_KernelDotProd,d_SelfDotProd,d_SelfDotProd+BJIndex,m);

}



void CpuMaxInd(float &BIValue, int &BIIndex,const float * value_inter,const  int * index_inter,const  int n)
{

	BIValue=value_inter[0];
	BIIndex=index_inter[0];

	for(int j=0;j<n;j++)
	{
		if (value_inter[j]>BIValue)
		{
			BIValue=value_inter[j];
			BIIndex=index_inter[j];

		}
	}   

}




void CpuMaxIndSvr(float &BIValue, int &BIIndex, const  float * value_inter,const  int * index_inter,int n,const  int m)
{

	BIValue=value_inter[0];
	BIIndex=index_inter[0];

	for(int j=0;j<n;j++)
	{
		if (value_inter[j]>BIValue)
		{
			BIValue=value_inter[j];
			BIIndex=j<n/2?index_inter[j]:index_inter[j]+m;

		}
	}

}




void CpuMin(float &SJValue, float * value_inter,int n)
{

	SJValue=value_inter[0];

	for(int j=0;j<n;j++)
	{
		if (value_inter[j]<SJValue)
		{
			SJValue=value_inter[j];

		}
	}

}



void DotProdVector(float * x, float* dotprod,int m, int n)
{

	for(int i=0;i<m;i++)
	{
		dotprod[i]=0;

		for(int j=0;j<n;j++)
			dotprod[i]+=(x[i+j*m])*(x[i+j*m]);

	}



}

void IncrementKernelCache(std::vector<int>& KernelCacheItersSinceUsed,const int &RowsInKernelCache)
{
	for(int k=0;k<RowsInKernelCache;k++)
	{
		KernelCacheItersSinceUsed[k]+=1;
	}
}

inline void UpdateAlphas(float& alphai,float& alphaj,const float& Kij,const float& yi,const float& yj,const float& Fi,const float& Fj,const float& C,const float& h_taumin)
{

	//This alpha update code is adapted from that in LIBSVM.  
	//Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support vector machines, 2001. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm 

	float lambda;
	float lambda_denom;


	lambda_denom=2.0-2.0*Kij;
	if (lambda_denom<h_taumin) {lambda_denom=h_taumin;}

	if (yi!=yj)
	{
		lambda=(-Fi-Fj)/lambda_denom;
		float alphadiff=alphai-alphaj;

		alphai+=lambda;
		alphaj+=lambda;


		if(alphadiff > 0)
		{
			if(alphaj < 0)
			{
				alphaj = 0;
				alphai = alphadiff;
			}



		}
		else
		{
			if(alphai < 0)
			{
				alphai = 0;
				alphaj = -alphadiff;
			}
		}


		if(alphadiff > 0)
		{
			if(alphai > C)
			{
				alphai = C;
				alphaj = C - alphadiff;
			}
		}
		else
		{
			if(alphaj > C)
			{
				alphaj = C;
				alphai = C + alphadiff;
			}
		}


	}
	else
	{
		float alphasum=alphai+alphaj;
		lambda=(Fi-Fj)/lambda_denom;
		alphai-=lambda;
		alphaj+=lambda;

		if(alphasum > C)
		{
			if(alphai > C)
			{
				alphai = C;
				alphaj = alphasum - C;
			}
			if(alphaj > C)
			{
				alphaj = C;
				alphai = alphasum - C;
			}
		}
		else
		{
			if(alphaj < 0)
			{
				alphaj = 0;
				alphai = alphasum;
			}
			if(alphai < 0)
			{
				alphai = 0;
				alphaj = alphasum;
			}
		}

	}

}







extern "C"
void SVRTrain(float *mexalpha,float* beta,float*y,float *x ,float _C, float _kernelwidth, float eps, int m, int n, float StoppingCrit)
{




	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	mxArray *mexelapsed =mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
	float * elapsed=(float *)mxGetData(mexelapsed);    


	cudaEventRecord(start,0);

	cublasInit();


	int numBlocks=64;
	dim3 ReduceGrid(numBlocks, 1, 1);
	dim3 ReduceBlock(256, 1, 1);



	float h_taumin=0.0001;
	mxCUDA_SAFE_CALL(cudaMemcpyToSymbol(taumin, &h_taumin, sizeof(float)));

	_kernelwidth*=-1;
	mxCUDA_SAFE_CALL(cudaMemcpyToSymbol(kernelwidth, &_kernelwidth, sizeof(float)));

	mxCUDA_SAFE_CALL(cudaMemcpyToSymbol(C, &_C, sizeof(float)));


	float *alphasvr=new float [2*m];
	float *ybinary=new float [2*m];
	float *F=new float [2*m];

	for(int j=0;j<m;j++)
	{
		alphasvr[j]=0;
		ybinary[j]=1;
		F[j]=-y[j]+eps;

		alphasvr[j+m]=0;
		ybinary[j+m]=-1;
		F[j+m]=y[j]+eps; 


	}


	float *SelfDotProd=new float [m];
	DotProdVector(x, SelfDotProd,m, n);

	int nbrCtas;
	int elemsPerCta;
	int threadsPerCta;

	VectorSplay (m, SAXPY_THREAD_MIN, SAXPY_THREAD_MAX, SAXPY_CTAS_MAX, &nbrCtas, &elemsPerCta,&threadsPerCta);


	float * d_x;
	float * d_xT;
	float * d_alpha;
	float* d_y;
	float* d_F;
	float *d_KernelDotProd;
	float *d_SelfDotProd;
	float *d_KernelJ;
	float *d_KernelI;
	float* d_KernelInterRow;


	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_x, m*n*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_xT, m*n*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_x, x, sizeof(float)*n*m,cudaMemcpyHostToDevice));
	dim3 gridtranspose(ceil((float)m / TRANS_BLOCK_DIM), ceil((float)n / TRANS_BLOCK_DIM), 1);
	dim3 threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);
	cudaThreadSynchronize();
	transpose<<< gridtranspose, threadstranspose >>>(d_xT, d_x, m, n);

	float *xT=new float [n*m];   
	mxCUDA_SAFE_CALL(cudaMemcpy(xT, d_xT, sizeof(float)*m*n,cudaMemcpyDeviceToHost));
	mxCUDA_SAFE_CALL(cudaFree(d_xT));


	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelInterRow, n*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_alpha, 2*m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_y, 2*m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_F, 2*m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_SelfDotProd, m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelDotProd, m*sizeof(float)));




	mxCUDA_SAFE_CALL(cudaMemcpy(d_y, ybinary, sizeof(float)*m*2,cudaMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha, alphasvr, sizeof(float)*m*2,cudaMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_F, F, sizeof(float)*m*2,cudaMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_SelfDotProd, SelfDotProd, sizeof(float)*m,cudaMemcpyHostToDevice));



	delete [] F;
	delete [] SelfDotProd;


	float* value_inter;
	int* index_inter;
	float* value_inter_svr;
	int* index_inter_svr;



	cudaMallocHost( (void**)&value_inter, numBlocks*sizeof(float) );
	cudaMallocHost( (void**)&index_inter, numBlocks*sizeof(int) );
	cudaMallocHost( (void**)&value_inter_svr, 2*numBlocks*sizeof(float) );
	cudaMallocHost( (void**)&index_inter_svr, 2*numBlocks*sizeof(int) );




	float* d_value_inter;
	int* d_index_inter;


	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_value_inter, numBlocks*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_index_inter, numBlocks*sizeof(int)));


	size_t free_mem, total;
	cuMemGetInfo(&free_mem, &total);


	int KernelCacheSize=free_mem-MBtoLeave*1024*1024;
	int RowsInKernelCache=KernelCacheSize/(sizeof(float)*m);

	float *d_Kernel_Cache;
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_Kernel_Cache, KernelCacheSize));

	std::vector<int> KernelCacheIndices(RowsInKernelCache,-1);
	std::vector<int> KernelCacheItersSinceUsed(RowsInKernelCache,0);
	std::vector<int>::iterator CachePosI;
	std::vector<int>::iterator CachePosJ;
	int CacheDiffI;
	int CacheDiffJ;





	int CheckStoppingCritEvery=255;
	int iter=0;

	float BIValue;
	int BIIndex;
	float SJValue;
	float BJSecondOrderValue;
	int BJIndex;
	float Kij;
	float yj;
	float yi;
	float alphai;
	float alphaj;
	float oldalphai;
	float oldalphaj;
	float Fi;
	float Fj;


	while (1)
	{



		FindBI<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter,d_index_inter, 2*m);

		mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();
		CpuMaxInd(BIValue,BIIndex,value_inter,index_inter,numBlocks);

		if ((iter & CheckStoppingCritEvery)==0)
		{
			FindStoppingJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter, 2*m);

			mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
			cudaThreadSynchronize();
			CpuMin(SJValue,value_inter,numBlocks);

			if(BIValue-SJValue<StoppingCrit) {*beta=(SJValue+BIValue)/2; break;}
		}


		CachePosI=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),(BIIndex>=m?BIIndex-m:BIIndex));
		if (CachePosI ==KernelCacheIndices.end())
		{
			CacheDiffI=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelI=d_Kernel_Cache+CacheDiffI*m;
			mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+(BIIndex>=m?BIIndex-m:BIIndex)*n, n*sizeof(float),cudaMemcpyHostToDevice));
			RBFKernel(d_KernelI,(BIIndex>=m?BIIndex-m:BIIndex),d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);

			*(KernelCacheIndices.begin()+CacheDiffI)=(BIIndex>=m?BIIndex-m:BIIndex);
		}
		else
		{
			CacheDiffI=CachePosI-KernelCacheIndices.begin();
			d_KernelI=d_Kernel_Cache+m*CacheDiffI;
		}
		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=-1;




		FindBJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_KernelI,d_value_inter,d_index_inter,BIValue, m);

		mxCUDA_SAFE_CALL(cudaMemcpy(value_inter_svr, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(index_inter_svr, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));


		FindBJ<256><<<ReduceGrid, ReduceBlock>>>(d_F+m, d_y+m,d_alpha+m,d_KernelI,d_value_inter,d_index_inter,BIValue,m);

		mxCUDA_SAFE_CALL(cudaMemcpy(value_inter_svr+numBlocks, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(index_inter_svr+numBlocks, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();
		CpuMaxIndSvr(BJSecondOrderValue,BJIndex,value_inter_svr,index_inter_svr,2*numBlocks,m);


		mxCUDA_SAFE_CALL(cudaMemcpy(&Kij, d_KernelI+(BJIndex>=m?BJIndex-m:BJIndex), sizeof(float),cudaMemcpyDeviceToHost));

		mxCUDA_SAFE_CALL(cudaMemcpy(&alphai, d_alpha+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&alphaj, d_alpha+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));


		mxCUDA_SAFE_CALL(cudaMemcpy(&yi, d_y+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&yj, d_y+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&Fi, d_F+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&Fj, d_F+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));

		oldalphai=alphai;
		oldalphaj=alphaj;


		UpdateAlphas(alphai,alphaj,Kij,yi,yj,Fi,Fj,_C,h_taumin);



		mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BIIndex, &alphai, sizeof(float),cudaMemcpyHostToDevice));
		mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BJIndex, &alphaj, sizeof(float),cudaMemcpyHostToDevice));

		float deltaalphai = alphai - oldalphai;
		float deltaalphaj = alphaj - oldalphaj;




		CachePosJ=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),(BJIndex>=m?BJIndex-m:BJIndex));
		if (CachePosJ ==KernelCacheIndices.end())
		{
			CacheDiffJ=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelJ=d_Kernel_Cache+CacheDiffJ*m;
			mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+(BJIndex>=m?BJIndex-m:BJIndex)*n, n*sizeof(float),cudaMemcpyHostToDevice));
			RBFKernel(d_KernelJ,(BJIndex>=m?BJIndex-m:BJIndex),d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);


			*(KernelCacheIndices.begin()+CacheDiffJ)=(BJIndex>=m?BJIndex-m:BJIndex);
		}
		else
		{
			CacheDiffJ=CachePosJ-KernelCacheIndices.begin();
			d_KernelJ=d_Kernel_Cache+m*CacheDiffJ;

		}


		UpdateF<<<nbrCtas,threadsPerCta>>>(d_F,d_KernelI,d_KernelJ,d_y,deltaalphai,deltaalphaj,yi,yj,m);

		UpdateF<<<nbrCtas,threadsPerCta>>>(d_F+m,d_KernelI,d_KernelJ,d_y+m,deltaalphai,deltaalphaj,yi,yj,m);


		IncrementKernelCache(KernelCacheItersSinceUsed,RowsInKernelCache);

		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=0;
		*(KernelCacheItersSinceUsed.begin()+CacheDiffJ)=0;



		iter++;

	}



	cublasGetVector(m*2,sizeof(float),d_alpha,1,alphasvr,1);

	for(int k=0;k<m;k++)
	{
		mexalpha[k]=(alphasvr[k]-alphasvr[k+m])*ybinary[k];
	}




	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(elapsed, start, stop);

	mexPutVariable("base","cuSVMTrainTimeInMS",mexelapsed);   



	delete [] ybinary;
	delete [] alphasvr;
	delete [] xT;

	cudaFreeHost(value_inter_svr);
	cudaFreeHost(index_inter_svr);
	cudaFreeHost(value_inter);
	cudaFreeHost(index_inter);

	mxCUDA_SAFE_CALL(cudaFree(d_x));
	mxCUDA_SAFE_CALL(cudaFree(d_y));
	mxCUDA_SAFE_CALL(cudaFree(d_alpha));
	mxCUDA_SAFE_CALL(cudaFree(d_Kernel_Cache));
	mxCUDA_SAFE_CALL(cudaFree(d_KernelInterRow));
	mxCUDA_SAFE_CALL(cudaFree(d_F));
	mxCUDA_SAFE_CALL(cudaFree(d_value_inter));
	mxCUDA_SAFE_CALL(cudaFree(d_index_inter));
	mxCUDA_SAFE_CALL(cudaFree(d_SelfDotProd));
	mxCUDA_SAFE_CALL(cudaFree(d_KernelDotProd));
	mxCUDA_SAFE_CALL( cudaThreadExit());
	return;
}














extern "C"
void SVMTrain(float *mexalpha,float* beta,float*y,float *x ,float _C, float _kernelwidth, int m, int n, float StoppingCrit)
{

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	mxArray *mexelapsed =mxCreateNumericMatrix(1, 1,mxSINGLE_CLASS, mxREAL);
	float * elapsed=(float *)mxGetData(mexelapsed);


	cudaEventRecord(start,0);



	int numBlocks=64;
	dim3 ReduceGrid(numBlocks, 1, 1);
	dim3 ReduceBlock(256, 1, 1);





	float h_taumin=0.0001;
	mxCUDA_SAFE_CALL(cudaMemcpyToSymbol(taumin, &h_taumin, sizeof(float)));

	_kernelwidth*=-1;
	mxCUDA_SAFE_CALL(cudaMemcpyToSymbol(kernelwidth, &_kernelwidth, sizeof(float)));

	mxCUDA_SAFE_CALL(cudaMemcpyToSymbol(C, &_C, sizeof(float)));



	float *h_alpha=new float [m];
	float *h_F=new float [m];

	for(int j=0;j<m;j++)
	{
		h_alpha[j]=0;
		h_F[j]=-1;
	}


	float *SelfDotProd=new float [m];
	DotProdVector(x, SelfDotProd,m, n);

	int nbrCtas;
	int elemsPerCta;
	int threadsPerCta;

	VectorSplay (m, SAXPY_THREAD_MIN, SAXPY_THREAD_MAX, SAXPY_CTAS_MAX, &nbrCtas, &elemsPerCta,&threadsPerCta);


	float * d_x;
	float * d_xT;
	float * d_alpha;
	float* d_y;
	float* d_F;
	float *d_KernelDotProd;
	float *d_SelfDotProd;
	float *d_KernelJ;
	float *d_KernelI;

	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_x, m*n*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_xT, m*n*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_x, x, sizeof(float)*n*m,cudaMemcpyHostToDevice));
	dim3 gridtranspose(ceil((float)m / TRANS_BLOCK_DIM), ceil((float)n / TRANS_BLOCK_DIM), 1);
	dim3 threadstranspose(TRANS_BLOCK_DIM, TRANS_BLOCK_DIM, 1);
	cudaThreadSynchronize();
	transpose<<< gridtranspose, threadstranspose >>>(d_xT, d_x, m, n);

	float *xT=new float [n*m];   
	mxCUDA_SAFE_CALL(cudaMemcpy(xT, d_xT, sizeof(float)*m*n,cudaMemcpyDeviceToHost));
	mxCUDA_SAFE_CALL(cudaFree(d_xT));


	float* d_KernelInterRow;
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelInterRow, n*sizeof(float)));


	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_alpha, m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_y, m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_F, m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_SelfDotProd, m*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_KernelDotProd, m*sizeof(float)));

	mxCUDA_SAFE_CALL(cudaMemcpy(d_y, y, sizeof(float)*m,cudaMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha, h_alpha, sizeof(float)*m,cudaMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_F, h_F, sizeof(float)*m,cudaMemcpyHostToDevice));
	mxCUDA_SAFE_CALL(cudaMemcpy(d_SelfDotProd, SelfDotProd, sizeof(float)*m,cudaMemcpyHostToDevice));



	delete [] SelfDotProd;


	float* value_inter;
	int* index_inter;


	cudaMallocHost( (void**)&value_inter, numBlocks*sizeof(float) );
	cudaMallocHost( (void**)&index_inter, numBlocks*sizeof(int) );


	float* d_value_inter;
	int* d_index_inter;


	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_value_inter, numBlocks*sizeof(float)));
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_index_inter, numBlocks*sizeof(int)));

	size_t free_mem, total;
	cuMemGetInfo(&free_mem, &total);
    //free_mem = INT_MAX;

	size_t KernelCacheSize=free_mem-MBtoLeave*1024*1024;
	int RowsInKernelCache=KernelCacheSize/(sizeof(float)*m);

	/* Do not use all memory available if not needed. */
	if (RowsInKernelCache > m) {
		RowsInKernelCache = m;
		KernelCacheSize = m * sizeof(float) * m;
	}

	float *d_Kernel_Cache;
	mxCUDA_SAFE_CALL(cudaMalloc( (void**) &d_Kernel_Cache, KernelCacheSize));


	std::vector<int> KernelCacheIndices(RowsInKernelCache,-1);
	std::vector<int> KernelCacheItersSinceUsed(RowsInKernelCache,0);
	std::vector<int>::iterator CachePosI;
	std::vector<int>::iterator CachePosJ;
	int CacheDiffI;
	int CacheDiffJ;





	int CheckStoppingCritEvery=255;
	int iter=0;

	float BIValue;
	int BIIndex;
	float SJValue;
	float BJSecondOrderValue;
	int BJIndex;
	float Kij;
	float yj;
	float yi;
	float alphai;
	float alphaj;
	float oldalphai;
	float oldalphaj;
	float Fi;
	float Fj;




	while (1)
	{

		FindBI<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter,d_index_inter, m);
		mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();
		CpuMaxInd(BIValue,BIIndex,value_inter,index_inter,numBlocks);





		cudaMemcpy(&Fi, d_F+BIIndex, sizeof(float),cudaMemcpyDeviceToHost);

		if ((iter & CheckStoppingCritEvery)==0)
		{
			FindStoppingJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_value_inter, m);
			mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
			cudaThreadSynchronize();
			CpuMin(SJValue,value_inter,numBlocks);




			if(BIValue-SJValue<StoppingCrit) 
			{


				if(BIValue-SJValue<StoppingCrit) {*beta=(SJValue+BIValue)/2; break;}

			}
		}




		CachePosI=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BIIndex);
		if (CachePosI ==KernelCacheIndices.end())
		{
			CacheDiffI=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelI=d_Kernel_Cache+CacheDiffI*m;
			mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+BIIndex*n, n*sizeof(float),cudaMemcpyHostToDevice));
			RBFKernel(d_KernelI,BIIndex,d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);
			*(KernelCacheIndices.begin()+CacheDiffI)=BIIndex;
		}
		else
		{
			CacheDiffI=CachePosI-KernelCacheIndices.begin();
			d_KernelI=d_Kernel_Cache+m*CacheDiffI;
		}
		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=-1;




		FindBJ<256><<<ReduceGrid, ReduceBlock>>>(d_F, d_y,d_alpha,d_KernelI,d_value_inter,d_index_inter,BIValue, m);
		mxCUDA_SAFE_CALL(cudaMemcpy(value_inter, d_value_inter, sizeof(float)*numBlocks,cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(index_inter, d_index_inter, sizeof(int)*numBlocks,cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();
		CpuMaxInd(BJSecondOrderValue,BJIndex,value_inter,index_inter,numBlocks);


		mxCUDA_SAFE_CALL(cudaMemcpy(&Kij, d_KernelI+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));

		mxCUDA_SAFE_CALL(cudaMemcpy(&alphai, d_alpha+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&alphaj, d_alpha+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));

		mxCUDA_SAFE_CALL(cudaMemcpy(&yi, d_y+BIIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&yj, d_y+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));
		mxCUDA_SAFE_CALL(cudaMemcpy(&Fj, d_F+BJIndex, sizeof(float),cudaMemcpyDeviceToHost));


		oldalphai=alphai;
		oldalphaj=alphaj;


		UpdateAlphas(alphai,alphaj,Kij,yi,yj,Fi,Fj,_C,h_taumin);



		mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BIIndex, &alphai, sizeof(float),cudaMemcpyHostToDevice));
		mxCUDA_SAFE_CALL(cudaMemcpy(d_alpha+BJIndex, &alphaj, sizeof(float),cudaMemcpyHostToDevice));

		float deltaalphai = alphai - oldalphai;
		float deltaalphaj = alphaj - oldalphaj;




		CachePosJ=find(KernelCacheIndices.begin(),KernelCacheIndices.end(),BJIndex);
		if (CachePosJ ==KernelCacheIndices.end())
		{
			CacheDiffJ=max_element(KernelCacheItersSinceUsed.begin(),KernelCacheItersSinceUsed.end())-KernelCacheItersSinceUsed.begin();
			d_KernelJ=d_Kernel_Cache+CacheDiffJ*m;
			mxCUDA_SAFE_CALL(cudaMemcpy(d_KernelInterRow, xT+BJIndex*n, n*sizeof(float),cudaMemcpyHostToDevice));
			RBFKernel(d_KernelJ,BJIndex,d_x,d_KernelInterRow,d_KernelDotProd,d_SelfDotProd,m,n,nbrCtas,threadsPerCta);
			*(KernelCacheIndices.begin()+CacheDiffJ)=BJIndex;
		}
		else
		{
			CacheDiffJ=CachePosJ-KernelCacheIndices.begin();
			d_KernelJ=d_Kernel_Cache+m*CacheDiffJ;

		}



		UpdateF<<<nbrCtas,threadsPerCta>>>(d_F,d_KernelI,d_KernelJ,d_y,deltaalphai,deltaalphaj,yi,yj,m);

		IncrementKernelCache(KernelCacheItersSinceUsed,RowsInKernelCache);

		*(KernelCacheItersSinceUsed.begin()+CacheDiffI)=0;
		*(KernelCacheItersSinceUsed.begin()+CacheDiffJ)=0;



		iter++;

	}



	cublasGetVector(m,sizeof(float),d_alpha,1,mexalpha,1);



	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(elapsed, start, stop);




	mexPutVariable("base","cuSVMTrainTimeInMS",mexelapsed);



	delete [] xT;
	cudaFreeHost(value_inter);
	cudaFreeHost(index_inter);

	mxCUDA_SAFE_CALL(cudaFree(d_x));
	mxCUDA_SAFE_CALL(cudaFree(d_y));
	mxCUDA_SAFE_CALL(cudaFree(d_alpha));
	mxCUDA_SAFE_CALL(cudaFree(d_KernelInterRow));
	mxCUDA_SAFE_CALL(cudaFree(d_Kernel_Cache));
	mxCUDA_SAFE_CALL(cudaFree(d_F));
	mxCUDA_SAFE_CALL(cudaFree(d_value_inter));
	mxCUDA_SAFE_CALL(cudaFree(d_index_inter));
	mxCUDA_SAFE_CALL(cudaFree(d_SelfDotProd));
	mxCUDA_SAFE_CALL(cudaFree(d_KernelDotProd));
	mxCUDA_SAFE_CALL(cudaThreadExit());
	return;
}

