#include "../../include/testing.h"
#include "../common/shared_kernel.cu"
#include "testing_kernel.cu"


/**
 * Tests the multiclass Support Vector Machine
 * @param h_xtraindata host pointer to the training set
 * @param h_xtestdata host pointer to the testing set
 * @param h_ltraindata host pointer to the labels of the training set
 * @param h_ltesthatdata host pointer to the estimated labels of the testing set
 * @param h_rdata host pointer to the binary matrix that encodes the output code
 * @param h_atraindata host pointer that will contain the values of the alphas
 * @param ntraining number of training samples in the training set
 * @param nfeatures number of features in each sample of the training set
 * @param nclasses number of classes of the multiclass problem
 * @param ntasks number of binary tasks to be solved
 * @param h_b host pointer to the offset parameter of each binary task
 * @param beta if using RBF kernel, the value of beta
 * @param a if using polynomial or sigmoid kernel the value of a x_i x_j
 * @param b if using polynomial or sigmoid kernel the value of b
 * @param d if using polynomial kernel
 * @param kernelcode type of kernel to use
 */

void testingclassifier(		float* h_xtraindata,
							float* h_xtestdata,
							int* h_ltraindata,
							int* h_ltesthatdata,
							int* h_rdata,
							float* h_atraindata,
							int ntraining,
							int ntesting,
							int nfeatures,
							int nclasses,
							int ntasks,
							float* h_b,
							float beta,
							float a,
							float b,
							float d,
							int kernelcode)
{


		int numThreads = (ntraining < MAXTHREADS*2) ? nextPow2((ntraining + 1)/ 2) : MAXTHREADS;
		int numBlocks = (ntraining + (numThreads * 2 - 1)) / (numThreads * 2);
		int numBlocksRed = min(MAXBLOCKS, numBlocks);

		size_t remainingMemory;
		size_t totalMemory;


		//printf("Allocating Device Memory...\n");

		// Allocate device memory for X
		float* d_xtraindata=0;
		cudaMalloc((void**) &d_xtraindata, sizeof(float) * ntraining* nfeatures);
		cudaMemcpy(d_xtraindata, h_xtraindata, sizeof(float) * ntraining* nfeatures,cudaMemcpyHostToDevice);


		//Allocate device memory for labels
		int* d_ltraindata=0;
		cudaMalloc((void**) &d_ltraindata, sizeof(int) * ntraining);
		cudaMemcpy(d_ltraindata, h_ltraindata, sizeof(int) * ntraining,cudaMemcpyHostToDevice);

		//Allocate device memory for codes
		int* d_rdata=0;
		cudaMalloc((void**) &d_rdata, sizeof(int) * nclasses * ntasks);
		cudaMemcpy(d_rdata, h_rdata, sizeof(int) * nclasses * ntasks,cudaMemcpyHostToDevice);

		// Allocate device memory for Y
		int * d_ytraindata=0;
		cudaMalloc((void**) &d_ytraindata, sizeof(int) * ntraining * ntasks);

		float* h_ytestdata= (float*) malloc(sizeof(float) * ntesting * nclasses);
		float * d_ytestdata=0;
		cudaMalloc((void**) &d_ytestdata, sizeof(float) * ntesting * nclasses);

		// Allocate device memory for A
		float * d_atraindata=0;
		cudaMalloc((void**) &d_atraindata, sizeof(float) * ntraining * ntasks);
		cudaMemcpy(d_atraindata, h_atraindata, sizeof(float) * ntraining * ntasks,cudaMemcpyHostToDevice);

		// Allocate device memory for F
		float* h_fdata= (float*) malloc(sizeof(float) * ntesting * ntasks);
		float* d_fdata=0;
		cudaMalloc((void**) &d_fdata, sizeof(float) * ntesting * ntasks);

		//Allocate device memory for b
		float * d_b=0;
		cudaMalloc((void**) &d_b, sizeof(float) * ntasks);
		cudaMemcpy(d_b, h_b, sizeof(float) * ntasks,cudaMemcpyHostToDevice);

		//Allocate memory for the reduction operation
		float* h_reduction= (float*) malloc(sizeof(float) *numBlocksRed * ntasks);
		float* d_reduction=0;
		cudaMalloc((void**) &d_reduction, sizeof(float) *numBlocksRed * ntasks);


		void* temp;
		size_t rowPitch;
		cudaMallocPitch(&temp, &rowPitch, ntraining*sizeof(float), 2);
		cudaFree(temp);

		cuMemGetInfo(&remainingMemory, &totalMemory);

		printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);

		size_t sizeOfCache = remainingMemory/((int)rowPitch);

		sizeOfCache = (size_t)((float)sizeOfCache*0.80);

		if (ntesting < sizeOfCache)
		{
			sizeOfCache = ntesting;
		}

		#ifdef __DEVICE_EMULATION__
		sizeOfCache = ntesting;
		#endif

		printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, (int)rowPitch);

		float* d_kdata;
		size_t cachePitch;


		printf("numThreads %i, numBlocksRed %i\n", numThreads, numBlocksRed);
		dim3 dimBlockInitialize(numThreads, 1, 1);
		dim3 dimGridInitialize(numBlocksRed, ntasks, 1);

		int smemSize = 0;
		bool isNtrainingPow2=isPow2(ntraining);

		if(isNtrainingPow2)
		{
			switch (numThreads)
			{
				case 512:
					generatebinarylabels <512,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 256:
					generatebinarylabels <256,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 128:
					generatebinarylabels <128,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 64:
					generatebinarylabels <64,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 32:
					generatebinarylabels <32,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 16:
					generatebinarylabels <16,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  8:
					generatebinarylabels <8,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  4:
					generatebinarylabels <4,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  2:
					generatebinarylabels <2,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  1:
					generatebinarylabels <1,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
			}
		}
		else
		{
			switch (numThreads)
			{
				case 512:
					generatebinarylabels <512,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 256:
					generatebinarylabels <256,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 128:
					generatebinarylabels <128,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 64:
					generatebinarylabels <64,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 32:
					generatebinarylabels <32,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case 16:
					generatebinarylabels <16,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  8:
					generatebinarylabels <8,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  4:
					generatebinarylabels <4,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  2:
					generatebinarylabels <2,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
				case  1:
					generatebinarylabels <1,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(	d_ltraindata,d_rdata,d_ytraindata,ntraining,ntasks); break;
			}
		}

		cudaError_t errgen = cudaGetLastError();
		if(errgen)
		{
			printf("Error: %s\n", cudaGetErrorString(errgen));
		}

		cudaThreadSynchronize();


		int ntestingfrag=sizeOfCache;
		int prevntestingfrag=0;



		while(true)
		{

			int ntestsize= ntestingfrag - prevntestingfrag;

			float* d_xtestdata=0;
			float* h_xtestdatatemp=0;
			cudaMalloc((void**) &d_xtestdata, sizeof(float) * ntestsize * nfeatures);


			h_xtestdatatemp = (float*) malloc(sizeof(float) * ntestsize * nfeatures);

			int k=0;
			for (int i=prevntestingfrag; i< ntestingfrag; i++)
			{
				for (int j=0; j<nfeatures; j++)
				{
					h_xtestdatatemp[j * ntestsize + k]= h_xtestdata[j * ntesting + i];
				}
				k++;
			}

			cudaMemcpy(d_xtestdata, h_xtestdatatemp, sizeof(float) * ntestsize * nfeatures, cudaMemcpyHostToDevice);


			cudaMallocPitch((void**)&d_kdata, &cachePitch, ntraining* sizeof(float), ntestsize);


			cudaError_t err = cudaGetLastError();
			if(err)
			{
				printf("Error: %s\n", cudaGetErrorString(err));
			}

			cublasSgemm('n', 't', ntraining, ntestsize, nfeatures, 1, d_xtraindata, ntraining, d_xtestdata, ntestsize, 0, d_kdata, ntraining);
			cudaThreadSynchronize();

			cublasStatus status = cublasGetError();
			if (status != CUBLAS_STATUS_SUCCESS) {
				fprintf (stderr, "!!!! kernel execution error.\n");
				break;
			}

			if(kernelcode==0)
			{

				float* d_dottraindata=0;
				cudaMalloc((void**) &d_dottraindata, sizeof(float) * ntraining);

				dim3 dimBlockTrainSelfDot(numThreads, 1, 1);
				dim3 dimGridTrainSelfDot(numBlocksRed, 1, 1);

				if(isNtrainingPow2)
				{
					switch (numThreads)
					{
						case 512:
							makeselfdot <512,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 256:
							makeselfdot <256,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 128:
							makeselfdot <128,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 64:
							makeselfdot <64,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 32:
							makeselfdot <32,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 16:
							makeselfdot <16,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  8:
							makeselfdot <8,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  4:
							makeselfdot <4,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  2:
							makeselfdot <2,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  1:
							makeselfdot <1,true><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
					}
				}
				else
				{
					switch (numThreads)
					{
						case 512:
							makeselfdot <512,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 256:
							makeselfdot <256,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 128:
							makeselfdot <128,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 64:
							makeselfdot <64,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 32:
							makeselfdot <32,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case 16:
							makeselfdot <16,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  8:
							makeselfdot <8,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  4:
							makeselfdot <4,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  2:
							makeselfdot <2,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
						case  1:
							makeselfdot <1,false><<< dimGridTrainSelfDot, dimBlockTrainSelfDot, smemSize >>>(d_xtraindata, d_dottraindata,nfeatures,ntraining); break;
					}
				}
				cudaThreadSynchronize();


				float* d_dottestdata=0;
				cudaMalloc((void**) &d_dottestdata, sizeof(float) * ntestsize);

				int numThreadsTest = (ntestsize < MAXTHREADS*2) ? nextPow2((ntestsize + 1)/ 2) : MAXTHREADS;
				int numBlocksTest = (ntestsize + (numThreadsTest * 2 - 1)) / (numThreadsTest * 2);
				int numBlocksRedTest = min(MAXBLOCKS, numBlocksTest);

				dim3 dimBlockTestSelfDot(numThreadsTest, 1, 1);
				dim3 dimGridTestSelfDot(numBlocksRedTest, 1, 1);

				bool isNtestingPow2=isPow2(ntestsize);

				if(isNtestingPow2)
				{
					switch (numThreadsTest)
					{
						case 512:
							makeselfdot <512,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 256:
							makeselfdot <256,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 128:
							makeselfdot <128,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 64:
							makeselfdot <64,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 32:
							makeselfdot <32,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 16:
							makeselfdot <16,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  8:
							makeselfdot <8,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  4:
							makeselfdot <4,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  2:
							makeselfdot <2,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  1:
							makeselfdot <1,true><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
					}
				}
				else
				{
					switch (numThreadsTest)
					{
						case 512:
							makeselfdot <512,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 256:
							makeselfdot <256,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 128:
							makeselfdot <128,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 64:
							makeselfdot <64,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 32:
							makeselfdot <32,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case 16:
							makeselfdot <16,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  8:
							makeselfdot <8,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  4:
							makeselfdot <4,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  2:
							makeselfdot <2,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
						case  1:
							makeselfdot <1,false><<< dimGridTestSelfDot, dimBlockTestSelfDot, smemSize >>>(d_xtestdata, d_dottestdata,nfeatures,ntestsize); break;
					}
				}
				cudaThreadSynchronize();


				for ( int i=0; i< ntestsize; i++)
				{

					dim3 dimBlockReduction(numThreads, 1, 1);
					dim3 dimGridReduction(numBlocksRed, ntasks, 1);

					if(isNtrainingPow2)
					{
						switch (numThreads)
						{
							case 512:
								reductionrbf <512,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction, d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 256:
								reductionrbf <256,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 128:
								reductionrbf <128,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 64:
								reductionrbf <64,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 32:
								reductionrbf <32,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 16:
								reductionrbf <16,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  8:
								reductionrbf <8,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  4:
								reductionrbf <4,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  2:
								reductionrbf <2,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  1:
								reductionrbf <1,true><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
						}
					}
					else
					{
						switch (numThreads)
						{
							case 512:
								reductionrbf <512,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 256:
								reductionrbf <256,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 128:
								reductionrbf <128,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 64:
								reductionrbf <64,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 32:
								reductionrbf <32,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case 16:
								reductionrbf <16,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  8:
								reductionrbf <8,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  4:
								reductionrbf <4,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  2:
								reductionrbf <2,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
							case  1:
								reductionrbf <1,false><<< dimGridReduction, dimBlockReduction, smemSize >>>(d_ytraindata,d_atraindata,d_reduction,d_dottestdata, d_dottraindata,d_kdata,ntraining,ntestsize,i,beta,a,b,d,kernelcode); break;
						}
					}
					cudaThreadSynchronize();

					cudaError_t errReduction = cudaGetLastError();
					if(errReduction)
					{
						printf("Error Reduction: %s\n", cudaGetErrorString(errReduction));
					}

					cudaMemcpy(h_reduction, d_reduction, sizeof(float) * numBlocksRed * ntasks,cudaMemcpyDeviceToHost);



					for (int j=0; j<ntasks; j++)
					{
						float sum=0;

						for (int k=0; k<numBlocksRed; k++)
						{
							sum+=h_reduction[j*numBlocksRed +k];
						}

						h_fdata[j * ntesting + prevntestingfrag + i]= sum + h_b[j];

					}



					for (int m=0; m< nclasses; m++)
					{
						h_ytestdata[m*ntesting +prevntestingfrag + i]=0;

						for (int n=0; n< ntasks; n++)
						{
							h_ytestdata[m*ntesting +prevntestingfrag + i]+= h_rdata[m*ntasks + n] * h_fdata[n*ntesting +prevntestingfrag + i];
						}
					}


				}

				cudaFree(d_dottraindata);
				cudaFree(d_dottestdata);
			}


			cudaFree(d_xtestdata);
			cudaFree(d_kdata);
			free(h_xtestdatatemp);

			if(ntestingfrag==ntesting)
			{

				break;
			}

			//Update
			prevntestingfrag=ntestingfrag;
			ntestingfrag+= sizeOfCache;

			if(ntestingfrag> ntesting )
			{
				ntestingfrag=ntesting;
			}


		}



		for (int i=0; i<ntesting; i++)
		{
			int maxj=0;
			float maxvalue=h_ytestdata[i];

			for (int j=0; j<nclasses; j++)
			{

				if(h_ytestdata[j*ntesting +i]> maxvalue)
				{
					maxvalue=h_ytestdata[j*ntesting +i];
					maxj=j;
				}
			}

			h_ltesthatdata[i]=maxj+1;
		}



		cudaFree(d_rdata);
		cudaFree(d_xtraindata);
		cudaFree(d_ltraindata);
		cudaFree(d_ytraindata);
		cudaFree(d_ytestdata);
		cudaFree(d_atraindata);
		cudaFree(d_fdata);
		cudaFree(d_b);
		cudaFree(d_reduction);

		free(h_ytestdata);
		free(h_fdata);
		free(h_reduction);


}
