#include "../../include/training.h"
#include "../common/shared_kernel.cu"
#include "training_kernel.cu"
#include "reduction_kernel.cu"
#include "globalparams.cu"
#include "reduction.cu"
#include "../common/Cache.cpp"
#include "../common/kernelevaluation.cpp"
#include "../common/utilities.cpp"

/**
 * Trains the multiclass Support Vector Machine
 * @param h_xtraindata host pointer to the training set
 * @param h_ltraindata host pointer to the labels of the training set
 * @param h_rdata host pointer to the binary matrix that encodes the output code
 * @param h_atraindata host pointer that will contain the values of the alphas
 * @param ntraining number of training samples in the training set
 * @param nfeatures number of features in each sample of the training set
 * @param nclasses number of classes of the multiclass problem
 * @param ntasks number of binary tasks to be solved
 * @param h_C host pointer to the regularization parameters for each binary task
 * @param h_b host pointer to the offset parameter of each binary task
 * @param tau stopping parameter of the SMO algorithm
 * @param kernelcode type of kernel to use
 * @param beta if using RBF kernel, the value of beta
 * @param a if using polynomial or sigmoid kernel the value of a x_i x_j
 * @param b if using polynomial or sigmoid kernel the value of b
 */
void trainclassifier (          float* h_xtraindata,
                                int* h_ltraindata,
                                int* h_rdata,
                                float* h_atraindata,
                                int ntraining,
                                int nfeatures,
                                int nclasses,
                                int ntasks,
                                float* h_C,
                                float* h_b,
                                float tau,
                                int kernelcode,
                                float beta,
                                float a,
                                float b,
                                float d)
{

    int blockY= ntasks;
    int blockX=(int) ceil((float)(ntraining)/(float)(TPB));

    int activeTasks= ntasks;

    int numThreads = (ntraining < MAXTHREADS*2) ? nextPow2((ntraining + 1)/ 2) : MAXTHREADS;
    int numBlocks = (ntraining + (numThreads * 2 - 1)) / (numThreads * 2);
    int numBlocksRed = min(MAXBLOCKS, numBlocks);

    size_t remainingMemory;
    size_t totalMemory;

    float* d_xtraindata=0;
    cudaMalloc((void**) &d_xtraindata, sizeof(float) * ntraining* nfeatures);
    cudaMemcpy(d_xtraindata, h_xtraindata, sizeof(float) * ntraining* nfeatures,cudaMemcpyHostToDevice);

    float * h_dottraindata = (float*) malloc(sizeof(float) * ntraining);

    for (int i=0; i<ntraining; i++)
    {
        h_dottraindata[i]=0;
        for(int j=0; j<nfeatures; j++)
        {
            h_dottraindata[i]+=h_xtraindata[j*ntraining+i]*h_xtraindata[j*ntraining+i];
        }
    }

    float* d_dottraindata=0;
    cudaMalloc((void**) &d_dottraindata, sizeof(float) * ntraining);
    cudaMemcpy(d_dottraindata, h_dottraindata, sizeof(float) * ntraining,cudaMemcpyHostToDevice);
    free(h_dottraindata);

    int* d_ltraindata=0;
    cudaMalloc((void**) &d_ltraindata, sizeof(int) * ntraining);
    cudaMemcpy(d_ltraindata, h_ltraindata, sizeof(int) * ntraining,cudaMemcpyHostToDevice);

    int* d_rdata=0;
    cudaMalloc((void**) &d_rdata, sizeof(int) * nclasses * ntasks);
    cudaMemcpy(d_rdata, h_rdata, sizeof(int) * nclasses * ntasks,cudaMemcpyHostToDevice);

    int * d_ytraindata=0;
    cudaMalloc((void**) &d_ytraindata, sizeof(int) * ntraining * ntasks);

    unsigned int asize= ntraining * ntasks;
    unsigned int amem_size= sizeof(float) * asize;
    float * d_atraindata=0;
    cudaMalloc((void**) &d_atraindata, amem_size);

    unsigned int anewsize= 2 * ntasks;
    unsigned int anewmem_size= sizeof(float) * anewsize;

    float* d_aoldtraindata;
    cudaMalloc((void**) &d_aoldtraindata, anewmem_size);

    float* d_anewtraindata;
    cudaMalloc((void**) &d_anewtraindata, anewmem_size);

    float * d_fdata=0;
    cudaMalloc((void**) &d_fdata, sizeof(float) * ntraining * ntasks);

    unsigned int bsize= numBlocksRed * blockY;
    unsigned int bmem_size= sizeof(float) * bsize;

    float * h_bup = (float*) malloc(bmem_size);
    float* d_bup=0;
    cudaMalloc((void**) &d_bup, bmem_size);

    float* h_blow = (float*) malloc(bmem_size);
    float* d_blow=0;
    cudaMalloc((void**) &d_blow, bmem_size);

    unsigned int imem_size= sizeof(unsigned int) * bsize;

    int* h_Iup = (int*) malloc(imem_size);
    int* d_Iup=0;
    cudaMalloc((void**) &d_Iup, imem_size);

    int* h_Ilow = (int*) malloc(imem_size);
    int* d_Ilow=0;
    cudaMalloc((void**) &d_Ilow, imem_size);


    float* h_bup_global= (float*) malloc(sizeof(float)*blockY);
    float* d_bup_global=0;
    cudaMalloc((void**) &d_bup_global, sizeof(float)* blockY);

    float* h_blow_global= (float*) malloc(sizeof(float)*blockY);
    float* d_blow_global=0;
    cudaMalloc((void**) &d_blow_global, sizeof(float)* blockY);

    int* h_Iup_global= (int*) malloc(sizeof(int)*blockY);
    int* d_Iup_global=0;
    cudaMalloc((void**) &d_Iup_global, sizeof(int)* blockY);

    int* h_Iup_cache= (int*) malloc(sizeof(int)*blockY);
    int* d_Iup_cache=0;
    cudaMalloc((void**) &d_Iup_cache, sizeof(int)* blockY);

    bool* h_Iup_compute= (bool*) malloc(sizeof(bool)*blockY);

    int* h_Ilow_global= (int*) malloc(sizeof(int)* blockY);
    int* d_Ilow_global=0;
    cudaMalloc((void**) &d_Ilow_global, sizeof(int)* blockY);

    int* h_Ilow_cache= (int*) malloc(sizeof(int)* blockY);
    int* d_Ilow_cache=0;
    cudaMalloc((void**) &d_Ilow_cache, sizeof(int)* blockY);

    bool* h_Ilow_compute= (bool*) malloc(sizeof(bool)*blockY);

    float * d_b=0;
    cudaMalloc((void**) &d_b, sizeof(float) * ntasks);
    cudaMemcpy(d_b, h_b, sizeof(float) * ntasks,cudaMemcpyHostToDevice);

    float * d_C=0;
    cudaMalloc((void**) &d_C, sizeof(float) * ntasks);
    cudaMemcpy(d_C, h_C, sizeof(float) * ntasks,cudaMemcpyHostToDevice);

    float* d_kernelrow=0;
    cudaMalloc((void**) &d_kernelrow, sizeof(float) * nfeatures);

    float* d_kerneldot=0;
    cudaMalloc((void**) &d_kerneldot, sizeof(float) * ntraining);

    int* h_done= (int*) malloc(sizeof(int)*blockY);
    int* d_done;
    cudaMalloc((void**) &d_done, sizeof(int)* blockY);

    for (int i=0; i< blockY; i++)
    {
        h_done[i]=0;
    }
    cudaMemcpy(d_done, h_done, sizeof(int)* blockY,cudaMemcpyHostToDevice);

    int* h_active= (int*) malloc(sizeof(int)*blockY);
    int* d_active;
    cudaMalloc((void**) &d_active, sizeof(int)* blockY);

    for (int i=0; i< blockY; i++)
    {
        h_active[i]=i;
    }
    cudaMemcpy(d_active, h_active, sizeof(int)* blockY,cudaMemcpyHostToDevice);

    void* temp;
    size_t rowPitch;
    cudaMallocPitch(&temp, &rowPitch, ntraining*sizeof(float), 2);
    cudaFree(temp);

    cuMemGetInfo(&remainingMemory, &totalMemory);

    printf("%u bytes of memory found on device, %u bytes currently free\n", totalMemory, remainingMemory);

    size_t sizeOfCache = remainingMemory/((int)rowPitch);

    sizeOfCache = (size_t)((float)sizeOfCache*0.90);
    if (ntraining < sizeOfCache)
    {
        sizeOfCache = ntraining;
    }

    #ifdef __DEVICE_EMULATION__
    sizeOfCache = ntraining;
    #endif


    printf("%u rows of kernel matrix will be cached (%u bytes per row)\n", sizeOfCache, (int)rowPitch);

    float* d_kdata;
    size_t cachePitch;
    cudaMallocPitch((void**)&d_kdata, &cachePitch, ntraining*sizeof(float), sizeOfCache);

    Cache kernelCache(ntraining, sizeOfCache);

    cudaError_t err = cudaGetLastError();
    if(err)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    dim3 dimBlockInitialize(numThreads, 1, 1);
    dim3 dimGridInitialize(numBlocksRed, activeTasks, 1);

    int smemSize = 0;
    bool isNtrainingPow2=isPow2(ntraining);

    if(isNtrainingPow2)
    {
        switch (numThreads)
        {
            case 512:
                initializetraining <512,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 256:
                initializetraining <256,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 128:
                initializetraining <128,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 64:
                initializetraining <64,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 32:
                initializetraining <32,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 16:
                initializetraining <16,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  8:
                initializetraining <8,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks,d_active); break;
            case  4:
                initializetraining <4,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  2:
                initializetraining <2,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  1:
                initializetraining <1,true><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks,d_active); break;
        }
    }
    else
    {
        switch (numThreads)
        {
            case 512:
                initializetraining <512,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 256:
                initializetraining <256,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 128:
                initializetraining <128,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks,d_active); break;
            case 64:
                initializetraining <64,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 32:
                initializetraining <32,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case 16:
                initializetraining <16,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  8:
                initializetraining <8,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  4:
                initializetraining <4,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  2:
                initializetraining <2,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
            case  1:
                initializetraining <1,false><<< dimGridInitialize, dimBlockInitialize, smemSize >>>(    d_ltraindata,d_rdata,d_ytraindata,d_atraindata,    d_fdata,ntraining,ntasks, d_active); break;
        }
    }
    cudaThreadSynchronize();

    cudaError_t errinit = cudaGetLastError();
    if(errinit)
    {
        printf("Error initialization: %s\n", cudaGetErrorString(errinit));
    }

    reductionstep(          d_ytraindata,
                            d_atraindata,
                            d_fdata,
                            h_bup,
                            h_blow,
                            h_Iup,
                            h_Ilow,
                            d_bup,
                            d_blow,
                            d_Iup,
                            d_Ilow,
                            h_bup_global,
                            h_blow_global,
                            h_Iup_global,
                            h_Ilow_global,
                            d_bup_global,
                            d_blow_global,
                            d_Iup_global,
                            d_Ilow_global,
                            h_done,
                            d_done,
                            d_active,
                            numThreads,
                            numBlocksRed,
                            ntraining,
                            ntasks,
                            activeTasks,
                            d_C);


    int keepgoing=1;
    int it=0;
    int sharedhits=0;

    dim3 dimBlockReduction(numThreads, 1, 1);
    dim3 dimGridReduction(numBlocksRed, blockY, 1);

    while (keepgoing==1)
    {
        it++;

        //if(it%1000==0)
        //{
        //    printf("ITERATION %i\n",it);
        //    kernelCache.printStatistics();
        //    printf ("SharedHits: %i\n", sharedhits);

        //}

        //Check Cache first

        int blockXKernelRow= (int) ceil((float)(nfeatures)/(float)(TPB));

        dim3 dimBlockKernelRow(TPB);
        dim3 dimGridKernelRow(blockXKernelRow);

        dim3 dimBlockKernelDot(TPB);
        dim3 dimGridKernelDot(blockX);


        //printf("It: %05d i: %05d j: %05d Gap: %e\n", it, h_Iup_global[0], h_Ilow_global[0], h_blow_global[0] - h_bup_global[0]); //LLLLLLLLLLLLLLLLLL

        for (int i=0; i< ntasks; i++)
        {
            if(h_done[i]==0)
            {
                //printf("taks %i\n", i);
                bool upseen=false;
                bool lowseen=false;

                for(int j=0; j<i; j++)
                {
                    if(h_Iup_global[i]==h_Iup_global[j])
                    {
                        h_Iup_cache[i]=h_Iup_cache[j];
                        h_Iup_compute[i]=h_Iup_compute[j];
                        sharedhits++;
                        upseen=true;
                    }
                    else if(h_Iup_global[i]==h_Ilow_global[j])
                    {
                        h_Iup_cache[i]=h_Ilow_cache[j];
                        h_Iup_compute[i]=h_Ilow_compute[j];
                        sharedhits++;
                        upseen=true;
                    }

                    if(h_Ilow_global[i]==h_Iup_global[j])
                    {
                        h_Ilow_cache[i]=h_Iup_cache[j];
                        h_Ilow_compute[i]=h_Iup_compute[j];
                        sharedhits++;
                        lowseen=true;
                    }
                    else if(h_Ilow_global[i]==h_Ilow_global[j])
                    {
                        h_Ilow_cache[i]=h_Ilow_cache[j];
                        h_Ilow_compute[i]=h_Ilow_compute[j];
                        sharedhits++;
                        lowseen=true;
                    }
                }

                if(!upseen)
                {

                    kernelCache.findData(h_Iup_global[i], h_Iup_cache[i], h_Iup_compute[i]);

                    if(h_Iup_compute[i])
                    {

                        kerneleval (     d_xtraindata,
                                        d_dottraindata,
                                        d_kernelrow,
                                        d_kerneldot,
                                        d_kdata,
                                        h_Iup_global[i],
                                        h_Iup_cache[i],
                                        ntraining,
                                        nfeatures,
                                        beta,
                                        a,
                                        b,
                                        d,
                                        kernelcode);

                    }
                }

                if(!lowseen)
                {

                    kernelCache.findData(h_Ilow_global[i], h_Ilow_cache[i], h_Ilow_compute[i]);

                    if(h_Ilow_compute[i])
                    {
                        kerneleval (     d_xtraindata,
                                        d_dottraindata,
                                        d_kernelrow,
                                        d_kerneldot,
                                        d_kdata,
                                        h_Ilow_global[i],
                                        h_Ilow_cache[i],
                                        ntraining,
                                        nfeatures,
                                        beta,
                                        a,
                                        b,
                                        d,
                                        kernelcode);
                    }
                }

            }
        }


        cudaMemcpy(d_Iup_cache, h_Iup_cache, sizeof(int)* blockY ,cudaMemcpyHostToDevice);
        cudaMemcpy(d_Ilow_cache, h_Ilow_cache, sizeof(int)* blockY ,cudaMemcpyHostToDevice);


        int blockYAlpha=(int) ceil((float)(ntasks)/(float)(TPB));

        dim3 dimBlockAlpha(TPB);
        dim3 dimGridAlpha(blockYAlpha);


        calculatealphas <<<dimGridAlpha, dimBlockAlpha>>>(    d_xtraindata,
                                                            d_kdata,
                                                            d_ytraindata,
                                                            d_atraindata,
                                                            d_anewtraindata,
                                                            d_aoldtraindata,
                                                            d_fdata,
                                                            d_Iup_global,
                                                            d_Ilow_global,
                                                            d_Iup_cache,
                                                            d_Ilow_cache,
                                                            d_done,
                                                            ntraining,
                                                            nfeatures,
                                                            ntasks,
                                                            d_C);
        cudaThreadSynchronize();

        cudaMemcpy(h_done, d_done, sizeof(int)* blockY,cudaMemcpyDeviceToHost);

        dim3 dimBlockActiveTaskReduction(numThreads, 1, 1);
        dim3 dimGridActiveTaskReduction(numBlocksRed, activeTasks, 1);
        cudaMemcpy(d_active, h_active, sizeof(int)* activeTasks,cudaMemcpyHostToDevice);


        if(isNtrainingPow2)
        {
            switch (numThreads)
            {
                case 512:
                    updateparams <512,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 256:
                    updateparams <256,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 128:
                    updateparams <128,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 64:
                    updateparams <64,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 32:
                    updateparams <32,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 16:
                    updateparams <16,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  8:
                    updateparams <8,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  4:
                    updateparams <4,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  2:
                    updateparams <2,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  1:
                    updateparams <1,true><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
            }
        }
        else
        {
            switch (numThreads)
            {
                case 512:
                    updateparams <512,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 256:
                    updateparams <256,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 128:
                    updateparams <128,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 64:
                    updateparams <64,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 32:
                    updateparams <32,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case 16:
                    updateparams <16,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  8:
                    updateparams <8,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  4:
                    updateparams <4,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  2:
                    updateparams <2,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
                case  1:
                    updateparams <1,false><<< dimGridActiveTaskReduction, dimBlockActiveTaskReduction, smemSize >>>(d_xtraindata,d_kdata,d_ytraindata,d_atraindata,d_anewtraindata,d_aoldtraindata,d_fdata,d_Iup_global,d_Ilow_global,    d_Iup_cache,d_Ilow_cache,d_done,d_active,ntraining,    nfeatures,ntasks,activeTasks,d_C); break;
            }
        }


        cudaThreadSynchronize();

        reductionstep(          d_ytraindata,
                                d_atraindata,
                                d_fdata,
                                h_bup,
                                h_blow,
                                h_Iup,
                                h_Ilow,
                                d_bup,
                                d_blow,
                                d_Iup,
                                d_Ilow,
                                h_bup_global,
                                h_blow_global,
                                h_Iup_global,
                                h_Ilow_global,
                                d_bup_global,
                                d_blow_global,
                                d_Iup_global,
                                d_Ilow_global,
                                h_done,
                                d_done,
                                d_active,
                                numThreads,
                                numBlocksRed,
                                ntraining,
                                ntasks,
                                activeTasks,
                                d_C);


        int acc=0;

        for (int i=0; i< blockY; i++)
        {

            if(h_blow_global[i] <= h_bup_global[i]+ 2 *tau)
            {
                h_done[i]=1;
            }

            if(h_done[i]==0)
            {
                h_active[acc]=i;
                acc++;
            }

        }
        activeTasks=acc;
        cudaMemcpy(d_done, h_done, sizeof(int)* blockY,cudaMemcpyHostToDevice);

        keepgoing=0;
        for (int i=0; i< blockY; i++)
        {
            if(h_done[i]==0)
            {
                keepgoing=1;;
            }

        }

    }

    printf("All tasks converged! Iterations %i\n",it);

    cudaMemcpy(h_atraindata, d_atraindata, amem_size ,cudaMemcpyDeviceToHost);


    for (int j=0; j<ntasks; j++)
    {
        h_b[j]= - (h_bup_global[j] + h_blow_global [j])/2;

        int svnum=0;
        for (int i=0; i<ntraining; i++)
        {
            if(h_atraindata[j*ntraining + i]!=0)
            {

                svnum++;
            }
        }

    }


    cudaFree(d_xtraindata);
    cudaFree(d_dottraindata);
    cudaFree(d_kdata);
    cudaFree(d_ltraindata);
    cudaFree(d_rdata);
    cudaFree(d_ytraindata);

    cudaFree(d_atraindata);
    cudaFree(d_anewtraindata);
    cudaFree(d_aoldtraindata);
    cudaFree(d_fdata);

    cudaFree(d_kerneldot);
    cudaFree(d_kernelrow);

    cudaFree(d_bup);
    cudaFree(d_blow);
    cudaFree(d_Iup);
    cudaFree(d_Ilow);
    cudaFree(d_bup_global);
    cudaFree(d_blow_global);
    cudaFree(d_Iup_global);
    cudaFree(d_Ilow_global);
    cudaFree(d_Iup_cache);
    cudaFree(d_Ilow_cache);
    cudaFree(d_b);
    cudaFree(d_done);
    cudaFree(d_active);
    cudaFree(d_C);

    free(h_bup);
    free(h_blow);
    free(h_Iup);
    free(h_Ilow);
    free(h_bup_global);
    free(h_blow_global);
    free(h_Iup_global);
    free(h_Ilow_global);
    free(h_Iup_cache);
    free(h_Ilow_cache);
    free(h_Iup_compute);
    free(h_Ilow_compute);
    free(h_done);
    free(h_active);

}
