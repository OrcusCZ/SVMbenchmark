#pragma once
#define maxvpm         30000  /* max number of method iterations allowed  */
//#define maxvpm          10
#define maxprojections 200 //TODO: rename this
#define alpha_max      1e10
#define alpha_min      1e-10
#define EPS_SV    1.0e-9   /* precision for multipliers */
//TODO: move alphas to shared memory
#define DEBUG_PRINTF 0

typedef double TFloatX;
//typedef float TFloatX;

__device__ float atomicMaxFloat(float * addr, float val)
{
    unsigned int * address_as_ui = (unsigned int *)addr;
    unsigned int old = *address_as_ui,
                 assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed, __float_as_int(max(__int_as_float(assumed), val)));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ TFloatX ProjectR(TFloatX & x, TFloatX lambda, TFloatX a, TFloatX b, TFloatX c, TFloatX l, TFloatX u, float * shSum)
{
    __shared__ TFloatX shSumD[256];
    x = max(l, min(u, -c + (TFloatX)lambda * a));
    shSumD[threadIdx.x] = a * x;
    __syncthreads();
    blockReduceSum(shSumD);
    TFloatX r = shSumD[0];
    __syncthreads();

#if DEBUG_PRINTF >= 3
    if (threadIdx.x == 0)
        printf("ProjectR returns %.12f, lambda: %.12f, x: %f, a: %f, b: %f, c: %f\n", r - b, lambda, x, a, b, c);
#endif
    return r - b;
}

template<unsigned int blockSize>
__device__ int ProjectDai(TFloatX a, TFloatX b, TFloatX c, TFloatX l, TFloatX u, TFloatX & x, TFloatX & lam_ext, float * shSum)
{
    TFloatX r, rl, ru, s;
    //float tol_lam = 1.0e-11,
          //tol_r   = 1.0e-10 * sqrt((u-l)*(float)n),
    TFloatX tol_lam = 1.0e-11,
          tol_r   = 1.0e-10 * sqrt((u-l)*(float)blockSize),
          lambda  = lam_ext,
          lambdal,
          lambdau,
          lambdar,
          dlambda = 0.5,
          lambda_new;
    int iter = 1;
    b = -b;

    // Bracketing Phase
#if DEBUG_PRINTF >= 3
    if (threadIdx.x == 0)
        printf("Bracketing phase\n");
#endif
    r = ProjectR(x, lambda, a, b, c, l, u, shSum);
#if DEBUG_PRINTF >= 3
    if (threadIdx.x == 0)
        printf("r: %f, n: %d, lambda: %.10f, b: %f, l: %f, u: %f\n", r, blockSize, lambda, b, l, u);
#endif
    if (fabs(r) < tol_r)
    {
#if DEBUG_PRINTF >= 2
        if (threadIdx.x == 0)
            printf("Projector exits after 0 iterations, r: %.10f, tol_r: %.10f, lambda: %f\n", r, tol_r, lambda);
#endif
        return 0;
    }

    if (r < 0.0)
    {
        lambdal = lambda;
        rl      = r;
        lambda  = lambda + dlambda;
        r       = ProjectR(x, lambda, a, b, c, l, u, shSum);
        while (r < 0.0)
        {
           lambdal = lambda;
           s       = rl/r - 1.0;
           if (s < 0.1) s = 0.1;
           dlambda = dlambda + dlambda/s;
           lambda  = lambda + dlambda;
           rl      = r;
           r       = ProjectR(x, lambda, a, b, c, l, u, shSum);
        }
        lambdau = lambda;
        ru      = r;
    }
    else
    {
        lambdau = lambda;
        ru      = r;
        lambda  = lambda - dlambda;
        r       = ProjectR(x, lambda, a, b, c, l, u, shSum);
        while (r > 0.0)
        {
           lambdau = lambda;
           s       = ru/r - 1.0;
           if (s < 0.1) s = 0.1;
           dlambda = dlambda + dlambda/s;
           lambda  = lambda - dlambda;
           ru      = r;
           r       = ProjectR(x, lambda, a, b, c, l, u, shSum);
        }
      lambdal = lambda;
      rl      = r;
    }


    // Secant Phase
#if DEBUG_PRINTF >= 3
    if (threadIdx.x == 0)
        printf("Secant phase\n");
#endif
    s       = 1.0 - rl/ru;
    dlambda = dlambda/s;
    lambda  = lambdau - dlambda;
    r       = ProjectR(x, lambda, a, b, c, l, u, shSum);

#if DEBUG_PRINTF >= 3
    if (threadIdx.x == 0)
        printf("r: %f, n: %d, lambda: %.10f, b: %f, l: %f, u: %f\n", r, blockSize, lambda, b, l, u);
#endif
    while (   fabs(r) > tol_r 
           && dlambda > tol_lam * (1.0 + fabs(lambda)) 
           && iter    < maxprojections                )
    {
       iter++;
       if (r > 0.0)
       {
           if (s <= 2.0)
           {
               lambdau = lambda;
               ru      = r;
               s       = 1.0 - rl/ru;
               dlambda = (lambdau - lambdal) / s;
               lambda  = lambdau - dlambda;
           }
           else
           {
               s          = ru/r-1.0;
               if (s < 0.1) s = 0.1;
               dlambda    = (lambdau - lambda) / s;
               lambda_new = 0.75*lambdal + 0.25*lambda;
               if (lambda_new < (lambda - dlambda))
                   lambda_new = lambda - dlambda;
               lambdau    = lambda;
               ru         = r;
               lambda     = lambda_new;
               s          = (lambdau - lambdal) / (lambdau - lambda);
           }
       }
       else
       {
           if (s >= 2.0)
           {
               lambdal = lambda;
               rl      = r;
               s       = 1.0 - rl/ru;
               dlambda = (lambdau - lambdal) / s;
               lambda  = lambdau - dlambda;
           }
           else
           {
               s          = rl/r - 1.0;
               if (s < 0.1) s = 0.1;
               dlambda    = (lambda-lambdal) / s;
               lambda_new = 0.75*lambdau + 0.25*lambda;
               if (lambda_new > (lambda + dlambda))
                   lambda_new = lambda + dlambda;
               lambdal    = lambda;
               rl         = r;
               lambda     = lambda_new;
               s          = (lambdau - lambdal) / (lambdau-lambda);
           }
       }
       r = ProjectR(x, lambda, a, b, c, l, u, shSum);
    }

    if (threadIdx.x == 0)
    {
        lam_ext = lambda;
        //__threadfence();
        if (threadIdx.x == 0 && iter >= maxprojections)
            printf("  error: Projector exits after max iterations: %d\n", iter);
#if DEBUG_PRINTF >= 2
        if (threadIdx.x == 0)
            printf("Projector exits after %d iterations\n", iter);
#endif
    }
    __syncthreads();

    return iter;
}

//TODO: this is not using workingset at all, fix it
template<unsigned int blockSize>
__global__ void
__launch_bounds__(256, 1)
kernelFletcherAlg2A(float *vecA, float *b, float c, float *iy, float *x, float *xdiff, float tol, const int * ws)
        //float * g, float * y, float * tempv, float * d, float * Ad, float * t, float * xplus, float * tplus, float * sk, float * yk)
{
    __shared__ float Ad[blockSize];
    __shared__ float t[blockSize];
    __shared__ float shSum[blockSize];
    __shared__ int wsi[blockSize];
    __shared__ float xold[blockSize];
    __shared__ float b_[blockSize];
    __shared__ float iy_[blockSize];
    __shared__ float g[blockSize];
    //__shared__ TFloatX y[blockSize];
    __shared__ float max, gd, ak, bk, akold, bkold, alpha;
    __shared__ TFloatX lam_ext;
    __shared__ int it1, it2;
    __shared__ float DELTAsv, ProdDELTAsv;

    const int L = 2;
    __shared__ int llast;
    __shared__ float fv0, fv, fr, fbest, fc;

    //int lscount = 0, projcount = 0;
    const float eps = 1.0e-16;

    //int    *ipt, *ipt2, *uv;
    //float *g, *y, *tempv, *d, *Ad, *t, *xplus, *tplus, *sk, *yk;

    /*** arrays allocation ***/
    //ipt   = (int    *)malloc(n * sizeof(int   ));
    //ipt2  = (int    *)malloc(n * sizeof(int   ));
    //uv    = (int    *)malloc(n * sizeof(int   ));
    //g     = (float *)malloc(n * sizeof(float));
    //y     = (float *)malloc(n * sizeof(float));
    //tempv = (float *)malloc(n * sizeof(float));
    //d     = (float *)malloc(n * sizeof(float));
    //Ad    = (float *)malloc(n * sizeof(float));
    //t     = (float *)malloc(n * sizeof(float));
    //xplus = (float *)malloc(n * sizeof(float));
    //tplus = (float *)malloc(n * sizeof(float));
    //sk    = (float *)malloc(n * sizeof(float));
    //yk    = (float *)malloc(n * sizeof(float));

    if (threadIdx.x == 0)
    {
        DELTAsv = EPS_SV * c;
        ProdDELTAsv = (tol <= 1.0e-5 || blockSize <= 20) ? 0.0f : (EPS_SV * c);
    }

    //for (i = 0; i < n; i++)
        //tempv[i] = -x[i];
    //for (int i = threadIdx.x; i < n; i += blockDim.x)
        //tempv[i] = -x[i];
    //int wsi = ws[threadIdx.x];
    wsi[threadIdx.x] = ws[threadIdx.x];
    //float b_ = b[wsi];
    //float b_ = b[threadIdx.x];
    //float iy_ = iy[wsi[threadIdx.x]];
    b_[threadIdx.x] = b[threadIdx.x];
    iy_[threadIdx.x] = iy[wsi[threadIdx.x]];
    TFloatX x_ = x[wsi[threadIdx.x]];
    //TFloatX x_ = 0;
    //x_ = x[wsi];
    TFloatX tempv = -x_;
    //float xold = x_;
    //TFloatX xold = x[wsi[threadIdx.x]];
    xold[threadIdx.x] = x[wsi[threadIdx.x]];

    lam_ext = 0;
    __syncthreads();
    
    //projcount += ProjectDai(n, iy, d_df_e, tempv, 0, c, x, &d_df_lam_ext, shSum);
    /*projcount +=*/ ProjectDai<blockSize>(iy_[threadIdx.x], d_df_e, tempv, 0, c, x_, lam_ext, shSum);

    // g = A*x + b;
    // SparseProd(n, t, A, x, ipt);
    //{
      //int   it;
      //float *tempA;

      //it = 0;
      //for (i = 0; i < n; i++)
          //if (fabs(x[i]) > ProdDELTAsv)
              //ipt[it++] = i;

      //memset(t, 0, n*sizeof(float));
      //for (i = 0; i < it; i++)
      //{
           //tempA = vecA + ipt[i] * n;
           //for (j = 0; j < n; j++)
               //t[j] += (tempA[j]*x[ipt[i]]);
      //}
    //}
    for (int i = 0; i < blockSize; i++)
    {
        //for (int j = threadIdx.x; j < n; j += blockDim.x)
            //shSum[threadIdx.x] += vecA[n * i + j] * x[j];
        shSum[threadIdx.x] = vecA[blockSize * i + threadIdx.x] * x_;
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            t[i] = shSum[0];
        __syncthreads();
    }

    //for (i = 0; i < n; i++)
    //{
      //g[i] = t[i] + b[i],
      //y[i] = g[i] - x[i];
    //}
    //for (int i = threadIdx.x; i < n; i += blockDim.x)
    //{
        //g[i] = t[i] + b[i];
        //y[i] = g[i] - x[i];
    //}
    g[threadIdx.x] = t[threadIdx.x] + b_[threadIdx.x];
    TFloatX y = g[threadIdx.x] - x_;
#if DEBUG_PRINTF >= 3
    if (threadIdx.x < 10)
        printf("%d: t: %f, b: %f, g: %f, y: %f, x: %f\n", threadIdx.x, t[threadIdx.x], b_[threadIdx.x], g[threadIdx.x], y, x_);
#endif

    //projcount += ProjectDai(n, iy, d_df_e, y, 0, c, tempv, &d_df_lam_ext, shSum);
    /*projcount +=*/ ProjectDai<blockSize>(iy_[threadIdx.x], d_df_e, y, 0, c, tempv, lam_ext, shSum);

    if (threadIdx.x == 0)
        max = alpha_min;
    __syncthreads();
    //for (i = 0; i < n; i++)
    //{
        //y[i] = tempv[i] - x[i];
        //if (fabs(y[i]) > max)
            //max = fabs(y[i]);
    //}
    //for (int i = threadIdx.x; i < n; i += blockDim.x)
    //{
        //y[i] = tempv[i] - x[i];
        //atomicMaxFloat(&max, fabs(y[i]));
    //}
    y = tempv - x_;
    atomicMaxFloat(&max, fabs(y));
    __syncthreads();

    if (max < c*tol*1e-3)
    {
        //lscount = 0;
        //TODO set x
        x[wsi[threadIdx.x]] = x_;
        xdiff[threadIdx.x] = (x_ - xold[threadIdx.x]) * iy_[threadIdx.x];
#if DEBUG_PRINTF >= 1
        if (threadIdx.x == 0)
            printf("Dai-Fletcher method exits after 0 iterations, max: %f, c: %f, tol: %f\n", max, c, tol);
#endif
        return;
    }

    if (threadIdx.x == 0)
        alpha = 1.0 / max;

    //fv0   = 0.0;
    //for (i = 0; i < n; i++)
        //fv0 += x[i] * (0.5*t[i] + b[i]);
    //shSum[threadIdx.x] = 0;
    //for (int i = threadIdx.x; i < n; i += blockDim.x)
        //shSum[threadIdx.x] += x[i] * (0.5*t[i] + b[i]);
    shSum[threadIdx.x] = x_ * (0.5 * t[threadIdx.x] + b_[threadIdx.x]);
    __syncthreads();
    blockReduceSum(shSum);
    if (threadIdx.x == 0)
    {
        fv0 = shSum[0];
        fr = alpha_max;
        fbest = fv0;
        fc    = fv0;
        akold = 0;
        bkold = 0;
    }
    __syncthreads();

    /*** adaptive nonmonotone linesearch ***/

    int iter = 1;
    for (; iter <= maxvpm; iter++)
    {
        //if (threadIdx.x == 0)
            //printf("alpha: %f, g[0]: %f, x[0]: %f\n", alpha, g[0], x[0]);
        //for (i = 0; i < n; i++)
            //tempv[i] = alpha*g[i] - x[i];
        //for (int i = threadIdx.x; i < n; i += blockDim.x)
            //tempv[i] = alpha * g[i] - x[i];
        tempv = alpha * g[threadIdx.x] - x_;

        //projcount += ProjectDai(n, iy, d_df_e, tempv, 0, c, y, &d_df_lam_ext, shSum);
        /*projcount +=*/ ProjectDai<blockSize>(iy_[threadIdx.x], d_df_e, tempv, 0, c, y, lam_ext, shSum);

        //if (threadIdx.x == 0)
            //for (int i = 0; i < 10; i++)
                //printf("y[%d]: %f\n", i, y[i]);

        //gd = 0.0;
        //for (i = 0; i < n; i++)
        //{
            //d[i] = y[i] - x[i];
            //gd  += d[i] * g[i];
        //}
        //shSum[threadIdx.x] = 0;
        //for (int i = threadIdx.x; i < n; i += blockDim.x)
        //{
            //d[i] = y[i] - x[i];
            //shSum[threadIdx.x] += d[i] * g[i];
        //}
        float d = y - x_;
        shSum[threadIdx.x] = d * g[threadIdx.x];
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            gd = shSum[0];
        __syncthreads();

        /* compute Ad = A*d  or  Ad = A*y - t depending on their sparsity */
        //{
           //int   i, it, it2;
           //float *tempA;

           //it = it2 = 0;
           //for (i = 0; i < n; i++)
               //if (fabs(d[i]) > (ProdDELTAsv*1.0e-2))
                   //ipt[it++]   = i;
           //for (i = 0; i < n; i++)
               //if (fabs(y[i]) > ProdDELTAsv)
                   //ipt2[it2++] = i;

           //memset(Ad, 0, n*sizeof(float));
           //if (it < it2) // compute Ad = A*d
           //{
              //for (i = 0; i < it; i++)
              //{
                  //tempA = vecA + ipt[i]*n;
                  //for (j = 0; j < n; j++)
                      //Ad[j] += (tempA[j] * d[ipt[i]]);
              //}
           //}
           //else          // compute Ad = A*y-t
           //{
              //for (i = 0; i < it2; i++)
              //{
                  //tempA = vecA + ipt2[i]*n;
                  //for (j = 0; j < n; j++)
                      //Ad[j] += (tempA[j] * y[ipt2[i]]);
              //}
              //for (j = 0; j < n; j++)
                  //Ad[j] -= t[j];
           //}
        //}
        {
            if (threadIdx.x == 0)
                it1 = it2 = 0;
            __syncthreads();
            //for (int i = threadIdx.x; i < n; i += blockDim.x)
            //{
                //if (fabs(d[i]) > ProdDELTAsv * 1.0e-2f)
                    //atomicAdd(&it1, 1);
                //if (fabs(y[i]) > ProdDELTAsv)
                    //atomicAdd(&it2, 1);
            //}
            if (fabs(d) > ProdDELTAsv * 1.0e-2f)
                atomicAdd(&it1, 1);
            if (fabs(y) > ProdDELTAsv)
                atomicAdd(&it2, 1);
            __syncthreads();
            //if (threadIdx.x == 0)
                //printf("it1: %d, it2: %d\n", it1, it2);
            if (it1 < it2)
            {
                for (int i = 0; i < blockSize; i++)
                {
                    //if (fabs(d[i]) <= ProdDELTAsv * 1.0e-2f)
                    //{
                        //if (threadIdx.x == 0)
                            //Ad[i] = 0;
                        //continue;
                    //}
                    //shSum[threadIdx.x] = 0;
                    //for (int j = threadIdx.x; j < n; j += blockDim.x)
                        //shSum[threadIdx.x] += vecA[n * i + j] * d[j];
                    shSum[threadIdx.x] = vecA[blockSize * i + threadIdx.x] * d;
                    __syncthreads();
                    blockReduceSum(shSum);
                    if (threadIdx.x == 0)
                        Ad[i] = shSum[0];
                    __syncthreads();
                }
            }
            else
            {
                for (int i = 0; i < blockSize; i++)
                {
                    //if (threadIdx.x == 0)
                        //printf("i: %d, t: %f, y: %f, d: %f\n", i, t[i], y[i], d[i]);
                    //if (i == 1 && threadIdx.x == 0)
                    //{
                        //for (int j = 0; j < n; j++)
                            //printf("j: %d, A: %f, y: %f\n", j, vecA[n * i + j], y[j]);
                    //}
                    //if (i == 1)
                    //{
                        //for (int j = threadIdx.x; j < n; j += blockDim.x)
                            //printf("j: %d, A: %f, y: %f\n", j, vecA[n * i + j], y[j]);
                    //}
                    //TODO tuta podminka nefunguje kdyz je odkomentovana. proc?
                    //if (fabs(y[i]) <= ProdDELTAsv)
                    //{
                        //if (threadIdx.x == 0)
                            //Ad[i] = -t[i];
                        //continue;
                    //}
                    //shSum[threadIdx.x] = 0;
                    //for (int j = threadIdx.x; j < n; j += blockDim.x)
                        //shSum[threadIdx.x] += vecA[n * i + j] * y[j];
                    shSum[threadIdx.x] = vecA[blockSize * i + threadIdx.x] * y;
                    __syncthreads();
                    blockReduceSum(shSum);
                    if (threadIdx.x == 0)
                        Ad[i] = shSum[0] - t[i];
                    __syncthreads();
                }
            }
            //if (threadIdx.x == 0)
                //for (int i = 0; i < 10; i++)
                    //printf("Ad[%d]: %f, t: %f\n", i, Ad[i], t[i]);
        }

        //ak = 0.0;
        //for (i = 0; i < n; i++)
            //ak += d[i] * d[i];

        //bk = 0.0;
        //for (i = 0; i < n; i++)
            //bk += d[i]*Ad[i];
        shSum[threadIdx.x] = d * d;
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            ak = shSum[0];
        __syncthreads();

        shSum[threadIdx.x] = d * Ad[threadIdx.x];
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            bk = shSum[0];
        __syncthreads();

        float lamnew;
        if (bk > eps*ak && gd < 0.0)    // ak is normd
            lamnew = -gd/bk;
        else
            lamnew = 1.0;

        //fv = 0.0;
        //for (i = 0; i < n; i++)
        //{
            //xplus[i] = x[i] + d[i];
            //tplus[i] = t[i] + Ad[i];
            //fv      += xplus[i] * (0.5*tplus[i] + b[i]);
        //}
        float xplus = x_ + d;
        float tplus = t[threadIdx.x] + Ad[threadIdx.x];
        shSum[threadIdx.x] = xplus * (0.5*tplus + b_[threadIdx.x]);
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            fv = shSum[0];
        __syncthreads();

        //if (threadIdx.x == 0)
            //printf("fv: %f, lamnew: %f, ak: %f, bk: %f, gd: %f\n", fv, lamnew, ak, bk, gd);

        if ((iter == 1 && fv >= fv0) || (iter > 1 && fv >= fr))
        {
            //lscount++;
            //fv = 0.0;
            //for (i = 0; i < n; i++)
            //{
                //xplus[i] = x[i] + lamnew*d[i];
                //tplus[i] = t[i] + lamnew*Ad[i];
                //fv      += xplus[i] * (0.5*tplus[i] + b[i]);
            //}
            xplus = x_ + lamnew * d;
            tplus = t[threadIdx.x] + lamnew * Ad[threadIdx.x];
            shSum[threadIdx.x] = xplus * (0.5*tplus + b_[threadIdx.x]);
            __syncthreads();
            blockReduceSum(shSum);
            if (threadIdx.x == 0)
                fv = shSum[0];
            __syncthreads();
        }

        //for (i = 0; i < n; i++)
        float sk = xplus - x_;
        float yk = tplus - t[threadIdx.x];
        x_  = xplus;
        t[threadIdx.x] = tplus;
        g[threadIdx.x]  = t[threadIdx.x] + b_[threadIdx.x];

        // update the line search control parameters

        if (threadIdx.x == 0)
        {
            if (fv < fbest)
            {
                fbest = fv;
                fc    = fv;
                llast = 0;
            }
            else
            {
                //fc = (fc > fv ? fc : fv);
                //fc = max(fc, fv);
                if (fv > fc)
                    fc = fv;
                llast++;
                if (llast == L)
                {
                    fr    = fc;
                    fc    = fv;
                    llast = 0;
                }
            }
        }

        //ak = bk = 0.0;
        //for (i = 0; i < n; i++)
        //{
            //ak += sk[i] * sk[i];
            //bk += sk[i] * yk[i];
        //}
        shSum[threadIdx.x] = sk * sk;
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            ak = shSum[0];
        __syncthreads();

        shSum[threadIdx.x] = sk * yk;
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            bk = shSum[0];
        __syncthreads();

        //if (threadIdx.x == 0)
            //printf("ak: %f, bk: %f, akold: %f, bkold: %f\n", ak, bk, akold, bkold);
        if (threadIdx.x == 0)
        {
            if (bk < eps*ak)
                alpha = alpha_max;
            else
            {
                if (bkold < eps*akold)
                    alpha = ak/bk;
                else
                    alpha = (akold+ak)/(bkold+bk);

                if (alpha > alpha_max)
                    alpha = alpha_max;
                else if (alpha < alpha_min)
                    alpha = alpha_min;
            }

            akold = ak;
            bkold = bk;
        }

        /*** stopping criterion based on KKT conditions ***/

        //bk = 0.0;
        //for (i = 0; i < n; i++)
            //bk +=  x[i] * x[i];
        shSum[threadIdx.x] = x_ * x_;
        __syncthreads();
        blockReduceSum(shSum);
        if (threadIdx.x == 0)
            bk = shSum[0];
        __syncthreads();

        //if (sqrt(ak) < tol*10 * sqrt(bk))
        //{
            //it     = 0;
            //luv    = 0;
            //kktlam = 0.0;
            //for (i = 0; i < n; i++)
            //{
                //if ((x[i] > DELTAsv) && (x[i] < c-DELTAsv))
                //{
                    //ipt[it++] = i;
                    //kktlam    = kktlam - iy[i]*g[i];
                //}
                //else
                    //uv[luv++] = i;
            //}

            //if (it == 0)
            //{
                //if (sqrt(ak) < tol*0.5 * sqrt(bk))
                    //goto Clean;
            //}
            //else
            //{

                //kktlam = kktlam/it;
                //info   = 1;
                //for (i = 0; i < it; i++)
                    //if ( fabs(iy[ipt[i]] * g[ipt[i]] + kktlam) > tol )
                    //{
                        //info = 0;
                        //break;
                    //}

                //if (info == 1)
                    //for (i = 0; i < luv; i++)
                        //if (x[uv[i]] <= DELTAsv)
                        //{
                            //if (g[uv[i]] + kktlam*iy[uv[i]] < -tol)
                            //{
                                //info = 0;
                                //break;
                            //}
                        //}
                        //else
                        //{
                            //if (g[uv[i]] + kktlam*iy[uv[i]] > tol)
                            //{
                                //info = 0;
                                //break;
                            //}
                        //}

                //if (info == 1)
                    //goto Clean;
            //}
        //}
#if DEBUG_PRINTF >= 1
        if (threadIdx.x == 0)
            printf("iter: %d, %f < %f, diff: %f\n", iter, sqrt(ak), tol*10*sqrt(bk), sqrt(ak) - tol*10*sqrt(bk));
#endif
        if (sqrt(ak) < tol*10 * sqrt(bk))
        {
            if (threadIdx.x == 0)
            {
                it1 = 0; //pocet free prvku
                it2 = 1; //1 kdyz je reseni optimalni
            }
            __syncthreads();
            if ((x_ > DELTAsv) && (x_ < c-DELTAsv))
            {
                shSum[threadIdx.x] = iy_[threadIdx.x] * g[threadIdx.x];
                atomicAdd(&it1, 1);
            }
            else
                shSum[threadIdx.x] = 0;
            __syncthreads();
            blockReduceSum(shSum);
            float kktlam = -shSum[0];
            __syncthreads();

            if (it1 == 0)
            {
#if DEBUG_PRINTF >= 2
                if (threadIdx.x == 0)
                    printf("no free vectors, diff2: %f\n", tol*0.5*sqrt(bk) - sqrt(ak));
#endif
                if (sqrt(ak) < tol*0.5 * sqrt(bk))
                    break;
            }
            else
            {
                kktlam = kktlam/it1;
                if ((x_ > DELTAsv) && (x_ < c-DELTAsv))
                    if (fabs(iy_[threadIdx.x] * g[threadIdx.x] + kktlam) > tol)
                    {
#if DEBUG_PRINTF >= 2
                        if (threadIdx.x == 0)
                            printf("free vector condition failed\n");
#endif
                        it2 = 0;
                        //break;
                    }
                __syncthreads();

                if (it2 == 1)
                {
                    if (x_ <= DELTAsv)
                    {
                        if (g[threadIdx.x] + kktlam * iy_[threadIdx.x] < -tol)
                        {
#if DEBUG_PRINTF >= 2
                            if (threadIdx.x == 0)
                                printf("lower bound vector condition failed\n");
#endif
                            it2 = 0;
                            //break;
                        }
                    }
                    else if (x_ >= c-DELTAsv)
                    {
                        if (g[threadIdx.x] + kktlam * iy_[threadIdx.x] > tol)
                        {
#if DEBUG_PRINTF >= 2
                            if (threadIdx.x == 0)
                                printf("upper bound vector condition failed\n");
#endif
                            it2 = 0;
                            //break;
                        }
                    }
                }
                __syncthreads();

                if (it2 == 1)
                    break;
            }
        }
    }

    x[wsi[threadIdx.x]] = x_;
    xdiff[threadIdx.x] = (x_ - xold[threadIdx.x]) * iy_[threadIdx.x];
#if DEBUG_PRINTF >= 1
    if (threadIdx.x == 0)
        printf("\nDai-Fletcher method exits after %d iterations.\n", iter);
#endif
}

template<unsigned int WS>
__global__ void kernelCalcE(const int * ws, const float * alpha, const float * y, int num_vec)
{
    __shared__ float shSum[WS];
    /*shSum[threadIdx.x] = 0;
    for (int k = threadIdx.x; k < num_vec; k += blockDim.x)
        shSum[threadIdx.x] += alpha[k] * y[k];
    if (threadIdx.x < WS)
    {
        int i = ws[threadIdx.x];
        shSum[threadIdx.x] -= alpha[i] * y[i];
    }*/
    double acc = 0;
    for (int k = threadIdx.x; k < num_vec; k += blockDim.x)
        acc += (double)alpha[k] * y[k];
    if (threadIdx.x < WS)
    {
        int i = ws[threadIdx.x];
        acc -= (double)alpha[i] * y[i];
    }
    shSum[threadIdx.x] = acc;
    __syncthreads();
    blockReduceSum(shSum);
    if (threadIdx.x == 0)
        d_df_e = shSum[0];
}
