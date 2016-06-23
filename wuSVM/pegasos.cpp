/*
Copyright (c) 2014, Washington University in St. Louis
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Washington University in St. Louis nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL WASHINGTON UNIVERSITY BE LIABLE FOR ANY 
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND 
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "pegasos.h"
#include "parsing.h"
#include "kernels.h"
#include "fileIO.h"
#include "lasp_matrix.h"
#include <cstdlib>
#include <ctime>
#include <set>


#ifdef CPP11
#include <random>
#define getrandom(max) dist(mt)
#else
#define getrandom(max) (rand() % max)
#endif

using namespace std;

namespace lasp{
	template<class T>
	int pegasos_svm_host(svm_problem& p) {
		bool gpu = p.options.usegpu;
		
		//Check that we have a CUDA device
		if (gpu && DeviceContext::instance()->getNumDevices() < 1) {
			cerr << "No CUDA device found, reverting to CPU-only version" << endl;
			p.options.usegpu = false;
			gpu = false;
		} else if(!gpu){
			DeviceContext::instance()->setNumDevices(0);
		} else if (p.options.maxGPUs > -1){
			DeviceContext::instance()->setNumDevices(p.options.maxGPUs);
		}
		
		//Random number generator
		#ifdef CPP11
		random_device rd;
		mt19937 mt(rd());
		uniform_int_distribution<int> dist(0, p.n - 1);
		#else
		srand (time(0));
		#endif
		
		//Timing Variables
		clock_t baseTime = clock();
		
		//kernel options struct for computing the kernel
		kernel_opt kernelOptions;
		kernelOptions.kernel = p.options.kernel;
		kernelOptions.gamma = p.options.gamma;
		kernelOptions.degree = p.options.degree;
		kernelOptions.coef = p.options.coef;
		
		//Training examples
		LaspMatrix<T> x = LaspMatrix<double>(p.n, p.features, p.xS).convert<T>();
		
		//Training labels
		LaspMatrix<T> y= LaspMatrix<double>(p.n,1,p.y).convert<T>();
		
		//Explicit bias
		T bias = 0;
		
		//Implicity incorporate bias
		if (p.options.usebias && p.options.bias != 0) {
			p.features++;
			x.resize(p.n, p.features, true, true, p.options.bias);
		}
		
		//Move data to the gpu
		if (gpu) {
			int err = x.transferToDevice();
			err += y.transferToDevice();
			
			if (err != MATRIX_SUCCESS) {
				x.transferToHost();
				y.transferToHost();
				
				cerr << "Device memory insufficient for data, reverting to CPU-only computation" << endl;
				gpu = false;
			}
		}
		
		//Norm of each training vector
		LaspMatrix<T> xNorm (p.n,1, 0.0);
		x.colSqSum(xNorm);
				
		//Output support vectors
		LaspMatrix<T> xS(0, p.features, 0.0, p.n, p.features, true, false);
		LaspMatrix<T> xnormS(0, 1, 0.0, p.n, 1, true, false);
		LaspMatrix<T> alphas(0, 1, 0.0, p.n, 1, true, true); //(betas)
		LaspMatrix<int> alphas_inds(p.n, 1, -1);
		vector<int> S;
		
		//Move support vector stuff to gpu
		if (gpu) {
			xS.transferToDevice();
			xnormS.transferToDevice();
			alphas.transferToDevice();
		}
		
		int k = p.options.set_size;
		T lambda = static_cast<T>(1.0 / p.options.C);
		bool linear = p.options.kernel == 1;
		
		int next_ind = 0;
		
		T w_norm = 0;
		
		//Main loop
		for (int t = 1; t <= p.options.maxiter; ++t) {
			//Generate random set for sub-gradient calculation
			LaspMatrix<int> At_ind(k, 1);
			set<int> unique;
			
			for (int i = 0; i < k; ++i) {
				int ind = getrandom(p.n);
				if (unique.count(ind) > 0) {
					--i;
				} else {
					unique.insert(ind);
					At_ind(i) = ind;
				}
			}
						
			//Gather the set
			LaspMatrix<T> At_x;
			x.gather(At_x, At_ind);
			
			LaspMatrix<T> At_xNorm;
			xNorm.gather(At_xNorm, At_ind);
			
			LaspMatrix<T> At_y;
			y.gather(At_y, At_ind);
			
			LaspMatrix<T> At_kernel;
			LaspMatrix<T> At_dist;
			
			//Compute distances to the hyperplane
			if (next_ind == 0) {
				At_dist = LaspMatrix<T>(k, 1, 0.0);
			}else {
				//Calculate the kernel
				At_kernel = compute_kernel<T>(kernelOptions, xS, xnormS, 0, 0, At_x, At_xNorm, 0, 0, gpu);
				
				At_kernel.colWiseMult(alphas);
				At_kernel.colSum(At_dist);
				At_dist.rowWiseMult(At_y);
				At_dist.add(bias);
				
				//Free up the kernel
				At_kernel = LaspMatrix<T>();
			}
				
			//Update alphas based on the learning rate
			T eta = 1.0 / (lambda * static_cast<T>(t));
			T etaK = eta / static_cast<T>(k);
			
			//For the linear kernel, just update the vector (w) directly
			if (!linear) {
				alphas.multiply(1.0 - (lambda * eta));
			} else {
				xS.multiply(1.0 - (lambda * eta));
			}
			
			w_norm *= (1.0 - (lambda * eta)) * (1.0 - (lambda * eta));
			
			//Move some things back to the host
			if (gpu) {
				At_dist.transferToHost();
				At_y.transferToHost();
				At_ind.transferToHost();
			}
			
			//Subgradient of the bias
			T biasGrad = 0;
			
			//Find the subset that violates the margin
			for (int i = 0; i < k; ++i) { //Subset index
				if (At_dist(i) < 1.0){
					int ind = At_ind(i); //Original vector index
					int alpha_ind = alphas_inds(ind); //Index in support set
					
					T alpha_update = At_y(i) * etaK;
					biasGrad += At_y(i);
					
					//Compute the update to the regularization term
					LaspMatrix<T> w_kernel;
					vector<int> ind_vec;
					ind_vec.push_back(i);
					
					LaspMatrix<T> x_new = At_x.gather(ind_vec);
					LaspMatrix<T> xNorm_new = At_xNorm.gather(ind_vec);
					
					if(next_ind != 0){
						LaspMatrix<T> w_dist;
						w_kernel = compute_kernel<T>(kernelOptions, xS, xnormS, 0, 0, x_new, xNorm_new, 0, 0, gpu);
						w_kernel.colWiseMult(alphas);
						
						if (gpu) {
							w_kernel.transferToHost();
						}
						
						w_kernel.colSum(w_dist);
						w_norm += 2.0 * w_dist(0) * alpha_update;
					}
					
					w_kernel = compute_kernel<T>(kernelOptions, x_new, xNorm_new, 0, 0, x_new, xNorm_new, 0, 0, gpu);
					
					if (gpu) {
						w_kernel.transferToHost();
					}
					
					w_norm += w_kernel(0) * alpha_update * alpha_update;
					
					//If the vector is not a support add it
					if (alpha_ind == -1 && !linear) {
						alpha_ind = next_ind;
						alphas_inds(ind) = alpha_ind;
						++next_ind;
						
						alphas.resize(next_ind, 1);
						xS.resize(next_ind, p.features);
						xnormS.resize(next_ind, 1);
						
						xS.setCol(alpha_ind, At_x, i);
						xnormS.setCol(alpha_ind, At_xNorm, i);
						S.push_back(ind);
					}
					
					//Update the weights
					if (gpu && !linear) {
						LaspMatrix<T> alpha_to_update = alphas(alpha_ind, 0, alpha_ind + 1, 1);
						alpha_to_update.add(alpha_update);
					} else if (!linear) {
						alphas(alpha_ind) += alpha_update;
					} else { //Linear kernel
						//Add a blank vector if nothing is there yet
						if (next_ind == 0) {
							xS.resize(1, p.features);
							xS.multiply(0);
							alphas.resize(1,1);
							alphas.add(1.0);
							S.push_back(1);
							++next_ind;
						}
						
						//Add our new weighted vector to the model
						x_new.multiply(alpha_update);
						xS.add(x_new);
						xS.colSqSum(xnormS);
					}
				}
			}
						
			//Do final alpha update
			T update = w_norm;
			update = (1.0 / sqrt(lambda)) / sqrt(update);
			
			if (update < 1.0) {
				if (!linear) {
					alphas.multiply(update);
				} else {
					xS.multiply(update);
				}
				
				w_norm *= (update * update);
			}
			
			//Update the bias (Not sure if correct)
			if (update > 1) {
				update = 1;
			}
			biasGrad *= (-1.0 / k);
			
			if (p.options.usebias && p.options.bias == 0) {
				bias -= eta * update * biasGrad;
			}
			
			//Print status update
			if (t % 50 == 0 && p.options.verb > 1) {
				//Calculate the objective value for the set
				T obj = (lambda / 2.0) * w_norm;
				
				At_kernel = compute_kernel<T>(kernelOptions, xS, xnormS, 0, 0, At_x, At_xNorm, 0, 0, gpu);
				
				At_kernel.colWiseMult(alphas);
				At_kernel.colSum(At_dist);
				At_dist.rowWiseMult(At_y);
				At_dist.add(bias);
				
				if (gpu) {
					At_dist.transferToHost();
				}
				
				T loss = 0.0;
				for (int i = 0; i < k; ++i) {
					loss += (1.0 / k) * max(0.0, 1.0 - At_dist(i));
				}
				
				obj += loss;
				
				double newTime = ((double)(clock() - baseTime)) / CLOCKS_PER_SEC;
				cout << "At iteration: " << t << ", time: " << newTime << ", support vectors: " << next_ind << ", obj: " << obj << endl;
				cout << "(loss: " << loss << ", regularizer: " << (lambda / 2.0) * w_norm << ", bias: " << bias << ")\n" << endl;
			}
		}
		
		if (gpu) {
			alphas.transferToHost();
			y.transferToHost();
			xS.transferToHost();
		}
		
		//Save output
		p.bs.push_back(bias);
		
		vector<double> betas;
		for (int i = 0; i < alphas.cols(); ++i) {
			betas.push_back(alphas(i));
		}
		
		p.betas.push_back(betas);
		
		p.y = y.template getRawArrayCopy<double>();
		p.xS = xS.template getRawArrayCopy<double>();
		p.S = S;
		
		if (p.options.verb > 0){
			cout << "Training Complete" << endl;
		}
		
		return CORRECT;
	}
	
	template int pegasos_svm_host<float>(svm_problem& p);
	template int pegasos_svm_host<double>(svm_problem& p);
}