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

#include "train_subset.h"
#include "kernels.h"
#include <iomanip>

template<class T>
double lasp::train_subset_host(svm_problem& p, vector<int> S, LaspMatrix<T> & x, LaspMatrix<T> HESS, vector<int>* & erv, LaspMatrix<T> K, LaspMatrix<T> Ksum, LaspMatrix<T> out, LaspMatrix<T>& old_out, LaspMatrix<T> &x_old, int& x_old_size, LaspMatrix<T> dX, LaspMatrix<T> dY, LaspMatrix<T> dXnorm,  LaspMatrix<T>& dXerv, LaspMatrix<T>& dYerv, LaspMatrix<T>& dXnormerv){
	
	//kernel options struct for computing the kernel
	kernel_opt kernelOptions;
	kernelOptions.kernel = p.options.kernel;
	kernelOptions.gamma = p.options.gamma;
	kernelOptions.degree = p.options.degree;
	kernelOptions.coef = p.options.coef;
	
	bool gpu = p.options.usegpu;
	if (gpu){
		out.transferToDevice();
	}
	
	double C = p.options.C;
	double obj;
	static double old_obj;
	int iter = 0;
	int d = HESS.rows();
	x.resize(1, d, true, true, 0.0);
	K.resize(K.cols(), d);
	vector<int>* old_erv = erv;
	erv = new vector<int>();
	
	LaspMatrix<T> out2(1,p.n,0.0);
	
	LaspMatrix<T> K_S;
	if (p.options.smallKernel) {
		if(S.size() > 0){
			K_S = compute_kernel(kernelOptions, dX, dXnorm, S.data(), S.size(), dX, dXnorm, S.data(), S.size(), p.options.usegpu);
			LaspMatrix<T> y_temp;
			dY.gather(y_temp, S);
			K_S.rowWiseMult(y_temp);
		}
		
	} else {
		LaspMatrix<T> K_rows_1 = K(0,1,K.cols(), d);
		K_rows_1.gather(K_S, S);
	}
	
	if (gpu){
		K_S.transferToDevice();
		K.transferToHost();
		x.transferToHost();
		out2.transferToHost();
	}
	
	K_S.transpose();
	
	if (p.options.smallKernel) {
		if (S.size() == 0) {
			LaspMatrix<T> tempNorm_out2;
			kernel_opt tempOpt_out2;
			out2.getKernel(tempOpt_out2, dY, tempNorm_out2, x, tempNorm_out2, true);
			
		} else {
			int maxElem = 1 + (((2 * p.options.set_size * p.options.set_size) - 1) / p.n);
			int numChunks = 1 + ((S.size() - 1) / maxElem);
			int chunkStart = 0;
			for (int i = 0; i < numChunks; ++i) {
				int chunkSize = (i == numChunks - 1) ? S.size() - chunkStart : S.size() / numChunks;
				
				LaspMatrix<T> K_temp;
				if(chunkStart > 0){
					K_temp = compute_kernel(kernelOptions, dX, dXnorm, S.data() + chunkStart, chunkSize, dX, dXnorm, 0, 0, gpu);
				} else {
					K_temp = LaspMatrix<T>(p.n, chunkSize + 1, 1.0);
					LaspMatrix<T> K_temp_temp = compute_kernel(kernelOptions, dX, dXnorm, S.data() + chunkStart, chunkSize, dX, dXnorm, 0, 0, gpu);
					LaspMatrix<T> K_temp_sub = K_temp(0, 1, K_temp.cols(), K_temp.rows());
					K_temp_sub.copy(K_temp_temp);
				}
				
				K_temp.rowWiseMult(dY);
				
				LaspMatrix<T> out2_temp;
				LaspMatrix<T> x_temp = (chunkStart == 0) ? x(0, chunkStart, 1, chunkStart + chunkSize + 1) : x(0, chunkStart + 1, 1, chunkStart + chunkSize + 1);
				
				LaspMatrix<T> tempNorm_out2;
				kernel_opt tempOpt_out2;
				out2_temp.getKernel(tempOpt_out2, K_temp, tempNorm_out2, x_temp, tempNorm_out2, true);
				
				out2.add(out2_temp);
				
				chunkStart += chunkSize;
			}
		}
	} else {
		LaspMatrix<T> tempNorm_out2;
		kernel_opt tempOpt_out2;
		out2.getKernel(tempOpt_out2, K, tempNorm_out2, x, tempNorm_out2, true);
	}
		
	vector<int> *erv2 = new vector<int>();
	
	//Not sure if there is anything to improve this
	for(int i = 0; i < p.n; ++i){
		if(out2(i) < 1){
			erv2->push_back(i);}
	}
	
	double obj2;

	calculate_obj( obj2, S,d,x,K,out2,erv2,p.options.C, dY, dX, dXnorm, p, K_S);
	
	if (S.size() != 0 && obj2 > old_obj){
		if (gpu){
			x_old.transferToHost();
		}

		{
			for(int i=1;i<=(d-x_old_size);++i){
				x_old(x_old_size+i-1) = 0;
			}
		}
		if (gpu){
			old_out.transferToHost();
		}
		x_old_size = d;
		for(int i=0; i < p.n; ++i){
			if(old_out(i) < 1){
				erv->push_back(i);
			}
		}
		delete erv2;
	}
	
	else{
		//TODO do we need to delete x_old first?
		x_old=x;
		x_old_size = d;
		x = LaspMatrix<T>(1, d, 0.0);
		old_obj=obj2;
		delete erv;
		erv=erv2;
		old_out=out2;
	}
		
	while (true){
		iter++;
		
		vector<int> erv_adds = lasp::mysetdiff(*erv, *old_erv);
		vector<int> erv_subs = lasp::mysetdiff(*old_erv, *erv);
		
		//this needs to remain doubles even on device
		//Could these also be incrementally updated? (Sum over cols of K?)
		LaspMatrix<double> kadds;
		LaspMatrix<double> ksubs;
		LaspMatrix<double> adds_and_subs(d,d,0.0);
		
		{
			LaspMatrix<T> kadds_Generic;
			LaspMatrix<T> ksubs_Generic;
			
			if (p.options.smallKernel) {
				//kernel options struct for computing the kernel
				kernel_opt kernelOptions;
				kernelOptions.kernel = p.options.kernel;
				kernelOptions.gamma = p.options.gamma;
				kernelOptions.degree = p.options.degree;
				kernelOptions.coef = p.options.coef;
					
				{
					kadds_Generic = LaspMatrix<T>(erv_adds.size(), d, 1.0);
					
					if (d > 1) {
						LaspMatrix<T> kadds_Generic_temp = compute_kernel(kernelOptions, dX, dXnorm, S.data(), d - 1, dX, dXnorm, erv_adds.data(), erv_adds.size(), gpu);
						LaspMatrix<T> kadds_Generic_sub = kadds_Generic(0, 1, kadds_Generic.cols(), kadds_Generic.rows());
						kadds_Generic_sub.copy(kadds_Generic_temp);
					}
						
					LaspMatrix<T> kaddsY;
					dY.gather(kaddsY, erv_adds);
					kadds_Generic.rowWiseMult(kaddsY);
				}
				
				{
					ksubs_Generic = LaspMatrix<T>(erv_subs.size(), d, 1.0);
					
					if (d > 1) {
						LaspMatrix<T> ksubs_Generic_temp = compute_kernel(kernelOptions, dX, dXnorm, S.data(), d - 1, dX, dXnorm, erv_subs.data(), erv_subs.size(), gpu);
						LaspMatrix<T> ksubs_Generic_sub = ksubs_Generic(0, 1, ksubs_Generic.cols(), ksubs_Generic.rows());
						ksubs_Generic_sub.copy(ksubs_Generic_temp);
					}
					
					LaspMatrix<T> ksubsY;
					dY.gather(ksubsY, erv_subs);
					ksubs_Generic.rowWiseMult(ksubsY);
				}
				
				
			} else {
				LaspMatrix<T> K_d = K(0, 0, K.cols(), d);
				
				K_d.gather(kadds_Generic, erv_adds);
				K_d.gather(ksubs_Generic, erv_subs);
			}
			
			kadds_Generic.transpose();
			ksubs_Generic.transpose();
			
			kadds = kadds_Generic.template convert<double>();
			ksubs = ksubs_Generic.template convert<double>();
			
			LaspMatrix<T> kadds_Sum = kadds_Generic.colSum();
			LaspMatrix<T> ksubs_Sum = ksubs_Generic.colSum();
			
			kadds_Sum.transpose();
			ksubs_Sum.transpose();
			
			LaspMatrix<T> Ksum_d = Ksum(0, 0, 1, d);
			Ksum_d.add(kadds_Sum);
			Ksum_d.subtract(ksubs_Sum);
		}
		
		if (gpu){
			kadds.transferToDevice();
			ksubs.transferToDevice();
		}
		
		if (erv_adds.size() > 0 ){
			kadds.multiply(kadds,adds_and_subs,true,false,C,0);
		}
		
		double negC = -C;
		
		if (erv_subs.size()>0){
			ksubs.multiply(ksubs,adds_and_subs,true,false,negC,1.0);
		}
		
		if (gpu){
			HESS.transferToHost();
			adds_and_subs.transferToHost();
		}
		
		{
			LaspMatrix<T> add_and_subs_Generic = adds_and_subs.template convert<T>();
			HESS.add(add_and_subs_Generic);
		}
		
		{
			LaspMatrix<T> HESS_diag = HESS.diag();
			LaspMatrix<T> HESS_diagSum = HESS_diag.colSum();
			
			if(gpu){
				HESS_diagSum.transferToHost();
			}
			
			T diagMean_Ep = (HESS_diagSum(0) / d) * 1e-10;
			HESS.diagAdd(diagMean_Ep);
		}

		delete old_erv;
		old_erv = erv;
		erv = 0;
		
		if (gpu){
			x.transferToHost();
			Ksum.transferToHost();
		}
		
		{
			LaspMatrix<T> x_d = x(0,0,1,d);
			LaspMatrix<T> Ksum_d = Ksum(0,0,1,d);
			x_d.copy(Ksum_d);
		}
		
		LaspMatrix<T> HESS_CPY;
		if (gpu) {
			HESS_CPY.transferToDevice();
		}
		
		HESS_CPY.copy(HESS);

		LaspMatrix<double> dHESS_CPY = HESS_CPY.template convert<double>();
		LaspMatrix<double> dx = x.template convert<double>();
		
		if(gpu){
			dHESS_CPY.transferToHost();
			dx.transferToHost();
		}

		dHESS_CPY.solve(dx);
		
		HESS_CPY = dHESS_CPY.template convert<T>();
		x = dx.template convert<T>();

		LaspMatrix<T> step = LaspMatrix<T>(1,d);
		{
			LaspMatrix<T> x_d = x(0,0,1,d);
			x_d.multiply(C);
						
			if(gpu){
				x.transferToHost();
			}
			
			LaspMatrix<T> x_old_d = x_old(0,0,1,d);
			x_d.subtract(x_old_d, step);
		}
				
		LaspMatrix<T> delta_out (1,p.n,0.0);
		if (p.options.smallKernel) {
			if (S.size() == 0) {
				LaspMatrix<T> tempNorm_delta_out;
				kernel_opt tempOpt_delta_out;
				delta_out.getKernel(tempOpt_delta_out, dY, tempNorm_delta_out, step, tempNorm_delta_out, true);
				
			} else {
				int maxElem = 1 + (((2 * p.options.set_size * p.options.set_size) - 1) / p.n);
				int numChunks = 1 + ((S.size() - 1) / maxElem);
				int chunkStart = 0;
				for (int i = 0; i < numChunks; ++i) {
					int chunkSize = (i == numChunks - 1) ? S.size() - chunkStart : S.size() / numChunks;
					
					LaspMatrix<T> K_temp;
					if(chunkStart > 0){
						K_temp = compute_kernel(kernelOptions, dX, dXnorm, S.data() + chunkStart, chunkSize, dX, dXnorm, 0, 0, gpu);
					} else {
						K_temp = LaspMatrix<T>(p.n, chunkSize + 1, 1.0);
						LaspMatrix<T> K_temp_temp = compute_kernel(kernelOptions, dX, dXnorm, S.data() + chunkStart, chunkSize, dX, dXnorm, 0, 0, gpu);
						LaspMatrix<T> K_temp_sub = K_temp(0, 1, K_temp.cols(), K_temp.rows());
						K_temp_sub.copy(K_temp_temp);
					}
					
					K_temp.rowWiseMult(dY);
									
					LaspMatrix<T> delta_out_temp;
					LaspMatrix<T> step_temp = (chunkStart == 0) ? step(0, chunkStart, 1, chunkStart + chunkSize + 1) : step(0, chunkStart + 1, 1, chunkStart + chunkSize + 1);
					
					LaspMatrix<T> tempNorm_delta_out;
					kernel_opt tempOpt_delta_out;
					delta_out_temp.getKernel(tempOpt_delta_out, K_temp, tempNorm_delta_out, step_temp, tempNorm_delta_out, true);
					
					delta_out.add(delta_out_temp);
					
					chunkStart += chunkSize;
				}
			}
		} else {
			LaspMatrix<T> tempNorm_delta_out;
			kernel_opt tempOpt_delta_out;
			delta_out.getKernel(tempOpt_delta_out, K, tempNorm_delta_out, step, tempNorm_delta_out, true);
		}
		
		int iii;
		double iii_2 = 1.0;
		for(iii = 0; iii <= p.options.maxiter; ++iii){
			LaspMatrix<T> step_temp;
			step.eWiseOp(step_temp, 0, (1/iii_2), 1);
			
			LaspMatrix<T> delta_temp;
			delta_out.eWiseOp(delta_temp, 0, (1/iii_2), 1);
			
			iii_2 *= 2.0;
			
			step_temp.addMatrix(x_old, x);
			
			if (gpu){
				out.transferToHost();
				old_out.transferToHost();
			}
			
			delta_temp.addMatrix(old_out, out);
			erv = new vector<int>();
			
			if (gpu){
				out.transferToHost();
			}
			
			for (int i = 0; i < p.n; i++) {
				if (out(i) < 1.0){
					erv->push_back(i);
				}
			}
						
			calculate_obj(obj, S, d, x, K, out, erv, C, dY, dX, dXnorm, p, K_S);
			
			if (obj < old_obj || obj != obj){
				break;
			}
		}
				
		x_old = LaspMatrix<T>(1, HESS.mRows(), 0.0);
		
		if (gpu){
			x.transferToHost();
		}
		
		{
			LaspMatrix<T> x_d = x(0,0,1,d);
			LaspMatrix<T> x_old_d = x_old(0,0,1,d);
			x_old_d.copy(x_d);
		}
		
		if (obj < old_obj){
			old_obj=obj;
		}
		
		old_out.copy(out);

		if (p.options.verb > 2){
			cout << "Nb basis = " << HESS.rows()-1  <<", iter Newton = " << iter-1 << ", backtracking = " << iii  << ", nerv = " << erv->size() << ", erv change = " << erv_adds.size()+erv_subs.size() << ", Obj = " << obj << endl;
		}
		
		if ((old_erv->size()==erv->size() && mysetdiff(*old_erv,*erv).empty()) || obj != obj){
			dXerv = LaspMatrix<T>(erv->size(), p.features);
			dYerv = LaspMatrix<T>(1, erv->size());
			dXnormerv = LaspMatrix<T>(1, erv->size());
			
			if (gpu){
				dX.transferToDevice();
				dY.transferToDevice();
				dXnorm.transferToDevice();
			}
			
			dX.gather(dXerv,*erv);
			dY.gather(dYerv,*erv);
			dXnorm.gather(dXnormerv,*erv);
			break;
		}
	}

	return obj;
}

//calculates the value of the objective function we are trying to minimize
template<class T>
void lasp::calculate_obj(double& obj, vector<int> S, int d, LaspMatrix<T> x, LaspMatrix<T> K, LaspMatrix<T> out, vector<int>* erv, int C, LaspMatrix<T> dY, LaspMatrix<T> dX, LaspMatrix<T> dXnorm, svm_problem& p, LaspMatrix<T> K_S_in){
	obj = 0;
	LaspMatrix<T> xTemp(1, d-1, 0.0);

	LaspMatrix<T> K_row_0 = dY;
	LaspMatrix<T> K_S_0;
	K_row_0.gather(K_S_0, S);
	
	LaspMatrix<T> x_rows_1 = x(0,1,1,S.size()+1);
	LaspMatrix<T> val;
	x_rows_1.colWiseMult(K_S_0, val);
	
	LaspMatrix<T> K_S;
	LaspMatrix<T> K_S_sum;
		
	K_S_in.colWiseMult(val, K_S);
	K_S.colSum(K_S_sum);
		
	K_S_sum.transpose();
	xTemp.add(K_S_sum);

	LaspMatrix<T> update;
	if(d > 1){
		LaspMatrix<T> x_d_minus_one = x(0,1,1,d);
		x_d_minus_one.multiply(xTemp, update, true, false);
		obj += update(0);
	}
	
	LaspMatrix<T> out_erv;
	out.transpose();
	out.gather(out_erv, *erv);
	out.transpose();
		
	out_erv.eWiseOp(out_erv, 1, -1, 1);
	out_erv.eWiseOp(out_erv, 0, C, 2);
		
	out_erv.transpose();
	update = LaspMatrix<T>();
	out_erv.colSum(update);
	obj += update(0);

	obj *= .5;
}

template double lasp::train_subset_host<float>(svm_problem& p, vector<int> S, LaspMatrix<float> & x, LaspMatrix<float> HESS, vector<int>* & erv, LaspMatrix<float> K, LaspMatrix<float> Ksum, LaspMatrix<float> out, LaspMatrix<float>& old_out, LaspMatrix<float> &x_old, int& x_old_size, LaspMatrix<float> dX, LaspMatrix<float> dY, LaspMatrix<float> dXnorm,  LaspMatrix<float>& dXerv, LaspMatrix<float>& dYerv, LaspMatrix<float>& dXnormerv);
template double lasp::train_subset_host<double>(svm_problem& p, vector<int> S, LaspMatrix<double> & x, LaspMatrix<double> HESS, vector<int>* & erv, LaspMatrix<double> K, LaspMatrix<double> Ksum, LaspMatrix<double> out, LaspMatrix<double>& old_out, LaspMatrix<double> &x_old, int& x_old_size, LaspMatrix<double> dX, LaspMatrix<double> dY, LaspMatrix<double> dXnorm,  LaspMatrix<double>& dXerv, LaspMatrix<double>& dYerv, LaspMatrix<double>& dXnormerv);

template void lasp::calculate_obj<float>(double& obj, vector<int> S, int d, LaspMatrix<float> x, LaspMatrix<float> K, LaspMatrix<float> out, vector<int>* erv, int C, LaspMatrix<float> dY, LaspMatrix<float> dX, LaspMatrix<float> dXnorm, svm_problem& p, LaspMatrix<float> K_S_in);
template void lasp::calculate_obj<double>(double& obj, vector<int> S, int d, LaspMatrix<double> x, LaspMatrix<double> K, LaspMatrix<double> out, vector<int>* erv, int C, LaspMatrix<double> dY, LaspMatrix<double> dX, LaspMatrix<double> dXnorm, svm_problem& p, LaspMatrix<double> K_S_in);
