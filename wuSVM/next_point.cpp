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

#include "next_point.h"

template<class T>
int lasp::choose_next_point(kernel_opt kernelOptions, vector<int> candidates, vector<int> S, LaspMatrix<T> dXs, LaspMatrix<T> dXnormS, LaspMatrix<T> dX, LaspMatrix<T> dXnorm, int rows, LaspMatrix<T>& d_x, int& d_xInd, LaspMatrix<T> dK2, int dK2cols, int dK2rows, LaspMatrix<T> d_out_minus1, int& d_out_minus1Length, T C, T gamma, LaspMatrix<T> dK2norm, bool useGPU){
	
	
	bool gpu = useGPU;
	
	if (gpu){
		dX.transferToDevice();
		dXnorm.transferToDevice();
		
	}
	
	d_out_minus1.resize(1,dK2.rows());
	
	LaspMatrix<T> dXc;
	dX.gather(dXc, candidates);
	
	LaspMatrix<T> dXnormc;
	dXnorm.gather(dXnormc, candidates);
	
	int indS = S.size();
	LaspMatrix<T> dK3;
	
	if(indS != 0){
		dK3 = compute_kernel<T>(kernelOptions, dXs, dXnormS, 0, 0, dXc, dXnormc, 0, 0, gpu);
	}
	
	LaspMatrix<T> h(1,dK2cols,0.0);
	LaspMatrix<T> g;
	LaspMatrix<T> g2;
	
	T alpha = 1;
	T beta = 0;
	
	if (gpu){
		h.transferToDevice();
		g.transferToDevice();
		dK2norm.transferToDevice();
		dK2.transferToDevice();
		dK3.transferToDevice();
		g2.transferToDevice();
	}
	
	dK2norm.eWiseOp(h, 1, C, 1);
	
	if(indS != 0){
		dK3.multiply(d_x, g , true, false, alpha, beta, 1);
	}
	
	dK2.multiply(d_out_minus1,g2,true,false,C,beta);
	
	if (gpu){
		
		g.transferToDevice();
		g2.transferToDevice();
	}
	
	
	if (indS==0){
		
		g=g2;
	}
	else{
		g.add(g2);
	}
	
	LaspMatrix<T> score;
	
	h.swap();
	g.eWiseDivM(h, score, 2, 1);
	
	if (gpu){
		score.transferToHost();
	}
	
	T maxscore = score(0);
	int select = 0;
	for (int i = 0; i< candidates.size(); ++i){
		if (score(i) > maxscore){
			maxscore = score(i);
			select = i;
		}
	}
	
	d_x.resize(1, d_x.rows()+1);
	chooseNextHelper<T>(d_x, d_xInd, g, h, select, d_out_minus1, dK2, gpu);
	d_xInd++;
	
	select = candidates[select];
	return select;
}

template<class T>
void lasp::chooseNextHelper(LaspMatrix<T>& d_x, int d_xInd, LaspMatrix<T> g, LaspMatrix<T> h, int select, LaspMatrix<T> d_out_minus1, LaspMatrix<T> dK2, bool useGPU){
	
	if (useGPU){
		d_x.transferToDevice();
	}
	
	//This piece of code is broken
	if (false && d_x.device()){
		g.transferToDevice();
		h.transferToDevice();
		d_out_minus1.transferToDevice();
		dK2.transferToDevice();
		DeviceParams params = d_x.context().setupOperation(&d_x, &dK2);
		device_chooseNextHelper(params, d_x.dData(),d_xInd,g.dData(),h.dData(),select,d_out_minus1.dData(),dK2.dData(),dK2.rows(),dK2.cols());
	}
	
	else{
		T xend = -g(select) / h(select);
		d_x(d_xInd) = xend;
		
		LaspMatrix<T> dK2_select_org = dK2(select, 0, select+1, dK2.rows());
		LaspMatrix<T> dK2_select;
		dK2_select.copy(dK2_select_org);
		dK2_select.multiply(xend);
		d_out_minus1.add(dK2_select);
		
	}
}

template int lasp::choose_next_point<float>(kernel_opt kernelOptions, vector<int> candidates, vector<int> S, LaspMatrix<float> dXs, LaspMatrix<float> dXnormS, LaspMatrix<float> dX, LaspMatrix<float> dXnorm, int rows, LaspMatrix<float>& d_x, int& d_xInd, LaspMatrix<float> dK2, int dK2cols, int dK2rows, LaspMatrix<float> d_out_minus1, int& d_out_minus1Length, float C, float gamma, LaspMatrix<float> dK2norm, bool useGPU);
template int lasp::choose_next_point<double>(kernel_opt kernelOptions, vector<int> candidates, vector<int> S, LaspMatrix<double> dXs, LaspMatrix<double> dXnormS, LaspMatrix<double> dX, LaspMatrix<double> dXnorm, int rows, LaspMatrix<double>& d_x, int& d_xInd, LaspMatrix<double> dK2, int dK2cols, int dK2rows, LaspMatrix<double> d_out_minus1, int& d_out_minus1Length, double C, double gamma, LaspMatrix<double> dK2norm, bool useGPU);

template void lasp::chooseNextHelper<float>(LaspMatrix<float>& d_x, int d_xInd, LaspMatrix<float> g, LaspMatrix<float> h, int select, LaspMatrix<float> d_out_minus1, LaspMatrix<float> dK2, bool useGPU);
template void lasp::chooseNextHelper<double>(LaspMatrix<double>& d_x, int d_xInd, LaspMatrix<double> g, LaspMatrix<double> h, int select, LaspMatrix<double> d_out_minus1, LaspMatrix<double> dK2, bool useGPU);

