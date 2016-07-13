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

#include "retraining.h"
#include "svm.h"

int lasp::retrain( int d, int dprev, svm_problem& p){
    int r = 0;
    double b = p.options.base_recomp;
    if (d <1){
		r = 1;
		return r;
    }
		
    if (d > 0 && d < p.options.start_size){
		r = 0;
		return r;
    }
	
    if (d == p.options.start_size){
		r = 1;
		return r;
    }
	
	
    r = (int) (floor(log((double)d)/log(b)) != floor(log((double)d-1)/log(b)));
    r = (int) ((d>dprev && r) || (d == p.options.set_size));
    return r;
}

int lasp::retrain2( int d, int dprev, svm_problem& p){
    int r = 0;
    double b = p.options.base_recomp;
    if (d <1){
		r = 1;
		return r;
    }
	
    if (d > 0 && d < 99){
		r = 0;
		return r;
    }
	
    if (d == 99){
		r = 1;
		return r;
    }
	
    r = (int) (floor(log((double)d)/log(b)) != floor(log((double)d-1)/log(b)));
    r = (int) ((d>dprev && r) || (d == p.options.set_size));
    return r;
}

vector<int> lasp::retrainIters(svm_problem& p)
{
    vector<int> r;
    for (int i=0; i< p.options.set_size; ++i){
		int x = retrain(i,0,p);
		if (x == 1){
			r.push_back(i);
		}
    }
	
    int ri = 0;
    for (int i = 1; i < r.size(); ++i){
		int temp = r[i] - r[i-1];
		if ( temp < p.options.maxnewbasis ){
			ri = i;
		}
    }
	
    r.resize(ri+1);
    int i=0;
    while (r.back() < p.options.set_size){
		r.push_back(r.back()+p.options.maxnewbasis);
		++i;
    }
	
    r.back() = p.options.set_size;
    return r;
}

