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

#ifndef LASP_TRAIN_SUBSET_H
#define LASP_TRAIN_SUBSET_H

#include "lasp_svm.h"
#include "lasp_matrix.h"

namespace lasp
{

template<class T>
double train_subset_host(svm_problem& p, vector<int> S, LaspMatrix<T> & x, LaspMatrix<T> HESS, vector<int>* & erv, LaspMatrix<T> K, LaspMatrix<T> Ksum, LaspMatrix<T> out, LaspMatrix<T>& old_out, LaspMatrix<T> &x_old, int& x_old_size, LaspMatrix<T> dX, LaspMatrix<T> dY, LaspMatrix<T> dXnorm,  LaspMatrix<T>& dXerv, LaspMatrix<T>& dYerv, LaspMatrix<T>& dXnormerv );
 
template<class T>
 void calculate_obj(double& obj, vector<int> S, int d, LaspMatrix<T> x, LaspMatrix<T> K, LaspMatrix<T> out, vector<int>* erv, int C, LaspMatrix<T> dY, LaspMatrix<T> dX, LaspMatrix<T> dXnorm, svm_problem& p, LaspMatrix<T> K_S_in);
}
#endif