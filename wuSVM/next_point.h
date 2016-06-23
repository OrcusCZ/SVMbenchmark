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

#ifndef LASP_NEXT_POINT_H
#define LASP_NEXT_POINT_H

#include "svm.h"
#include "kernels.h"
#include "lasp_matrix.h"

namespace lasp
{
template<class T>
  int choose_next_point(kernel_opt kernelOptions, vector<int> candidates, vector<int> S, LaspMatrix<T> dXs, LaspMatrix<T> dXnormS, LaspMatrix<T> dX, LaspMatrix<T> dXnorm, int rows, LaspMatrix<T>& d_x, int& d_xInd, LaspMatrix<T> dK2, int dK2cols, int dK2rows, LaspMatrix<T> d_out_minus1, int& d_out_minus1Length, T C, T gamma, LaspMatrix<T> dK2norm, bool);

template<class T>
 void chooseNextHelper(LaspMatrix<T>&, int, LaspMatrix<T>, LaspMatrix<T>, int, LaspMatrix<T>, LaspMatrix<T>,bool);


}
#endif
