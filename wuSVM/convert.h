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

#ifdef WUCUDA
namespace lasp{
	class DeviceParams;

  	template<class N, class T>
  	__global__ void device_kernel_convert(T* in, N* out, int rows, int cols, int mRows){
    	int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    	int index_y = blockDim.y * blockIdx.y + threadIdx.y;
	    int grid_width = gridDim.x * blockDim.x;
	    int threadID = index_y * grid_width + index_x;

	    int col = threadID / rows;
	    int row = threadID % rows;
	    if(threadID < (rows * cols)){
	      out[col * mRows + row] = (N)in[col * mRows + row];
	    }
  	}

  	template<class N, class T>
  	int device_convert_helper(DeviceParams params, T* in, N* out, int rows, int cols, int mRows){
  		int threadsperblock(256);
	    dim3 blockspergrid;
	    
	    blockspergrid.x =((cols * rows + threadsperblock-1) / threadsperblock)/100 + 1;
	    blockspergrid.y = 100;
	    device_kernel_convert<<<blockspergrid,threadsperblock>>>(in, out, rows, cols, mRows);
	    return 0;
  	}
}
#endif