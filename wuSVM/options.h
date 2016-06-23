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

#ifndef LASP_OPTIONS_H
#define LASP_OPTIONS_H

namespace lasp{

enum kernel_types { RBF, LINEAR, POLYNOMIAL, SIGMOID, PRECOMPUTED};

struct opt {
    int nb_cand;
    int set_size;
	  int start_size;
    int maxiter;
    double base_recomp;
    int verb;
    int contigify;
    int maxnewbasis;
    double stoppingcriterion;
    int candbufsize;
    int maxcandbatch;
    int kernel;
    double C;
    double gamma;
    double coef;
    double degree;
    char* modelFile;
    int plattScale;
    bool usegpu;
  	bool randomize;
  	bool single;
  	bool pegasos;
  	bool usebias;
  	double bias;
  	bool shuffle;
    bool smallKernel;
	  int maxGPUs;
	int stopIters;
  };

  //struct holding info regarding how kernels are computed
  struct kernel_opt {
    int kernel;
    double gamma;
    double coef;
    double degree;
  };
}

#endif