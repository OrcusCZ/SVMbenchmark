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

#ifndef LASP_ABSTRACT_MATRIX_H
#define LASP_ABSTRACT_MATRIX_H

#include <iostream>
using namespace std;

namespace lasp
{
	//Matrix Error Enum
	enum { MATRIX_SUCCESS = 0,
		UNSPECIFIED_MATRIX_ERROR, //catch all error, return this if you can't return something better
		INVALID_DIMENSIONS, //matricies have the wrong number of dimensions
		UNALLOCATED_OUTPUT_MATRIX, //output matrix has not been allocated yet.
		MATRIX_TYPE_MISMATCH, //operation attempted on two incompatable matricies
		METHOD_NOT_IMPLEMENTED, //operation not implemented for a given type
		ARGUMENT_INVALID, //Operation give an illegal argument
		CANNOT_COMPLETE_OPERATION,
		INVALID_LOCATION,
		OUT_OF_BOUNDS
	};
}
#endif
