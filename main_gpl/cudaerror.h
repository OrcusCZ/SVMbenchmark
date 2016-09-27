/*
Copyright (C) 2015  University of West Bohemia

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>

#define STRINGIZE2(x) #x
#define STRINGIZE(x) STRINGIZE2(x)

//Runtime API error
static void assert_cuda_(cudaError_t t, const char * msg)
{
    if (t == cudaSuccess)
        return;
    std::string w(msg);
    w += ": ";
    w += cudaGetErrorString(t);
    throw std::runtime_error(w);
}

//Driver API error
static void assert_cuda_(CUresult t, const char * msg)
{
    if (t == CUDA_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    const char * str;
    if (cuGetErrorString(t, &str) == CUDA_SUCCESS)
        w += str;
    else
        w += "Unknown error";
    throw std::runtime_error(w);
}

static void assert_cufft_(cufftResult_t t, const char * msg)
{
    if (t == CUFFT_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    switch (t)
    {
        case CUFFT_INVALID_PLAN:
            w += "CUFFT was passed an invalid plan handle"; break;
        case CUFFT_ALLOC_FAILED:
            w += "CUFFT failed to allocate GPU or CPU memory"; break;
        case CUFFT_INVALID_TYPE:
            w += "Unused"; break;
        case CUFFT_INVALID_VALUE:
            w += "User specified an invalid pointer or parameter"; break;
        case CUFFT_INTERNAL_ERROR:
            w += "Used for all driver and internal CUFFT library errors"; break;
        case CUFFT_EXEC_FAILED:
            w += "CUFFT failed to execute an FFT on the GPU"; break;
        case CUFFT_SETUP_FAILED:
            w += "The CUFFT library failed to initialize"; break;
        case CUFFT_INVALID_SIZE:
            w += "User specified an invalid transform size"; break;
        default:
            w += "Unknown CUFFT error";
    }
    throw std::runtime_error(w);
}

static void assert_cublas_(cublasStatus_t t, const char * msg)
{
    if (t == CUBLAS_STATUS_SUCCESS)
        return;
    std::string w(msg);
    w += ": ";
    switch (t)
    {
        case CUBLAS_STATUS_NOT_INITIALIZED:
            w += "The library was not initialized"; break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            w += "The resource allocation failed"; break;
        case CUBLAS_STATUS_INVALID_VALUE:
            w += "An invalid numerical value was used as an argument"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            w += "An absent device architectural feature is required"; break;
        case CUBLAS_STATUS_MAPPING_ERROR:
            w += "An access to GPU memory space failed"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            w += "The GPU program failed to execute"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR:
            w += "An internal operation failed"; break;
        default:
            w += "Unknown CUBLAS error";
    }
    throw std::runtime_error(w);
}

#ifdef _MSC_VER
#define assert_cuda(t) assert_cuda_(t, __FILE__ ":" STRINGIZE(__LINE__) " in function " __FUNCTION__)
#define assert_cufft(t) assert_cufft_(t, __FILE__ ":" STRINGIZE(__LINE__) " in function " __FUNCTION__)
#define assert_cublas(t) assert_cublas_(t, __FILE__ ":" STRINGIZE(__LINE__) " in function " __FUNCTION__)
#else
#define assert_cuda(t) assert_cuda_(t, __FILE__ ":" STRINGIZE(__LINE__))
#define assert_cufft(t) assert_cufft_(t, __FILE__ ":" STRINGIZE(__LINE__))
#define assert_cublas(t) assert_cublas_(t, __FILE__ ":" STRINGIZE(__LINE__))
#endif
