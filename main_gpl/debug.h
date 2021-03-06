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

#include <string>
#include <iostream>
#include <cuda_runtime.h>

#ifdef USE_TIMERS
#define TIME_KERNEL(kernel) \
do { \
    cudaEvent_t e1, e2; \
    cudaEventCreate(&e1); \
    cudaEventCreate(&e2); \
    cudaEventRecord(e1); \
    kernel; \
    cudaEventRecord(e2); \
    cudaEventSynchronize(e2); \
    float t; \
    cudaEventElapsedTime(&t, e1, e2); \
    std::cout << "Kernel " << #kernel << " elapsed time: " << t << " ms\n"; \
    cudaEventDestroy(e1); \
    cudaEventDestroy(e2); \
} while (false)

#define ACCUMULATE_KERNEL_TIME(timer, counter, kernel) \
do { \
    cudaEvent_t e1, e2; \
    cudaEventCreate(&e1); \
    cudaEventCreate(&e2); \
    cudaEventRecord(e1); \
    kernel; \
    cudaEventRecord(e2); \
    cudaEventSynchronize(e2); \
    float t; \
    cudaEventElapsedTime(&t, e1, e2); \
    timer += t; \
	counter++; \
    cudaEventDestroy(e1); \
    cudaEventDestroy(e2); \
} while (false)

#define PRINT_KERNEL_TIME(label, timer, counter) std::cout << "Kernel " << label << ": " << counter << " calls, average: " << ((counter > 0) ? timer/counter : 0) << " ms, total: " << timer << " ms" << std::endl;
#else
#define TIME_KERNEL(kernel) kernel
#define ACCUMULATE_KERNEL_TIME(timer, counter, kernel) kernel
#define PRINT_KERNEL_TIME(label, timer, counter)
#endif

void export_c_buffer(const void * data, size_t width, size_t height, size_t elemsize, std::string filename);
void export_cuda_buffer(const void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);
void import_cuda_buffer(void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);

template<typename T>
void print_cuda_buffer(const T * d_data, size_t width, size_t height, std::ostream & os = std::cout)
{
	size_t datasize = width * height;
	T * data = new T[datasize];
	assert_cuda(cudaMemcpy(data, d_data, datasize * sizeof(T), cudaMemcpyDeviceToHost));

	for (int i = 0; i < width * height; i++)
		os << data[i] << " ";
	os << std::endl;

	delete[] data;
}

template<typename T>
void print_c_buffer(const T * data, size_t width, size_t height, std::ostream & os = std::cout)
{
	for (int i = 0; i < width * height; i++)
		os << data[i] << " ";
	os << std::endl;
}
