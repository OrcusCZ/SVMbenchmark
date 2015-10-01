#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include "debug.h"
#include "cudaerror.cuh"

#ifdef _MSC_VER
#if _MSC_VER < 1700
	#define uint32_t unsigned int
#else
	#include <cstdint>
#endif
#else //linux
	#include <cstdint>
#endif
void export_c_buffer(const void * data, size_t width, size_t height, size_t elemsize, std::string filename)
{
    std::cout << "[Debug] Exporting C buffer of size " << width << " x " << height << " x " << elemsize << " B to file '" << filename.c_str() << "'\n";
    size_t datasize = width * height * elemsize;

    uint32_t w = (uint32_t) width,
        h = (uint32_t) height,
        e = (uint32_t) elemsize;
    std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
    fout.write((const char *)&w, sizeof(w));
    fout.write((const char *)&h, sizeof(h));
    fout.write((const char *)&e, sizeof(e));
    fout.write((const char *)data, datasize);
    fout.close();
}

void export_cuda_buffer(const void * d_data, size_t width, size_t height, size_t elemsize, std::string filename)
{
    std::cout << "[Debug] Exporting CUDA buffer of size " << width << " x " << height << " x " << elemsize << " B to file '" << filename.c_str() << "'\n";

    size_t datasize = width * height * elemsize;
    char * data = new char [datasize];
    assert_cuda(cudaMemcpy(data, d_data, datasize, cudaMemcpyDeviceToHost));

    uint32_t w = (uint32_t) width,
        h = (uint32_t) height,
        e = (uint32_t) elemsize;
    std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
    fout.write((const char *)&w, sizeof(w));
    fout.write((const char *)&h, sizeof(h));
    fout.write((const char *)&e, sizeof(e));
    fout.write((const char *)data, datasize);
    fout.close();

    delete [] data;
}