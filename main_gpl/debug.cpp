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

#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include "debug.h"
#include "cudaerror.h"

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

void import_cuda_buffer(void * d_data, size_t width, size_t height, size_t elemsize, std::string filename)
{
    std::cout << "[Debug] Importing CUDA buffer of size " << width << " x " << height << " x " << elemsize << " B to file '" << filename.c_str() << "'\n";

    uint32_t w, h, e;
    std::ifstream fin(filename.c_str(), std::ios::in | std::ios::binary);
    fin.read((char *)&w, sizeof(w));
    fin.read((char *)&h, sizeof(h));
    fin.read((char *)&e, sizeof(e));
    if (w == width && h == height && e == elemsize)
    {
        size_t datasize = width * height * elemsize;
        char * data = new char [datasize];
        fin.read(data, datasize);
        assert_cuda(cudaMemcpy(d_data, data, datasize, cudaMemcpyHostToDevice));
        delete [] data;
    }
    else
        std::cerr << "[Error] Cannot import CUDA buffer, dimension mismatch\n";
    fin.close();
}
