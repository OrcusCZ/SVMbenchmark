#pragma once

#include <string>

void export_c_buffer(const void * data, size_t width, size_t height, size_t elemsize, std::string filename);
void export_cuda_buffer(const void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);