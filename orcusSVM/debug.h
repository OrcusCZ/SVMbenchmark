#pragma once

#include <string>

void export_c_buffer(void * data, size_t width, size_t height, size_t elemsize, std::string filename);
void export_cuda_buffer(void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);