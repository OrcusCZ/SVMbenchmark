#pragma once

void OpenCLSvmTrain(float * alpha, float * rho, bool sparse, const float * x, const float * y, unsigned int num_vec, unsigned int num_vec_aligned, unsigned int dim, unsigned int dim_aligned, float C, float gamma, float eps);
