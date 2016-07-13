#pragma once

void OrcusSvmCPUTrain(float * alpha, float * rho, bool sparse, const float * x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps);