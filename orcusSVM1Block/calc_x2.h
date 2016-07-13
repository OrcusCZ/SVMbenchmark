#pragma once
#include "csr.h"

namespace OrcusSVM1B
{
    void computeX2Dense(float * d_x2, const float * d_x, int num_vec, int num_vec_aligned, int dim, int dim_aligned);
    void computeX2DenseT(float * d_x2, const float * d_xT, int num_vec, int num_vec_aligned, int dim, int dim_aligned);
    void computeX2Sparse(float * d_x2, const csr_gpu & x, int num_vec);
    void computeX2Sparse(float * d_x2, const ellpack_gpu & x, int num_vec);
    void computeX2Sparse(float * d_x2, const jds_gpu & x, int num_vec);
}
