#pragma once
#include "csr.h"

namespace OrcusSvm1B
{
    union Data
    {
        const float * dense;
        const OrcusSVM1B::csr * sparse;
    };

    ///////////////////////////////////////////////
    /// \brief Trains SVM model on GPU
    /// \param alpha output array for trained alphas, must be allocated by the caller using malloc and large enough to hold num_vec elements
    /// \param rho output value
    /// \param sparse true if training data is sparse
    /// \param x training data. if dense x.dense contains linearized matrix aligned to num_vec_aligned, if sparse x.sparse contains data in csr format
    /// \param y training data labels, must contain values +1 or -1
    /// \param num_vec number of training vectors
    /// \param num_vec_aligned number of training vectors aligned up to the multiple of size of warp (32)
    /// \param dim training data dimension
    /// \param dim_aligned training data dimension aligned up to the multiple of size of warp (32)
    /// \param C training parameter C
    /// \param gamma RBF kernel parameter gamma
    /// \param eps training threshold
    /// \param ws_size working set size, 0 means to choose size automatically. it migt be necessary to lower the size for large datasets
    ///////////////////////////////////////////////
    void Train(float * alpha, float * rho, bool sparse, const Data & x, const float * y, size_t num_vec, size_t num_vec_aligned, size_t dim, size_t dim_aligned, float C, float gamma, float eps, int ws_size = 0);
}
