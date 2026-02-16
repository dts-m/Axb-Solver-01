#pragma once

#include <vector>

namespace svd14 {

using Matrix = std::vector<std::vector<double>>;

struct SVDResult {
    Matrix U;              // m x n (thin)
    std::vector<double> S; // n singular values
    Matrix Vt;             // n x n
};

// Compute thin SVD of A (m x n), recommended for m >= n.
SVDResult svd(const Matrix& A);

// Utility helpers
Matrix transpose(const Matrix& A);
Matrix multiply(const Matrix& A, const Matrix& B);
Matrix diag_from_vector(const std::vector<double>& v);
Matrix scale_columns(const Matrix& A, const std::vector<double>& inv_scales);

} // namespace svd14
