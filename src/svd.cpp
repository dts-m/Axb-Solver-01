#include "svd.h"

#include <cmath>
#include <algorithm>
#include <limits>

namespace svd14 {

static Matrix identity(size_t n) {
    Matrix I(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

Matrix transpose(const Matrix& A) {
    if (A.empty()) return {};
    size_t m = A.size();
    size_t n = A[0].size();
    Matrix T(n, std::vector<double>(m, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}

Matrix multiply(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty()) return {};
    size_t m = A.size();
    size_t n = B[0].size();
    size_t k = B.size();
    Matrix C(m, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t p = 0; p < k; ++p) {
                sum += A[i][p] * B[p][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

Matrix diag_from_vector(const std::vector<double>& v) {
    Matrix D(v.size(), std::vector<double>(v.size(), 0.0));
    for (size_t i = 0; i < v.size(); ++i) D[i][i] = v[i];
    return D;
}

Matrix scale_columns(const Matrix& A, const std::vector<double>& inv_scales) {
    if (A.empty()) return {};
    Matrix B = A;
    for (size_t j = 0; j < inv_scales.size(); ++j) {
        for (size_t i = 0; i < A.size(); ++i) {
            B[i][j] = A[i][j] * inv_scales[j];
        }
    }
    return B;
}

static void jacobi_eigen(const Matrix& A, std::vector<double>& eigvals, Matrix& eigvecs) {
    const size_t n = A.size();
    Matrix B = A;
    eigvecs = identity(n);

    const int max_iters = 100;
    const double eps = 1e-12;

    for (int iter = 0; iter < max_iters; ++iter) {
        // Find largest off-diagonal element
        size_t p = 0, q = 1;
        double max_off = 0.0;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double val = std::fabs(B[i][j]);
                if (val > max_off) {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off < eps) break;

        double app = B[p][p];
        double aqq = B[q][q];
        double apq = B[p][q];

        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
        double c = 1.0 / std::sqrt(1.0 + t * t);
        double s = t * c;

        // Rotate rows/cols p and q
        for (size_t k = 0; k < n; ++k) {
            if (k != p && k != q) {
                double bkp = B[k][p];
                double bkq = B[k][q];
                B[k][p] = c * bkp - s * bkq;
                B[p][k] = B[k][p];
                B[k][q] = c * bkq + s * bkp;
                B[q][k] = B[k][q];
            }
        }

        double bpp = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        double bqq = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        B[p][p] = bpp;
        B[q][q] = bqq;
        B[p][q] = 0.0;
        B[q][p] = 0.0;

        // Update eigenvectors
        for (size_t k = 0; k < n; ++k) {
            double vkp = eigvecs[k][p];
            double vkq = eigvecs[k][q];
            eigvecs[k][p] = c * vkp - s * vkq;
            eigvecs[k][q] = s * vkp + c * vkq;
        }
    }

    eigvals.resize(n);
    for (size_t i = 0; i < n; ++i) eigvals[i] = B[i][i];
}

static void sort_eigs_desc(std::vector<double>& eigvals, Matrix& eigvecs) {
    const size_t n = eigvals.size();
    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return eigvals[a] > eigvals[b];
    });

    std::vector<double> sorted_vals(n);
    Matrix sorted_vecs(n, std::vector<double>(n, 0.0));
    for (size_t j = 0; j < n; ++j) {
        sorted_vals[j] = eigvals[idx[j]];
        for (size_t i = 0; i < n; ++i) {
            sorted_vecs[i][j] = eigvecs[i][idx[j]];
        }
    }
    eigvals = std::move(sorted_vals);
    eigvecs = std::move(sorted_vecs);
}

SVDResult svd(const Matrix& A) {
    SVDResult res;
    if (A.empty()) return res;

    const size_t m = A.size();
    const size_t n = A[0].size();

    Matrix At = transpose(A);
    Matrix AtA = multiply(At, A);

    std::vector<double> eigvals;
    Matrix V;
    jacobi_eigen(AtA, eigvals, V);
    sort_eigs_desc(eigvals, V);

    res.S.resize(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double v = eigvals[i];
        res.S[i] = (v > 0.0) ? std::sqrt(v) : 0.0;
    }

    // Compute U = A * V * S^{-1}
    Matrix AV = multiply(A, V);
    std::vector<double> inv_s(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        inv_s[i] = (res.S[i] > 1e-12) ? (1.0 / res.S[i]) : 0.0;
    }
    res.U = scale_columns(AV, inv_s);

    res.Vt = transpose(V);

    return res;
}

} // namespace svd14
