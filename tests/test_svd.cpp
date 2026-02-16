#include "svd.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <string>

using svd14::Matrix;

static Matrix diag_from_vector(const std::vector<double>& v) {
    return svd14::diag_from_vector(v);
}

static Matrix multiply(const Matrix& A, const Matrix& B) {
    return svd14::multiply(A, B);
}

static Matrix transpose(const Matrix& A) {
    return svd14::transpose(A);
}

static bool near(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

static bool matrix_near(const Matrix& A, const Matrix& B, double eps = 1e-6) {
    if (A.size() != B.size() || (A.empty() ? 0 : A[0].size()) != (B.empty() ? 0 : B[0].size())) {
        return false;
    }
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            if (!near(A[i][j], B[i][j], eps)) return false;
        }
    }
    return true;
}

static Matrix eye(size_t n) {
    Matrix I(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

static void expect(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "FAILED: " << msg << "\n";
        std::exit(1);
    }
}

int main() {
    // Test 1: Identity matrix
    Matrix I = {
        {1.0, 0.0},
        {0.0, 1.0}
    };
    auto r1 = svd14::svd(I);
    Matrix S1 = diag_from_vector(r1.S);
    Matrix recon1 = multiply(multiply(r1.U, S1), r1.Vt);
    expect(matrix_near(I, recon1), "Reconstruction of identity");

    // Test 2: Rectangular matrix
    Matrix A = {
        {3.0, 1.0, 1.0},
        {-1.0, 3.0, 1.0}
    };
    auto r2 = svd14::svd(A);
    Matrix S2 = diag_from_vector(r2.S);
    Matrix recon2 = multiply(multiply(r2.U, S2), r2.Vt);
    expect(matrix_near(A, recon2, 1e-5), "Reconstruction of rectangular matrix");

    // Test 3: Orthogonality of V (V * Vt = I)
    Matrix V = transpose(r2.Vt);
    Matrix VVt = multiply(V, r2.Vt);
    expect(matrix_near(VVt, eye(VVt.size()), 1e-5), "V orthogonality");

    std::cout << "All tests passed.\n";
    return 0;
}
