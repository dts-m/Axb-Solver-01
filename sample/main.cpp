#include "svd.h"
#include <iostream>
#include <iomanip>

using svd14::Matrix;

static void print_matrix(const Matrix& A, const std::string& name) {
    std::cout << name << ":\n";
    for (const auto& row : A) {
        for (double v : row) {
            std::cout << std::setw(10) << std::setprecision(6) << std::fixed << v << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    Matrix A = {
        {3.0, 1.0, 1.0},
        {-1.0, 3.0, 1.0}
    };

    auto res = svd14::svd(A);

    print_matrix(A, "A");
    print_matrix(res.U, "U");
    std::cout << "S:\n";
    for (double s : res.S) {
        std::cout << "  " << s << "\n";
    }
    print_matrix(res.Vt, "Vt");

    return 0;
}
