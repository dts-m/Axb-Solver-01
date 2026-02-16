# Self‑Contained SVD (C++14)

This project provides a **minimal, self‑contained Singular Value Decomposition (SVD)** implementation written in C++14 with **no external dependencies**.

## What is SVD?

Given a real matrix **A (m×n)**, the Singular Value Decomposition expresses it as:

```
A = U * S * Vᵀ
```

- **U** is an orthonormal matrix (m×n for thin SVD),
- **S** is a diagonal matrix of non‑negative singular values (n),
- **Vᵀ** is the transpose of an orthonormal matrix (n×n).

SVD is used in:
- dimensionality reduction,
- least‑squares solving,
- numerical stability analysis,
- compression.

## Implementation Notes

This implementation computes SVD by:

1. Building **AᵀA** (symmetric).
2. Performing **Jacobi eigenvalue decomposition** on AᵀA to obtain **V** and eigenvalues.
3. Singular values are √(eigenvalues).
4. **U = A * V * S⁻¹** (thin SVD).

**Limitations / Notes**
- Works best for **m ≥ n**.
- Produces a **thin SVD**: U is m×n, S has n values, Vᵀ is n×n.
- For near‑zero singular values, corresponding U columns are set to zero.

## Build (Visual Studio 2017)

```bash
mkdir build
cd build
cmake -G "Visual Studio 15 2017" ..
cmake --build . --config Release
```

## Run Sample

```bash
./sample_svd
```

## Run Tests

```bash
./test_svd
```

## File Layout

```
include/svd.h
src/svd.cpp
sample/main.cpp
tests/test_svd.cpp
CMakeLists.txt
