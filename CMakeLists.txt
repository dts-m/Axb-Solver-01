cmake_minimum_required(VERSION 3.8)
project(svd14 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(svd14
    src/svd.cpp
)
target_include_directories(svd14 PUBLIC include)

add_executable(sample_svd
    sample/main.cpp
)
target_link_libraries(sample_svd svd14)

add_executable(test_svd
    tests/test_svd.cpp
)
target_link_libraries(test_svd svd14)
