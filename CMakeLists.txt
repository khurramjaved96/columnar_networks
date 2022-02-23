#CC Build instructions
# Download and store the pybind11 source in pybind11/ directory
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c

cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CUDA CXX)
#enable_language(CUDA)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#GCloud
#set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")

#Cedar
message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )
set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")

add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)

pybind11_add_module(FlexibleNN include/environments/mnist/binary_mnist_loader.hpp pybind_interface/interface.cpp)

target_link_libraries(FlexibleNN PUBLIC tbb)
target_link_libraries(FlexibleNN PUBLIC mariadb)

project(FlexibleNN LANGUAGES CXX)

