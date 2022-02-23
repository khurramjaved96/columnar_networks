#include <pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../include/environments/mnist/binary_mnist_loader.hpp"

namespace py = pybind11;

PYBIND11_MODULE(FlexibleNN, m) {
    py::class_<BinaryMnistLoader>(m, "BinaryMnistLoader")
        .def(py::init<int>())
        .def("get_data", &BinaryMnistLoader::get_data);
}
