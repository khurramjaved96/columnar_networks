#ifndef INCLUDE_MNIST_LOADER_HPP_
#define INCLUDE_MNIST_LOADER_HPP_

#include <vector>
#include <map>
#include <random>
#include <string>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <execution>

#include "mnist_reader.hpp"
#include "mnist_utils.hpp"

class BinaryMnistLoader{
  std::mt19937 mt;
  std::uniform_int_distribution<int> index_sampler;
  std::vector<std::vector<float>> images;
  std::vector<float> targets;
 public:
  BinaryMnistLoader(int seed);

  std::pair<std::vector<float>, float> get_data();
};


BinaryMnistLoader::BinaryMnistLoader(int seed) : mt(seed) {
  std::cout << "Loading dataset..." << std::endl;
  auto dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("data/");

  int total_data_points = 60000;
  this->index_sampler = std::uniform_int_distribution<int>(0, total_data_points - 1);

  for(int counter = 0; counter < total_data_points; counter++){
    std::vector<float> x_temp;
    for(auto inner: dataset.training_images[counter]){
      x_temp.push_back(float(unsigned(inner)));
    }
    float y_temp;
    if (int(unsigned(dataset.training_labels[counter])) % 2 == 0)
      y_temp = 1.0;
    else
      y_temp = 0.0;
    this->images.push_back(x_temp);
    this->targets.push_back(y_temp);
  }
}


std::pair<std::vector<float>, float> BinaryMnistLoader::get_data(){
  int index = this->index_sampler(this->mt);
  auto x = images[index];
  float y = targets[index];
  return std::make_pair(x, y);
}
#endif //INCLUDE_MNIST_LOADER_HPP_
