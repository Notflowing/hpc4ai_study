# g++ -O3 -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

# g++ -std=c++14 tensor_demo1.cpp -I/data0/home/wjl/.conda/envs/pytorch/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/data0/home/wjl/.conda/envs/pytorch/include/python3.9 -I/data0/home/wjl/.conda/envs/pytorch/lib/python3.9/site-packages/torch/include -o tensor_demo1

cd build

cmake -DCMAKE_PREFIX_PATH=`python3 -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build .

./tensor_demo