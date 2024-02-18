nvcc -std=c++11 demo.cu -I /data0/home/wjl/software/TensorRT-8.4.3.1/include \
                        -I /data0/home/wjl/software/cuda-11.0/include \
                        -L /data0/home/wjl/software/cuda-11.0/lib64 -lcudart \
                        -L /data0/home/wjl/software/cuda-11.0/lib64 -lcudnn \
                        -L /data0/home/wjl/software/TensorRT-8.4.3.1/lib/ -lnvinfer
