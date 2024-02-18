cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(tensor_demo LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} tensor_demo1.cu)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)