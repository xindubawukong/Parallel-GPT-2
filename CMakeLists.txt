cmake_minimum_required(VERSION 3.15.0)
project(
  cs-228-project
  VERSION 0.1
  DESCRIPTION ""
  HOMEPAGE_URL ""
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Torch_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/parlaylib/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann-json/include
)
add_subdirectory(third_party/gflags)

add_compile_options(
  -Ofast
  -march=native
  -Wall -Wextra
  -pthread
)
add_link_options(
  -pthread
)

add_executable(gpt2 gpt2.cpp)
target_link_libraries(gpt2 PRIVATE ${TORCH_LIBRARIES} gflags::gflags)
