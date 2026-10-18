cmake_minimum_required(VERSION 3.13)
project(cuad CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake at the LLVM/Clang 14 install if not already on the prefix path.
if(NOT DEFINED CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "/usr/lib/llvm-14")
endif()

find_package(LLVM 14 REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

message(STATUS "cuad: using LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_INSTALL_PREFIX}")

add_executable(cuad src/main.cpp)
target_include_directories(cuad SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

separate_arguments(_llvm_defs NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(cuad PRIVATE ${_llvm_defs})

# LLVM/Clang are built without RTTI; match to avoid link/ABI surprises.
target_compile_options(cuad PRIVATE -fno-rtti)

# Prefer the single shared libclang-cpp + libLLVM (Ubuntu packaging).
# If those imported targets are missing, fall back to the component libs.
if(TARGET clang-cpp AND TARGET LLVM)
  target_link_libraries(cuad PRIVATE clang-cpp LLVM)
else()
  target_link_libraries(cuad PRIVATE
    clangTooling clangFrontend clangDriver clangSerialization clangParse
    clangSema clangAnalysis clangEdit clangASTMatchers clangAST clangLex
    clangBasic)
  llvm_map_components_to_libnames(_llvm_libs support core option frontendopenmp)
  target_link_libraries(cuad PRIVATE ${_llvm_libs})
endif()
