CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(SVMbenchmark)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCL REQUIRED)
FIND_PACKAGE(Boost REQUIRED) #needed for gtSVM

SET(cuSVM_src
    cuSVM/cuSVMSolver.cu
    cuSVM/cuSVMutil.h
    cuSVM/mex.h
    cuSVM/cuSVM_wrapper.cpp
    cuSVM/cuSVM_wrapper.h
)

SET(GPULibSVM_src
    #GPULibSVM/cross_validation_with_matrix_precomputation.cpp
    GPULibSVM/gpulibsvm.cpp
    GPULibSVM/gpulibsvm.h
    #GPULibSVM/kernel_matrix_calculation.cpp
    GPULibSVM/GPUlibSVM_wrapper.cpp
    GPULibSVM/GPUlibSVM_wrapper.h
)

SET(gpuSVM_src
    gpuSVM/cutil.h
    gpuSVM/common/deviceSelect.cu
    gpuSVM/common/deviceSelect.h
    gpuSVM/common/framework.h
    gpuSVM/common/svmIO.cpp
    gpuSVM/common/svmIO.h
    gpuSVM/include/svmCommon.h
    gpuSVM/include/svmTrain.h
    gpuSVM/training/Cache.cpp
    gpuSVM/training/Cache.h
    gpuSVM/training/Controller.cpp
    gpuSVM/training/Controller.h
    gpuSVM/training/firstOrder.h
    gpuSVM/training/initialize.h
    gpuSVM/training/kernelType.h
    gpuSVM/training/memoryRoutines.h
    gpuSVM/training/reduce.h
    gpuSVM/training/reductionOperators.h
    gpuSVM/training/secondOrder.h
    gpuSVM/training/svmKernels.h
    gpuSVM/training/svmTrain.cu
    gpuSVM/gpuSVM_wrapper.cpp
    gpuSVM/gpuSVM_wrapper.h
)

SET(gtSVM_src
    gtSVM/bin/auto_context.hpp
    gtSVM/lib/cuda.hpp
    gtSVM/lib/cuda_array.cu
    gtSVM/lib/cuda_array.hpp
    gtSVM/lib/cuda_exception.cpp
    gtSVM/lib/cuda_exception.hpp
    gtSVM/lib/cuda_find_largest.cu
    gtSVM/lib/cuda_find_largest.hpp
    gtSVM/lib/cuda_helpers.hpp
    gtSVM/lib/cuda_partial_sum.cu
    gtSVM/lib/cuda_partial_sum.hpp
    gtSVM/lib/cuda_reduce.cu
    gtSVM/lib/cuda_reduce.hpp
    gtSVM/lib/cuda_sparse_kernel.cu
    gtSVM/lib/cuda_sparse_kernel.hpp
    gtSVM/lib/gtsvm.cpp
    gtSVM/lib/gtsvm.h
    gtSVM/lib/headers.hpp
    gtSVM/lib/helpers.hpp
    gtSVM/lib/svm_gt.cpp
    gtSVM/lib/svm_gt.hpp
    gtSVM/gtSVM_wrapper.cpp
    gtSVM/gtSVM_wrapper.h
)

SET(libSVM_src
    libSVM/libSVM_utils.cpp
    libSVM/libSVM_utils.h
    libSVM/svm.cpp
    libSVM/svm.h
    libSVM/libSVM_wrapper.cpp
    libSVM/libSVM_wrapper.h
)

SET(multiSVM_src
    multiSVM/multisvm.cu
    multiSVM/multiSVM_wrapper.cpp
    multiSVM/multiSVM_wrapper.h
)

SET(orcusSVM_src
    orcusSVM/cudaerror.cuh
    orcusSVM/debug.cpp
    orcusSVM/debug.h
    orcusSVM/OrcusSvm.cu
    orcusSVM/OrcusSvm.h
    orcusSVM/orcusSVM_wrapper.cpp
    orcusSVM/orcusSVM_wrapper.h
)

SET(openclSVM_src
    openclSVM/OpenCLSvm.cpp
    openclSVM/OpenCLSvm.h
    openclSVM/simpleCL.cpp
    openclSVM/simpleCL.h
    openclSVM/openclSVM_wrapper.cpp
    openclSVM/openclSVM_wrapper.h
)

SET(src
    cuda_utils.h
    stopwatch.h
    svm_template.cpp
    svm_template.h
    svm-train.cpp
    utils.cpp
    utils.h
)

FIND_PATH(CUDA_COMMON_INCLUDE_DIRS
    helper_cuda.h
    PATHS ${CUDA_SDK_SEARCH_PATH}
    "/usr/local/cuda"
    "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/common/inc"
    DOC "Location of helper_cuda.h"
    NO_DEFAULT_PATH)

INCLUDE_DIRECTORIES(
    ${CUDA_COMMON_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    .
    libSVM
    GPULibSVM
    cuSVM
    gpuSVM
    gtSVM
    multiSVM
    orcusSVM
    multiSVM/src
    multiSVM/include
    gtSVM/bin
    gtSVM/lib
    gpuSVM/common
    gpuSVM/include
    gpuSVM/training
)

CUDA_ADD_EXECUTABLE(SVMbenchmark ${src} ${cuSVM_src} ${GPULibSVM_src} ${gpuSVM_src} ${gtSVM_src} ${libSVM_src} ${multiSVM_src} ${orcusSVM_src} ${openclSVM_src})
CUDA_ADD_CUBLAS_TO_TARGET(SVMbenchmark)
TARGET_LINK_LIBRARIES(SVMbenchmark ${CUDA_CUDA_LIBRARY} ${OpenCL_LIBRARY})