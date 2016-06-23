CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)
PROJECT(SVMbenchmark)

SET(NO_CUDA "OFF" CACHE BOOL "Don't use CUDA")
SET(NO_BOOST "ON" CACHE BOOL "Compile without Boost even if it is available")

SET(COMPILE_WITH_LIBSVM ON CACHE BOOL "Compile with LibSVM")
SET(COMPILE_WITH_CUSVM ON CACHE BOOL "Compile with CuSVM")
SET(COMPILE_WITH_GPUSVM ON CACHE BOOL "Compile with GpuSVM")
SET(COMPILE_WITH_MULTISVM ON CACHE BOOL "Compile with MultiSVM")
SET(COMPILE_WITH_GTSVM ON CACHE BOOL "Compile with GTSVM")
SET(COMPILE_WITH_WUSVM ON CACHE BOOL "Compile with WuSVM")

IF(NOT NO_CUDA)
    FIND_PACKAGE(CUDA)
    IF(NOT CUDA_FOUND)
        MESSAGE(WARNING "CUDA not found, CUDA SVM implementations disabled")
    ELSE()
        LIST(APPEND CUDA_NVCC_FLAGS "-lineinfo;--ptxas-options=-v")
    ENDIF()
ENDIF()

SET(USE_CUDA_DP OFF)

IF(NO_CUDA OR NOT CUDA_FOUND)
    SET(COMPILE_WITH_CUSVM OFF)
    SET(COMPILE_WITH_GPUSVM OFF)
    SET(COMPILE_WITH_MULTISVM OFF)
    SET(COMPILE_WITH_GTSVM OFF)
ELSE()
    ADD_DEFINITIONS(-DWUCUDA) #wuSVM
    IF(USE_CUDA_DP)
        LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--library=cublas_device")
        SET(CUDA_SEPARABLE_COMPILATION ON)
    ELSE()
        LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")
    ENDIF()
ENDIF()

IF(COMPILE_WITH_GTSVM)
    IF(NOT NO_BOOST)
        FIND_PACKAGE(Boost) #needed for gtSVM
        IF(NOT Boost_FOUND)
            MESSAGE("Boost not found, using workaround for gtSVM")
        ENDIF()
    ENDIF()
ENDIF()
IF(CMAKE_COMPILER_IS_GNUCC)
    STRING(REGEX REPLACE ".*/" "" C ${CUDA_HOST_COMPILER})
    IF(${C} STREQUAL "ccache")
        MESSAGE(WARNING "CUDA_HOST_COMPILER is set to: ${CUDA_HOST_COMPILER}, changing to /usr/bin/g++")
        SET(CUDA_HOST_COMPILER "/usr/bin/g++")
    ENDIF()
ENDIF()


IF(COMPILE_WITH_WUSVM)
#module add intelcdk-12
    SET (MKL_MY_LIB_FILES
        mkl_intel_lp64
        mkl_core
        mkl_sequential
    )
    SET (IPP_MY_LIB_FILES
        ipps
    )

    FIND_PACKAGE(OpenMP)
    IF (OPENMP_FOUND)
        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ENDIF()
ENDIF()

ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)

SET(cuSVM_src
    cuSVM/cuSVMSolver.cu
    cuSVM/cuSVMutil.h
    cuSVM/mex.h
    cuSVM/cuSVM_wrapper.cpp
    cuSVM/cuSVM_wrapper.h
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
    gtSVM/boost_helpers.h #workaround so we don't need boost
    gtSVM/boost_headers.h #workaround so we don't need boost
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

SET(wuSVM_src
    wuSVM/wuSVM_wrapper.h
    wuSVM/wuSVM_wrapper.cpp
    wuSVM/abstract_matrix.h
    wuSVM/blas_wrappers.h
    wuSVM/Config.h
    wuSVM/convert.h
    wuSVM/device_context.h
    wuSVM/device_wrappers.cu
    wuSVM/fileIO.h
    wuSVM/fileIO.cpp
    wuSVM/hessian.h
    wuSVM/hessian.cpp
    wuSVM/host_wrappers.cpp    
    wuSVM/kernel_mult.cpp
    wuSVM/kernels.h
    wuSVM/kernels.cpp
    wuSVM/lasp_matrix.h
    wuSVM/lasp_svm.h
    wuSVM/lasp_svm.cpp
    wuSVM/next_point.h
    wuSVM/next_point.cpp
    wuSVM/options.h
    wuSVM/parsing.h
    wuSVM/parsing.cpp
    wuSVM/pegasos.h
    wuSVM/pegasos.cpp
    wuSVM/retraining.h
    wuSVM/retraining.cpp
    wuSVM/train_subset.h
    wuSVM/train_subset.cpp
)
ADD_DEFINITIONS(-DCPP11) #wusvm    

SET(SRC
    debug.h
    debug.cpp
    cudaerror.h
    cuda_utils.h
    my_stopwatch.h
    svm_template.cpp
    svm_template.h
    svm-train.cpp
    utils.cpp
    utils.h
)

#beginning of Visual Studio Source Group definitions
FUNCTION(FILTER_LIST RESULT FILTER)
    SET(RES)
    FOREACH(IT ${ARGN})
        IF(IT MATCHES ${FILTER})
            LIST(APPEND RES ${IT})
        ENDIF()
    ENDFOREACH()
    SET(${RESULT} ${RES} PARENT_SCOPE)
ENDFUNCTION()

FUNCTION(SVM_SOURCE_GROUP LIBNAME)
    FILTER_LIST(HDR "\\.(h|hpp)$" ${ARGN})
    FILTER_LIST(SRC "\\.(cpp|cu)$" ${ARGN})
    SOURCE_GROUP("${LIBNAME} Header Files" FILES ${HDR})
    SOURCE_GROUP("${LIBNAME} Source Files" FILES ${SRC})
ENDFUNCTION()

FOREACH(LIBDIR cuSVM gpuSVM gtSVM libSVM multiSVM wuSVM)
    SVM_SOURCE_GROUP(${LIBDIR} ${${LIBDIR}_src})
ENDFOREACH()
#end of Visual Studio Source Group definitions

FIND_PATH(CUDA_COMMON_INCLUDE_DIRS
    helper_cuda.h
    PATHS ${CUDA_SDK_SEARCH_PATH}
    "/usr/local/cuda-${CUDA_VERSION}/samples/common/inc"
    "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v${CUDA_VERSION}/common/inc"
    DOC "Location of helper_cuda.h"
    NO_DEFAULT_PATH)

INCLUDE_DIRECTORIES(
    ${CUDA_COMMON_INCLUDE_DIRS}
    ${OpenCL_INCLUDE_DIR}
    $ENV{MKLROOT}/include
    $ENV{IPPROOT}/include     
    .
    libSVM
    cuSVM
    gpuSVM
    gtSVM
    multiSVM
    multiSVM/src
    multiSVM/include
    gtSVM/bin
    gtSVM/lib
    gpuSVM/common
    gpuSVM/include
    gpuSVM/training
    wuSVM
)

FUNCTION(ADD_IMP IMP SRCVAR)
    IF(COMPILE_WITH_${IMP})
        MESSAGE("Compiling with: ${IMP}")
        SET(${SRCVAR} ${${SRCVAR}} ${ARGN} PARENT_SCOPE)
        ADD_DEFINITIONS(-DCOMPILE_WITH_${IMP})
    ENDIF()
ENDFUNCTION()

ADD_IMP(LIBSVM SRC ${libSVM_src})
ADD_IMP(CUSVM SRC ${cuSVM_src})
ADD_IMP(GPUSVM SRC ${gpuSVM_src})
ADD_IMP(MULTISVM SRC ${multiSVM_src})
ADD_IMP(GTSVM SRC ${gtSVM_src})
ADD_IMP(WUSVM SRC ${wuSVM_src})

IF(COMPILE_WITH_WUSVM)
    LINK_DIRECTORIES (
        $ENV{MKLROOT}/lib/intel64
        $ENV{IPPROOT}/lib/intel64
    )
ENDIF()

IF(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_BOOST)
ENDIF()

CUDA_ADD_EXECUTABLE(SVMbenchmark ${SRC})
CUDA_ADD_CUBLAS_TO_TARGET(SVMbenchmark)
TARGET_LINK_LIBRARIES(SVMbenchmark ${CUDA_CUDA_LIBRARY})
IF(USE_CUDA_DP)
    GET_FILENAME_COMPONENT(CUDADEVRTPATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
    TARGET_LINK_LIBRARIES(SVMbenchmark "${CUDADEVRTPATH}/libcudadevrt.a" "${CUDADEVRTPATH}/libcublas_device.a")
ENDIF()
IF(COMPILE_WITH_WUSVM)
    TARGET_LINK_LIBRARIES(SVMbenchmark ${MKL_MY_LIB_FILES} ${IPP_MY_LIB_FILES})
ENDIF()
SET_PROPERTY(TARGET SVMbenchmark PROPERTY CXX_STANDARD 11)
