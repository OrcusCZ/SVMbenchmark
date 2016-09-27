CMAKE_MINIMUM_REQUIRED(VERSION 3.1 FATAL_ERROR)

FIND_PACKAGE(CUDA REQUIRED)

IF(NOT NO_BOOST)
    FIND_PACKAGE(Boost) #needed for gtSVM
    IF(NOT Boost_FOUND)
        MESSAGE("Boost not found, using workaround for gtSVM")
    ENDIF()
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCC)
    STRING(REGEX REPLACE ".*/" "" C ${CUDA_HOST_COMPILER})
    IF(${C} STREQUAL "ccache")
        MESSAGE(WARNING "CUDA_HOST_COMPILER is set to: ${CUDA_HOST_COMPILER}, changing to /usr/bin/g++")
        SET(CUDA_HOST_COMPILER "/usr/bin/g++")
    ENDIF()
ENDIF()

ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)

SET(gtSVM_src
    ../gtSVM/bin/auto_context.hpp
    ../gtSVM/lib/cuda.hpp
    ../gtSVM/lib/cuda_array.cu
    ../gtSVM/lib/cuda_array.hpp
    ../gtSVM/lib/cuda_exception.cpp
    ../gtSVM/lib/cuda_exception.hpp
    ../gtSVM/lib/cuda_find_largest.cu
    ../gtSVM/lib/cuda_find_largest.hpp
    ../gtSVM/lib/cuda_helpers.hpp
    ../gtSVM/lib/cuda_partial_sum.cu
    ../gtSVM/lib/cuda_partial_sum.hpp
    ../gtSVM/lib/cuda_reduce.cu
    ../gtSVM/lib/cuda_reduce.hpp
    ../gtSVM/lib/cuda_sparse_kernel.cu
    ../gtSVM/lib/cuda_sparse_kernel.hpp
    ../gtSVM/lib/gtsvm.cpp
    ../gtSVM/lib/gtsvm.h
    ../gtSVM/lib/headers.hpp
    ../gtSVM/lib/helpers.hpp
    ../gtSVM/lib/svm_gt.cpp
    ../gtSVM/lib/svm_gt.hpp
    ../gtSVM/boost_helpers.h #workaround so we don't need boost
    ../gtSVM/boost_headers.h #workaround so we don't need boost
    ../gtSVM/gtSVM_wrapper.cpp
    ../gtSVM/gtSVM_wrapper.h
) 

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

FOREACH(LIBDIR gtSVM)
    SVM_SOURCE_GROUP(${LIBDIR} ${${LIBDIR}_src})
ENDFOREACH()
#end of Visual Studio Source Group definitions

INCLUDE_DIRECTORIES(
    .
    ../gtSVM/bin
    ../gtSVM/lib
)

FUNCTION(ADD_IMP IMP SRCVAR)
    IF(COMPILE_WITH_${IMP})
        MESSAGE("Compiling with: ${IMP}")
        SET(${SRCVAR} ${${SRCVAR}} ${ARGN} PARENT_SCOPE)
        ADD_DEFINITIONS(-DCOMPILE_WITH_${IMP})
    ENDIF()
ENDFUNCTION()

ADD_IMP(GTSVM SRC ${gtSVM_src})

IF(Boost_FOUND)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS(-DUSE_BOOST)
ENDIF()

CUDA_ADD_EXECUTABLE(SVMbenchmark_gpl ${SRC})
TARGET_LINK_LIBRARIES(SVMbenchmark_gpl ${CUDA_CUDA_LIBRARY})
SET_PROPERTY(TARGET SVMbenchmark_gpl PROPERTY CXX_STANDARD 11)
