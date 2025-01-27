################################### START VECTORIZATION ###################################
option(USE_VECTORIZATION "Enable generations of SIMD vector instructions through omp-simd" ON)
if (USE_VECTORIZATION)
    MESSAGE(STATUS "vectorization enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_VECTORIZED_CODE=1")
    # list of available options
    set(VECTOR_INSTRUCTIONS_OPTIONS "NATIVE;SSE;AVX;AVX2;KNL")
    # set instruction set type
    set(VECTOR_INSTRUCTIONS "NATIVE" CACHE STRING "Vector instruction set to use\
 (${VECTOR_INSTRUCTIONS_OPTIONS}).")
    # let ccmake and cmake-gui offer the options
    set_property(CACHE VECTOR_INSTRUCTIONS PROPERTY STRINGS ${VECTOR_INSTRUCTIONS_OPTIONS})

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp-simd")

        if (VECTOR_INSTRUCTIONS MATCHES "^NATIVE$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^SSE$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^AVX$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^AVX2$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^KNL$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=knl")
        else ()
            message(SEND_ERROR "\"${VECTOR_INSTRUCTIONS}\" is an unknown vector instruction set option.\
     Available options: ${VECTOR_INSTRUCTIONS_OPTIONS}")
        endif ()
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopenmp-simd")

        if (VECTOR_INSTRUCTIONS MATCHES "^NATIVE$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^SSE$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^AVX$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^AVX2$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2 -fma")
        elseif (VECTOR_INSTRUCTIONS MATCHES "^KNL$")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xMIC-AVX512")
        else ()
            message(SEND_ERROR "\"${VECTOR_INSTRUCTIONS}\" is an unknown vector instruction set option.\
     Available options: ${VECTOR_INSTRUCTIONS_OPTIONS}")
        endif ()
    else()
        message(WARNING "vectorization not yet supported on this compiler")
        message(STATUS "you can enable vectorization support by editing cmake/modules/vectorization.cmake")
    endif ()
elseif ()
    MESSAGE(STATUS "vectorization disabled")
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-tree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -no-vec")
    endif ()
endif ()
###################################  END VECTORIZATION  ###################################