option(CUDA "Activates CUDA parallelization." OFF)

if (CUDA)
    message(STATUS "CUDA enabled.")
    #set(CUDA_NVCC_FLAGS
    #        ${CUDA_NVCC_FLAGS};
    #        -std=c++11;
    #        -gencode)
    # enable_language(CUDA)

    # packages
    find_package(CUDA REQUIRED)

    # nvcc flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};#-gencode arch=compute_20,code=sm_20
            )
    add_definitions(-DENABLE_CUDA)
else()
    message(STATUS "CUDA disabled.")
endif()
