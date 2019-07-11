option(CUDA "Activates CUDA parallelization." OFF)

if (CUDA)
    set(CMAKE_CUDA_COMPILER_WORKS ON)
    message(STATUS "CUDA enabled.")
    enable_language(CUDA)
   # set(CUDA_SEPARABLE_COMPILATION ON)
   # set(CUDA_NVCC_FLAGS
   #         ${CUDA_NVCC_FLAGS};
   #         -std=c++11;
   #         -arch=sm_60;
   #         -gencode=arch=compute_60,code=sm_60)

    # packages
    #find_package(CUDA REQUIRED)

    # nvcc flags
    #set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};#-gencode arch=compute_20,code=sm_20
    #        )
    add_definitions(-DENABLE_CUDA)
else()
    message(STATUS "CUDA disabled.")
endif()
