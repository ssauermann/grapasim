option(CUDA "Activates CUDA parallelization." OFF)

if (CUDA)
    message(STATUS "CUDA enabled.")
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)
else()
    message(STATUS "CUDA disabled.")
endif()
