file(GLOB_RECURSE MY_SRC
        "*.cpp"
        "*.h"
        "*.cu"
        )
# if mpi is not enabled, remove the uneeded source files
#if(NOT ENABLE_MPI)
#    # duplicate the list
#    set(MY_SRC_BACK ${MY_SRC})
#    # exclude everything from parallel
#    list(FILTER MY_SRC EXCLUDE REGEX "/parallel/")
#
#    # but include DomainDecompBase* and LoadCalc*
#    list(FILTER MY_SRC_BACK INCLUDE REGEX "/parallel/")
#    list(FILTER MY_SRC_BACK INCLUDE REGEX "DomainDecompBase|LoadCalc")
#    list(APPEND MY_SRC ${MY_SRC_BACK})
#endif()

# if vtk is disabled, remove everything in a vtk directory
if(NOT ENABLE_VTK)
 #   set (EXCLUDE_DIR "/VTK/")
 #   foreach (TMP_PATH ${MY_SRC})
 #       string (FIND ${TMP_PATH} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
 #       if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
 #           list (REMOVE_ITEM MY_SRC ${TMP_PATH})
 #       endif ()
 #   endforeach(TMP_PATH)
  list(FILTER MY_SRC EXCLUDE REGEX "/VTK/") # this requires cmake > 3.6
endif()
if(NOT CUDA)
  list(FILTER MY_SRC EXCLUDE REGEX ".*\\.cu") # this requires cmake > 3.6
else()
    foreach(FILE ${MY_SRC})
        get_filename_component(FILE_EXT ${FILE} LAST_EXT)

        if(FILE_EXT STREQUAL ".cpp")
                get_filename_component(FILE_NAME ${FILE} NAME_WLE)
                get_filename_component(FILE_DIR ${FILE} DIRECTORY)
                if ("${FILE_DIR}/${FILE_NAME}.cu" IN_LIST MY_SRC)
                    list(REMOVE_ITEM MY_SRC "${FILE}")
                    message(STATUS "Removed ${FILE} in favor of ${FILE_DIR}/${FILE_NAME}.cu")
                endif()
            endif()

#        message(STATUS "${FILE}")
#        set_source_files_properties( ${FILE} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
    endforeach()
    # add the executable
    #CUDA_ADD_EXECUTABLE(GraPaSim
     #       ${MY_SRC}
     #       )
endif()

# add the executable
ADD_EXECUTABLE(GraPaSim
        ${MY_SRC}
        )
# we just add all libraries here. If a library is not set, it will simply be ignored.
TARGET_LINK_LIBRARIES(GraPaSim
        ${VTK_LIB}     # for VTK/xerces
        )

if(CUDA)
    set_target_properties(GraPaSim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(GraPaSim PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++11 -arch=sm_60 -gencode arch=compute_60,code=sm_60>)
endif()