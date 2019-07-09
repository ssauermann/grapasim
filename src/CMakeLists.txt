file(GLOB_RECURSE MY_SRC
        "*.cpp"
        "*.h"
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

# add the executable
ADD_EXECUTABLE(GraPaSim
        ${MY_SRC}
        )

# we just add all libraries here. If a library is not set, it will simply be ignored.
TARGET_LINK_LIBRARIES(GraPaSim
        ${VTK_LIB}     # for VTK/xerces
        )
