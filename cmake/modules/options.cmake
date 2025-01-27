

# ---- PRECISION ----
# list of available options
set(PRECISION_OPTIONS "DOUBLE;SINGLE;MIXED")
# set instruction set type
set(PRECISION "DOUBLE" CACHE STRING "Precision to use (${PRECISION_OPTIONS}).")
# let ccmake and cmake-gui offer the options
set_property(CACHE PRECISION PROPERTY STRINGS ${PRECISION_OPTIONS})
if (PRECISION MATCHES "^DOUBLE$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMARDYN_DPDP")
elseif(PRECISION MATCHES "^SINGLE$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMARDYN_SPSP")
elseif(PRECISION MATCHES "^MIXED$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMARDYN_SPDP")
else()
    message(FATAL_ERROR "wrong precision option ")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTIMERS")

