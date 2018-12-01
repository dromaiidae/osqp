cmake_minimum_required(VERSION 3.12)
project(cs294Solver C)

set(CMAKE_C_STANDARD 11)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

file(GLOB cs294Solver_src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        )

file(GLOB cs294Solver_headers
        ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
        )

add_library(cs294Solver SHARED ${cs294Solver_src} ${cs294Solver_headers})


file(GLOB cs294SolverExec_src
        ${CMAKE_CURRENT_SOURCE_DIR}/example/*.c
        )
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/data/matrix.mat ${EXECUTABLE_OUTPUT_PATH}/data/matrix.mat COPYONLY)

add_executable(cs294SolverExec ${cs294SolverExec_src})

