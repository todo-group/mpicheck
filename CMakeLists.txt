cmake_minimum_required(VERSION 3.14)
include(cmake/prefix.cmake)
project(mpicheck CXX)
include(cmake/postfix.cmake)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

set(PROGS mpicheck)
foreach(name ${PROGS})
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} ${MPI_CXX_LIBRARIES})
endforeach(name)
