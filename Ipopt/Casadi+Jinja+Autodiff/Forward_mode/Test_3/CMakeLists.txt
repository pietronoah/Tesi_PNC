cmake_minimum_required(VERSION 3.22.0)
project(test_1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -pipe -DNDEBUG -Wparentheses -Wreturn-type -Wcast-qual -Wall -Wpointer-arith -Wwrite-strings -Wconversion -Wno-unknown-pragmas -Wno-long-long   -DIPOPT_BUILD")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(IPOPT_LIBRARY_DIRS /opt/homebrew/Cellar/ipopt/3.14.4/lib)
set(IPOPT_INCLUDE_DIRS /opt/homebrew/Cellar/ipopt/3.14.4/include/coin-or)

set(AUTODIFF_LIBRARY_DIRS /opt/homebrew/Cellar/autodiff/0.6.4/lib)
set(AUTODIFF_INCLUDE_DIRS /opt/homebrew/Cellar/autodiff/0.6.4/include/)

set(EIGEN_INCLUDE_DIRS /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/)


message("ipopt binary path is: " ${IPOPT_LIBRARY_DIRS} )
message("ipopt include path is: " ${IPOPT_INCLUDE_DIRS} )

include_directories(${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
link_directories(${IPOPT_LIBRARY_DIRS})

include_directories(${AUTODIFF_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
link_directories(${AUTODIFF_LIBRARY_DIRS})

include_directories(${EIGEN_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)


add_library(ipopt_interface test_1_nlp.cpp)

add_executable(${PROJECT_NAME} test_1_main.cpp)


target_link_libraries(${PROJECT_NAME} ipopt_interface)
target_link_libraries(${PROJECT_NAME} ipopt)