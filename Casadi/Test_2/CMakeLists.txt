
cmake_minimum_required(VERSION 3.21.0)
project(First_test_with_casadi_implementation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -pipe -DNDEBUG -Wparentheses -Wreturn-type -Wcast-qual -Wall -Wpointer-arith -Wwrite-strings -Wconversion -Wno-unknown-pragmas -Wno-long-long   -DIPOPT_BUILD")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(IPOPT_LIBRARY_DIRS /opt/homebrew/Cellar/ipopt/3.14.4/lib)
set(IPOPT_INCLUDE_DIRS /opt/homebrew/Cellar/ipopt/3.14.4/include/coin-or)

message("ipopt binary path is: " ${IPOPT_LIBRARY_DIRS} )
message("ipopt include path is: " ${IPOPT_INCLUDE_DIRS} )

include_directories(${IPOPT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/include)
link_directories(${IPOPT_LIBRARY_DIRS})



add_library(ipopt_interface hs071_nlp.cpp pattern_functions.cpp pattern_functions.h)

add_executable(${PROJECT_NAME} hs071_main.cpp)


target_link_libraries(${PROJECT_NAME} ipopt_interface)
target_link_libraries(${PROJECT_NAME} ipopt)



 