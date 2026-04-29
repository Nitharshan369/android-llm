cmake_minimum_required(VERSION 3.22)

project("firstapplication")

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(llama.cpp)

add_library(firstapplication SHARED native-lib.cpp)

find_library(log-lib log)

target_link_libraries(
    firstapplication
    llama
    ${log-lib}
)