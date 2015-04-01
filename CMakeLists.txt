cmake_minimum_required(VERSION 2.8)

project( denoise )

find_package( OpenCV REQUIRED )
include_directories(include)

set(HEADER_LIB gc_min.hpp)

add_executable( test main.cpp ${HEADER_LIB} )

target_link_libraries( test ${OpenCV_LIBS} )

