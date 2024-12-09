cmake_minimum_required(VERSION 3.10)
project(SNIC_Superpixels C)

set(CMAKE_C_STANDARD 11)
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}'")
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Og -Wall -Wextra -Wpedantic -fsanitize=address -fsanitize=undefined")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -flto -ffast-math")
endif()

add_executable(snic_test
        snic.c
        snic_test.c
)

target_link_libraries(snic_test m)

enable_testing()
add_test(NAME snic_basic_test COMMAND snic_test)