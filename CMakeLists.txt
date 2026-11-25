cmake_minimum_required(VERSION 3.20)
project(EE538Int8Linear LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(CTest)

function(project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

# Each component adds its library, tests and tools below.

# --- Quantization (Darius) ---------------------------------------------------
add_library(quantizer src/Quantizer.cpp src/QuantError.cpp)
target_include_directories(quantizer PUBLIC include)
project_warnings(quantizer)

if(BUILD_TESTING)
    add_executable(quantizer_tests tests/quantizer_tests.cpp)
    target_include_directories(quantizer_tests PRIVATE tests)
    target_link_libraries(quantizer_tests PRIVATE quantizer)
    project_warnings(quantizer_tests)
    add_test(NAME quantizer COMMAND quantizer_tests)

    add_executable(quant_error_tests tests/quant_error_tests.cpp)
    target_include_directories(quant_error_tests PRIVATE tests)
    target_link_libraries(quant_error_tests PRIVATE quantizer)
    project_warnings(quant_error_tests)
    add_test(NAME quant_error COMMAND quant_error_tests)
endif()

add_executable(quant_error_analysis tools/quant_error_analysis.cpp)
target_link_libraries(quant_error_analysis PRIVATE quantizer)
project_warnings(quant_error_analysis)
