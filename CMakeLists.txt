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

# --- Testing & integration (Gerson) ------------------------------------------
# compile_commands.json for editors/clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -DQLINEAR_SANITIZE=ON builds every target in this directory with
# AddressSanitizer + UndefinedBehaviorSanitizer (CMAKE_CXX_FLAGS is applied at
# generate time, so Darius's and Jenny's targets are covered as well).
option(QLINEAR_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
if(QLINEAR_SANITIZE)
    string(APPEND CMAKE_CXX_FLAGS
        " -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -fsanitize=address,undefined")
endif()

add_library(qlinear_app
    src/DataGenerator.cpp
    src/MatrixIO.cpp
    src/InputValidator.cpp
    src/ReferenceKernel.cpp
    src/LinearOperator.cpp
    src/Benchmark.cpp)
target_include_directories(qlinear_app PUBLIC include)
target_link_libraries(qlinear_app PUBLIC quantizer)
project_warnings(qlinear_app)
# Recorded in every benchmark CSV row.
string(TOUPPER "${CMAKE_BUILD_TYPE}" qlinear_build_type)
set(qlinear_build_flags
    "${CMAKE_BUILD_TYPE}: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${qlinear_build_type}}")
string(STRIP "${qlinear_build_flags}" qlinear_build_flags)
target_compile_definitions(qlinear_app PRIVATE QLINEAR_BUILD_FLAGS="${qlinear_build_flags}")

add_executable(qlinear src/main.cpp)
target_link_libraries(qlinear PRIVATE qlinear_app)
project_warnings(qlinear)

if(BUILD_TESTING)
    foreach(test_name data_generator matrix_io input_validator reference_kernel integration)
        add_executable(${test_name}_tests tests/${test_name}_tests.cpp)
        target_include_directories(${test_name}_tests PRIVATE tests)
        target_link_libraries(${test_name}_tests PRIVATE qlinear_app)
        target_compile_definitions(${test_name}_tests PRIVATE
            QLINEAR_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
        project_warnings(${test_name}_tests)
        add_test(NAME ${test_name} COMMAND ${test_name}_tests)
    endforeach()

    # End-to-end CLI checks: exit code plus a regex on stdout+stderr.
    set(qlinear_data "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
    function(qlinear_cli_test name expect_exit regex)
        list(JOIN ARGN "|" joined_args)
        add_test(NAME cli_${name}
            COMMAND ${CMAKE_COMMAND}
                -DQLINEAR=$<TARGET_FILE:qlinear>
                "-DARGS=${joined_args}"
                -DEXPECT_EXIT=${expect_exit}
                "-DEXPECT_REGEX=${regex}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_expect.cmake)
    endfunction()

    qlinear_cli_test(help 0 "usage:" help)
    qlinear_cli_test(no_args 2 "usage:")
    qlinear_cli_test(unknown_command 2 "unknown command" frobnicate)
    qlinear_cli_test(golden_reference 0 "-1\\.5 +-1\\.5.*relative_fro 0"
        run --x ${qlinear_data}/golden_x.txt --w ${qlinear_data}/golden_w.txt
            --kernel reference)
    qlinear_cli_test(missing_w 2 "missing required option --w"
        run --x ${qlinear_data}/golden_x.txt)
    qlinear_cli_test(missing_file 1 "cannot open file"
        run --x ${qlinear_data}/does_not_exist.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(empty_file 1 "empty input"
        run --x ${qlinear_data}/empty.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(bad_header 1 "line 2: header: cols must be a positive integer"
        run --x ${qlinear_data}/bad_header.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(non_numeric 1 "line 2: expected a number, got 'abc'"
        run --x ${qlinear_data}/non_numeric.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(nan_value 1 "line 2: NaN is not allowed"
        run --x ${qlinear_data}/nan_value.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(inf_value 1 "line 2: infinity is not allowed"
        run --x ${qlinear_data}/inf_value.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(out_of_range 1 "outside the float32 range"
        run --x ${qlinear_data}/out_of_range.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(truncated 1 "expected 4 values .* ends after 3"
        run --x ${qlinear_data}/truncated.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(extra_values 1 "line 2: unexpected extra value '3'"
        run --x ${qlinear_data}/extra_values.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(zero_dims 1 "rows must be at least 1"
        run --x ${qlinear_data}/zero_dims.txt --w ${qlinear_data}/golden_w.txt)
    qlinear_cli_test(mismatched_inner 1 "inner dimensions do not match"
        run --x ${qlinear_data}/golden_x.txt --w ${qlinear_data}/mismatched_w.txt)
    qlinear_cli_test(block_zero 1 "--block must be at least 1"
        run --x ${qlinear_data}/golden_x.txt --w ${qlinear_data}/golden_w.txt --block 0)
    qlinear_cli_test(unknown_kernel 1 "unknown operator 'fast'"
        run --x ${qlinear_data}/golden_x.txt --w ${qlinear_data}/golden_w.txt --kernel fast)
    qlinear_cli_test(gen_unknown_dist 1 "unknown distribution 'gamma'"
        gen --rows 2 --cols 2 --dist gamma --out -)
    qlinear_cli_test(bench_quick 0 "seed,dist,shape,m,k,n,operator"
        bench --config quick --sizes 8,16 --reps 1 --out -)
endif()
