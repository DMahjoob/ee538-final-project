# Runs the qlinear CLI and checks its exit code and output.
#
#   cmake -DQLINEAR=<exe> -DARGS="a|b|c" -DEXPECT_EXIT=<code>
#         [-DEXPECT_REGEX=<regex matched against stdout+stderr>] -P cli_expect.cmake
#
# Arguments are joined with '|' because ';' is CMake's list separator.

string(REPLACE "|" ";" cli_args "${ARGS}")
execute_process(
    COMMAND "${QLINEAR}" ${cli_args}
    RESULT_VARIABLE exit_code
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)

set(all_output "${out}${err}")
if(NOT exit_code STREQUAL EXPECT_EXIT)
    message(FATAL_ERROR
        "expected exit code ${EXPECT_EXIT}, got ${exit_code}\n--- output ---\n${all_output}")
endif()
if(DEFINED EXPECT_REGEX AND NOT all_output MATCHES "${EXPECT_REGEX}")
    message(FATAL_ERROR
        "output does not match '${EXPECT_REGEX}'\n--- output ---\n${all_output}")
endif()
message(STATUS "exit ${exit_code} as expected")
