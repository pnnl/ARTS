if(NOT DEFINED C_EXECUTABLE OR NOT DEFINED CPP_EXECUTABLE)
    message(FATAL_ERROR "C_EXECUTABLE and CPP_EXECUTABLE are required")
endif()

execute_process(
    COMMAND "${C_EXECUTABLE}"
    RESULT_VARIABLE c_result
    OUTPUT_VARIABLE c_output
    ERROR_VARIABLE c_error)
if(NOT c_result EQUAL 0)
    message(FATAL_ERROR
        "C layout probe failed (${c_result})\nstdout:\n${c_output}\nstderr:\n${c_error}")
endif()

execute_process(
    COMMAND "${CPP_EXECUTABLE}"
    RESULT_VARIABLE cpp_result
    OUTPUT_VARIABLE cpp_output
    ERROR_VARIABLE cpp_error)
if(NOT cpp_result EQUAL 0)
    message(FATAL_ERROR
        "C++ layout probe failed (${cpp_result})\nstdout:\n${cpp_output}\nstderr:\n${cpp_error}")
endif()

string(REGEX MATCH "T219_LAYOUT c [^\r\n]+" c_line "${c_output}")
string(REGEX MATCH "T219_LAYOUT cpp [^\r\n]+" cpp_line "${cpp_output}")
if(c_line STREQUAL "")
    message(FATAL_ERROR "C layout probe emitted no fingerprint\n${c_output}")
endif()
if(cpp_line STREQUAL "")
    message(FATAL_ERROR "C++ layout probe emitted no fingerprint\n${cpp_output}")
endif()

string(REGEX REPLACE "^T219_LAYOUT c " "" c_fingerprint "${c_line}")
string(REGEX REPLACE "^T219_LAYOUT cpp " "" cpp_fingerprint "${cpp_line}")
if(NOT c_fingerprint STREQUAL cpp_fingerprint)
    message(FATAL_ERROR
        "arts_edt_s/arts_event_s C/C++ layout mismatch\nC:   ${c_line}\nC++: ${cpp_line}")
endif()

message("PASS runtime_edt_event_layout_compare: C/C++ fingerprints match")
