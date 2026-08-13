# Merge static archives into one, in place, and localize everything the merged
# archive did not itself export.
#
# A linker searches a single archive to a fixpoint but each of several archives
# only once, in command-line order.  Folding a library's private archives into
# its own therefore does two things at once: the consumer names one -l, and a
# mutually referential set (a static provider table and the providers it names)
# resolves without the explicit group separate archives would need.
#
# Merging alone, though, publishes the bundled libraries' symbols as the
# library's own.  A consumer that links its own copy of one of them then either
# collides with it or, worse, silently binds its own calls to the bundled copy,
# splitting that library's state across two instances of itself.  The members
# are therefore partially linked into one relocatable object -- which resolves
# every reference between them while they are still all visible -- and only the
# names the library exported before the merge are left global.
#
# Script mode:
#   cmake -DMERGE_AR=<ar> -DMERGE_RANLIB=<ranlib> -DMERGE_NM=<nm>
#         -DMERGE_LD=<ld> -DMERGE_OBJCOPY=<objcopy> -DMERGE_ARCHIVE=<out.a>
#         -P MergeStaticArchives.cmake -- <in.a>...
#
# The output archive is both an input and the result: the allowlist is read from
# it before anything is added, so it must hold exactly the library's own objects
# on entry -- drive this from a rule that recreates the output first, which is
# what a static-library link rule does.

cmake_minimum_required(VERSION 3.16)

foreach(_var MERGE_AR MERGE_RANLIB MERGE_NM MERGE_LD MERGE_OBJCOPY MERGE_ARCHIVE)
    if(NOT ${_var})
        message(FATAL_ERROR "MergeStaticArchives: ${_var} not set")
    endif()
endforeach()

# Inputs arrive after the `--` separator, one argument each, so a path may
# contain characters a list separator would split on.
set(inputs "")
set(_after_separator FALSE)
math(EXPR _argc_last "${CMAKE_ARGC} - 1")
foreach(_i RANGE 0 ${_argc_last})
    set(_arg "${CMAKE_ARGV${_i}}")
    if(_after_separator)
        list(APPEND inputs "${_arg}")
    elseif(_arg STREQUAL "--")
        set(_after_separator TRUE)
    endif()
endforeach()
if(NOT inputs)
    message(FATAL_ERROR "MergeStaticArchives: no input archives given")
endif()

set(_work "${MERGE_ARCHIVE}.merge")
set(_mri_file "${MERGE_ARCHIVE}.mri")
set(_allow_file "${MERGE_ARCHIVE}.exports")
set(_redef_file "${MERGE_ARCHIVE}.unversion")
# The single member's file name is what `ar t` and every linker diagnostic
# report, so it is the archive's own name rather than a scratch one.
get_filename_component(_archive_dir "${MERGE_ARCHIVE}" DIRECTORY)
get_filename_component(_archive_stem "${MERGE_ARCHIVE}" NAME_WE)
set(_merged_obj "${_archive_dir}/${_archive_stem}.o")

# Collects the DEFINED global symbols of <archive> into ${out} (caller scope).
# `nm -g --defined-only` prints an address, a one-letter type and the name; an
# archive's output additionally carries a bare "member:" line per member, which
# the address anchor rejects.
function(_merge_defined_globals out archive)
    execute_process(COMMAND "${MERGE_NM}" -g --defined-only "${archive}"
                    OUTPUT_VARIABLE _out OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "MergeStaticArchives: nm failed on ${archive} (${_rc}): ${_err}")
    endif()
    string(REPLACE ";" "\\;" _out "${_out}")
    string(REPLACE "\n" ";" _out "${_out}")
    set(_syms "")
    foreach(_line IN LISTS _out)
        if(_line MATCHES "^[0-9a-fA-F]+[ \t]+[A-Za-z][ \t]+([^ \t]+)$")
            list(APPEND _syms "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _syms)
    set(${out} ${_syms} PARENT_SCOPE)
endfunction()

# The allowlist is the library's own export surface, read before a single
# vendored member is added.  Deriving it rather than naming it is what keeps it
# honest: every public entry point, every internal the shim and the tests reach
# for, and every weak definition an application is meant to override are in it
# by construction, and a new one needs no edit here.
_merge_defined_globals(_exports "${MERGE_ARCHIVE}")
if(NOT _exports)
    message(FATAL_ERROR
        "MergeStaticArchives: ${MERGE_ARCHIVE} defines no global symbols; it "
        "must hold the library's own objects when this runs")
endif()
list(JOIN _exports "\n" _exports_text)
file(WRITE "${_allow_file}" "${_exports_text}\n")

# MRI script: `addlib` copies every member of an archive, which is what keeps
# member-name collisions between the inputs harmless -- the index the final
# ranlib writes maps symbols to member offsets, not to names.
set(_mri "create ${_work}\naddlib ${MERGE_ARCHIVE}\n")
foreach(_in IN LISTS inputs)
    if(NOT EXISTS "${_in}")
        message(FATAL_ERROR "MergeStaticArchives: input archive missing: ${_in}")
    endif()
    string(APPEND _mri "addlib ${_in}\n")
endforeach()
string(APPEND _mri "save\nend\n")
file(WRITE "${_mri_file}" "${_mri}")

execute_process(COMMAND "${MERGE_AR}" -M INPUT_FILE "${_mri_file}"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: ar -M failed (${_rc}): ${_err}")
endif()

# An archive that was itself built with archives on its link line can carry
# them as members.  No linker recurses into a nested archive, so those members
# are dead weight, their names shadow the real inputs in `ar t` output, and a
# partial link refuses them outright ("member in archive is not an object").
execute_process(COMMAND "${MERGE_AR}" t "${_work}"
                OUTPUT_VARIABLE _members OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: ar t failed on ${_work}")
endif()
string(REPLACE "\n" ";" _members "${_members}")
set(_nested "")
foreach(_m IN LISTS _members)
    if(_m MATCHES "\\.a$")
        list(APPEND _nested "${_m}")
    endif()
endforeach()
if(_nested)
    list(REMOVE_DUPLICATES _nested)
    execute_process(COMMAND "${MERGE_AR}" dD "${_work}" ${_nested}
                    RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "MergeStaticArchives: ar d failed (${_rc}): ${_err}")
    endif()
endif()

execute_process(COMMAND "${MERGE_RANLIB}" -D "${_work}"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: ranlib failed (${_rc}): ${_err}")
endif()

# ELF symbol versioning is a property of the FINAL link: a reference to the
# bare name binds to the default-version definition (name@@NODE) only when the
# linker is producing an executable or a shared object.  A partial link does no
# such matching, so a bundled library whose exports are versioned would leave
# its own callers -- and ours -- with dangling references to the bare name.
# Dropping the version from each default definition makes the bare name the
# definition, which is what a static archive wants in any case: nothing about
# an archive is version-selected, and every one of these becomes local below.
_merge_defined_globals(_merged_syms "${_work}")
set(_redef "")
foreach(_sym IN LISTS _merged_syms)
    if(_sym MATCHES "^([^@]+)@@")
        string(APPEND _redef "${_sym} ${CMAKE_MATCH_1}\n")
    endif()
endforeach()
if(_redef)
    file(WRITE "${_redef_file}" "${_redef}")
    execute_process(COMMAND "${MERGE_OBJCOPY}" "--redefine-syms=${_redef_file}" "${_work}"
                    RESULT_VARIABLE _rc ERROR_VARIABLE _err)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "MergeStaticArchives: objcopy --redefine-syms failed (${_rc}): ${_err}")
    endif()
endif()

# Partial link, then localize.  The order is the whole point: while every member
# is still an input, `ld -r` binds each cross-member reference to the symbol
# table entry that defines it, so demoting that entry afterwards cannot break
# the binding.  --whole-archive is required -- a partial link otherwise pulls
# only members that resolve something already undefined.
execute_process(COMMAND "${MERGE_LD}" -r -o "${_merged_obj}" --whole-archive "${_work}"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: ld -r failed (${_rc}): ${_err}")
endif()

execute_process(COMMAND "${MERGE_OBJCOPY}" "--keep-global-symbols=${_allow_file}"
                        "${_merged_obj}"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: objcopy --keep-global-symbols failed (${_rc}): ${_err}")
endif()

# One member from here on.  D on both keeps the result byte-identical across
# rebuilds from identical inputs (no timestamps, uids or modes recorded).
file(REMOVE "${_work}")
execute_process(COMMAND "${MERGE_AR}" crD "${_work}" "${_merged_obj}"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: ar cr failed (${_rc}): ${_err}")
endif()
execute_process(COMMAND "${MERGE_RANLIB}" -D "${_work}"
                RESULT_VARIABLE _rc ERROR_VARIABLE _err)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "MergeStaticArchives: ranlib failed (${_rc}): ${_err}")
endif()

file(RENAME "${_work}" "${MERGE_ARCHIVE}")
file(REMOVE "${_mri_file}" "${_allow_file}" "${_redef_file}" "${_merged_obj}")
