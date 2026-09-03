#pragma once

#include <hpx/hpx.hpp>
#include <hpx/hpx_finalize.hpp>

#include <charconv>
#include <iostream>
#include <string>
#include <system_error>

namespace arts_hpx {

// The whole token must be the number: a trailing character is a mistyped
// argument, not a value to truncate silently.
template <typename T>
bool parse_int(std::string const& text, T& value)
{
    if (text.empty())
        return false;
    char const* const begin = text.data();
    char const* const end = begin + text.size();
    auto const result = std::from_chars(begin, end, value);
    return result.ec == std::errc{} && result.ptr == end;
}

// Argument validation is identical on every locality and happens before the
// first barrier: a failing locality returns from hpx_main together with the
// others, never alone.
inline int fail_with_usage(char const* message, char const* usage)
{
    if (hpx::get_locality_id() == 0)
        std::cerr << message << '\n' << "Usage: " << usage << '\n';
    hpx::finalize();
    return 1;
}

}    // namespace arts_hpx
