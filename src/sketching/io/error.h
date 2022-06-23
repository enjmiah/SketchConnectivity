#pragma once

#include <stdexcept>

namespace sketching {

struct io_error : std::runtime_error {
  io_error(const std::string& what)
    : std::runtime_error(what) {}
};

} // namespace sketching
