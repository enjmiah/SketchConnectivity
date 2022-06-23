#pragma once

#include <cstdio>

namespace sketching {

/** RAII class for a FILE handle. */
struct FileGuard {
  FILE* const fp_;

  explicit FileGuard(FILE* file)
    : fp_(file) {}

  ~FileGuard() { fclose(fp_); }

  // Disable move.  (Could implement a move constructor, but we don't need one.)
  FileGuard(FileGuard&& mE) noexcept = delete;
  FileGuard& operator=(FileGuard&& mE) noexcept = delete;

  // Disable copy.
  FileGuard(FileGuard& mE) noexcept = delete;
  FileGuard& operator=(FileGuard& mE) noexcept = delete;
};

} // namespace sketching
