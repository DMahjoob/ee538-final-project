#pragma once

// Text matrix format used by the CLI and the test fixtures:
//
//   # comment (anything after '#' on a line is ignored)
//   <rows> <cols>
//   v00 v01 ... v0(cols-1)
//   ...
//
// rows and cols are positive integers. Exactly rows*cols finite float values
// follow, separated by any whitespace (line breaks are not significant, but are
// used for error messages). Values are stored row-major.

#include <cstddef>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

namespace qlinear {

struct MatrixData {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;  // row-major, size rows * cols
};

// A malformed or unreadable matrix file. line() is 1-based, or 0 when the
// problem is not tied to one line (for example, the file cannot be opened).
class InputError : public std::runtime_error {
public:
    InputError(std::string source, std::size_t line, const std::string& message);

    const std::string& source() const { return source_; }
    std::size_t line() const { return line_; }

private:
    std::string source_;
    std::size_t line_;
};

// Parses one matrix from a stream. source_name is used only in messages.
// Throws InputError on any format violation.
MatrixData read_matrix(std::istream& in, const std::string& source_name);

// Opens and parses a file. Throws InputError if it cannot be opened.
MatrixData load_matrix(const std::string& path);

// Writes the format above with 9 significant digits, which reproduces every
// float32 value exactly when read back.
void write_matrix(std::ostream& out, const MatrixData& m);

// Writes to a file. Throws std::runtime_error if it cannot be written.
void save_matrix(const std::string& path, const MatrixData& m);

}  // namespace qlinear
