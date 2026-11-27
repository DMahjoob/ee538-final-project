#include "MatrixIO.h"

#include "InputValidator.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <utility>

namespace qlinear {

namespace {

std::string format_message(const std::string& source, std::size_t line,
                           const std::string& message) {
    if (line == 0) {
        return source + ": " + message;
    }
    return source + ", line " + std::to_string(line) + ": " + message;
}

// Yields whitespace-separated tokens with their 1-based line numbers.
// Everything from '#' to the end of a line is ignored.
class Tokenizer {
public:
    explicit Tokenizer(std::istream& in) : in_(in) {}

    bool next(std::string& token, std::size_t& line) {
        while (true) {
            if (line_stream_ >> token) {
                line = line_number_;
                return true;
            }
            std::string raw;
            if (!std::getline(in_, raw)) {
                return false;
            }
            ++line_number_;
            const auto hash = raw.find('#');
            if (hash != std::string::npos) {
                raw.erase(hash);
            }
            line_stream_.clear();
            line_stream_.str(raw);
        }
    }

    // Line number of the last line read (for "file ended" messages).
    std::size_t last_line() const { return line_number_; }

private:
    std::istream& in_;
    std::istringstream line_stream_;
    std::size_t line_number_ = 0;
};

std::size_t parse_dimension(const std::string& token, const char* name,
                            const std::string& source, std::size_t line) {
    bool digits_only = !token.empty();
    for (char ch : token) {
        if (ch < '0' || ch > '9') {
            digits_only = false;
            break;
        }
    }
    if (!digits_only) {
        throw InputError(source, line,
                         std::string("header: ") + name +
                             " must be a positive integer, got '" + token + "'");
    }
    errno = 0;
    const unsigned long long value = std::strtoull(token.c_str(), nullptr, 10);
    if (errno == ERANGE || value > std::numeric_limits<std::size_t>::max()) {
        throw InputError(source, line,
                         std::string("header: ") + name + " '" + token + "' is too large");
    }
    if (value == 0) {
        throw InputError(source, line,
                         std::string("header: ") + name + " must be at least 1, got 0");
    }
    return static_cast<std::size_t>(value);
}

float parse_value(const std::string& token, const std::string& source,
                  std::size_t line) {
    const char* begin = token.c_str();
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(begin, &end);
    if (end == begin || *end != '\0') {
        throw InputError(source, line, "expected a number, got '" + token + "'");
    }
    if (std::isnan(value)) {
        throw InputError(source, line, "NaN is not allowed ('" + token + "')");
    }
    if (std::isinf(value)) {
        // Either a literal "inf" or a finite literal outside float32 range.
        if (errno == ERANGE) {
            throw InputError(source, line,
                             "value '" + token + "' is outside the float32 range");
        }
        throw InputError(source, line, "infinity is not allowed ('" + token + "')");
    }
    // ERANGE with a finite result is underflow to a subnormal or zero: accept.
    return value;
}

}  // namespace

InputError::InputError(std::string source, std::size_t line, const std::string& message)
    : std::runtime_error(format_message(source, line, message)),
      source_(std::move(source)),
      line_(line) {}

MatrixData read_matrix(std::istream& in, const std::string& source_name) {
    Tokenizer tokens(in);
    std::string token;
    std::size_t line = 0;

    if (!tokens.next(token, line)) {
        throw InputError(source_name, 0,
                         "empty input: expected a header line '<rows> <cols>'");
    }
    const std::size_t header_line = line;
    const std::size_t rows = parse_dimension(token, "rows", source_name, line);
    if (!tokens.next(token, line)) {
        throw InputError(source_name, header_line,
                         "header: expected '<rows> <cols>', found only one number");
    }
    if (line != header_line) {
        throw InputError(source_name, header_line,
                         "header: expected '<rows> <cols>' on one line");
    }
    const std::size_t cols = parse_dimension(token, "cols", source_name, line);
    try {
        validate_dimensions(rows, cols, "matrix");
    } catch (const std::invalid_argument& e) {
        throw InputError(source_name, header_line, std::string("header: ") + e.what());
    }

    MatrixData m;
    m.rows = rows;
    m.cols = cols;
    const std::size_t expected = rows * cols;
    m.values.reserve(expected);
    while (m.values.size() < expected) {
        if (!tokens.next(token, line)) {
            throw InputError(source_name, tokens.last_line(),
                             "expected " + std::to_string(expected) + " values for a " +
                                 std::to_string(rows) + " x " + std::to_string(cols) +
                                 " matrix, but the input ends after " +
                                 std::to_string(m.values.size()));
        }
        m.values.push_back(parse_value(token, source_name, line));
    }
    if (tokens.next(token, line)) {
        throw InputError(source_name, line,
                         "unexpected extra value '" + token + "': the header declares " +
                             std::to_string(rows) + " x " + std::to_string(cols) + " = " +
                             std::to_string(expected) + " values");
    }
    return m;
}

MatrixData load_matrix(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw InputError(path, 0, "cannot open file");
    }
    return read_matrix(file, path);
}

void write_matrix(std::ostream& out, const MatrixData& m) {
    const auto old_precision = out.precision(std::numeric_limits<float>::max_digits10);
    out << m.rows << ' ' << m.cols << '\n';
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            if (c != 0) {
                out << ' ';
            }
            out << m.values[r * m.cols + c];
        }
        out << '\n';
    }
    out.precision(old_precision);
}

void save_matrix(const std::string& path, const MatrixData& m) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error(path + ": cannot open file for writing");
    }
    write_matrix(file, m);
    if (!file) {
        throw std::runtime_error(path + ": write failed");
    }
}

}  // namespace qlinear
