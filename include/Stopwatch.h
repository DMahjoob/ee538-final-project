#pragma once

// Monotonic wall-clock timer in milliseconds (std::chrono::steady_clock).

#include <chrono>

namespace qlinear {

class Stopwatch {
public:
    Stopwatch() : start_(Clock::now()) {}

    void restart() { start_ = Clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

}  // namespace qlinear
