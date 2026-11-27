#pragma once

// Minimal dependency-free test harness shared by every component's tests.

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace test {

inline int& failure_count() {
    static int count = 0;
    return count;
}

inline void report_failure(const char* expr, const char* file, int line) {
    ++failure_count();
    std::cerr << "  FAILED: " << expr << " (" << file << ":" << line << ")\n";
}

class Runner {
public:
    void add(std::string name, std::function<void()> fn) {
        cases_.emplace_back(std::move(name), std::move(fn));
    }

    int run() {
        int failed_cases = 0;
        for (auto& [name, fn] : cases_) {
            const int before = failure_count();
            try {
                fn();
            } catch (const std::exception& e) {
                std::cerr << "  unexpected exception: " << e.what() << "\n";
                ++failure_count();
            }
            const bool ok = failure_count() == before;
            std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
            if (!ok) {
                ++failed_cases;
            }
        }
        std::cout << (cases_.size() - failed_cases) << "/" << cases_.size()
                  << " test cases passed\n";
        return failed_cases == 0 ? 0 : 1;
    }

private:
    std::vector<std::pair<std::string, std::function<void()>>> cases_;
};

}  // namespace test

#define CHECK(expr)                                         \
    do {                                                    \
        if (!(expr)) {                                      \
            ::test::report_failure(#expr, __FILE__, __LINE__); \
        }                                                   \
    } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(std::fabs((a) - (b)) <= (tol))

#define CHECK_THROWS(expr, ExceptionType) \
    do {                                  \
        bool threw_ = false;              \
        try {                             \
            (void)(expr);                 \
        } catch (const ExceptionType&) {  \
            threw_ = true;                \
        }                                 \
        CHECK(threw_ && #expr " should throw " #ExceptionType); \
    } while (0)
