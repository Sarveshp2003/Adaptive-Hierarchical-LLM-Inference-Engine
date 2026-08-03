#include "logger.h"
#include <iostream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>

namespace {
    std::mutex mu;
    std::string timestamp() {
        using namespace std::chrono;
        auto now = system_clock::now();
        auto itt = system_clock::to_time_t(now);
        std::tm tm;
        localtime_s(&tm, &itt);
        auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
        std::ostringstream ss;
        ss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
}

namespace util {

void log_info(const std::string &msg) {
    std::lock_guard<std::mutex> lk(mu);
    std::cout << "[INFO] " << timestamp() << " " << msg << std::endl;
}

void log_warn(const std::string &msg) {
    std::lock_guard<std::mutex> lk(mu);
    std::cerr << "[WARN] " << timestamp() << " " << msg << std::endl;
}

}
