#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <sstream>
#include <cstdlib>

// Levels: 0=ERROR,1=WARN,2=INFO
static inline int _get_log_level()
{
    static int lvl = -1;
    if(lvl != -1) return lvl;
    const char* env = std::getenv("ADAPTIVE_LOG_LEVEL");
    if(!env) { lvl = 2; return lvl; }
    int v = atoi(env);
    if(v < 0) v = 0; if(v > 2) v = 2;
    lvl = v;
    return lvl;
}

#define LOG_INFO_STREAM(...) do { if(_get_log_level() >= 2) { std::ostringstream oss; oss << __VA_ARGS__; std::cout << "[INFO] " << oss.str() << std::endl; } } while(0)
#define LOG_WARN_STREAM(...) do { if(_get_log_level() >= 1) { std::ostringstream oss; oss << __VA_ARGS__; std::cout << "[WARN] " << oss.str() << std::endl; } } while(0)
#define LOG_ERROR_STREAM(...) do { if(_get_log_level() >= 0) { std::ostringstream oss; oss << __VA_ARGS__; std::cerr << "[ERROR] " << oss.str() << std::endl; } } while(0)

// Compatibility macros used by older code
#ifndef LOG_INFO
#define LOG_INFO(msg) LOG_INFO_STREAM(msg)
#endif
#ifndef LOG_WARN
#define LOG_WARN(msg) LOG_WARN_STREAM(msg)
#endif
#ifndef LOG_ERROR
#define LOG_ERROR(msg) LOG_ERROR_STREAM(msg)
#endif
#ifndef LOG_DEBUG
// Map debug to info level if debug not separately supported
#define LOG_DEBUG(msg) LOG_INFO_STREAM(msg)
#endif

#endif // LOGGER_H
