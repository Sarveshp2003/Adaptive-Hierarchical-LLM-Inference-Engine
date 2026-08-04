#!/bin/bash

# ============================================================================
# Build script for adaptive_scheduler JNI native library (Linux/macOS)
# Phase 2.1 C++ Native Implementation
# ============================================================================

set -e  # Exit on error

echo "[BUILD] Phase 2.1 Native Library - adaptive_scheduler"
echo ""

# Check for Java installation
if ! command -v java &> /dev/null; then
    echo "[ERROR] Java not found in PATH. Please install Java Development Kit (JDK)."
    exit 1
fi

echo "[INFO] Found Java in: $(which java)"

# Determine JAVA_HOME
if [ -z "$JAVA_HOME" ]; then
    # Try to detect JAVA_HOME
    if command -v java_home &> /dev/null; then
        export JAVA_HOME=$(java_home)
    else
        # Alternative detection
        JAVA_BIN=$(dirname "$(readlink -f "$(which java)")")
        export JAVA_HOME=$(dirname "$JAVA_BIN")
    fi
fi

echo "[INFO] JAVA_HOME: $JAVA_HOME"

# Verify javac is available
if ! command -v javac &> /dev/null; then
    echo "[ERROR] javac not found. Please ensure JDK is properly installed."
    exit 1
fi

# Create output directories
mkdir -p lib
mkdir -p build/native

echo "[INFO] Output directory: lib/"

# Detect platform
PLATFORM=$(uname -s)
case "$PLATFORM" in
    Linux*)
        echo "[INFO] Platform: Linux"
        INCLUDE_PATHS="-I\"$JAVA_HOME/include\" -I\"$JAVA_HOME/include/linux\""
        OUTPUT_LIB="lib/libadaptive_scheduler.so"
        ;;
    Darwin*)
        echo "[INFO] Platform: macOS"
        INCLUDE_PATHS="-I\"$JAVA_HOME/include\" -I\"$JAVA_HOME/include/darwin\""
        OUTPUT_LIB="lib/libadaptive_scheduler.dylib"
        ;;
    MINGW*|MSYS*)
        echo "[INFO] Platform: Windows (MinGW)"
        INCLUDE_PATHS="-I\"$JAVA_HOME/include\" -I\"$JAVA_HOME/include/win32\""
        OUTPUT_LIB="lib/adaptive_scheduler.dll"
        ;;
    *)
        echo "[ERROR] Unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

# Check for C++ compiler
if ! command -v g++ &> /dev/null; then
    if ! command -v clang++ &> /dev/null; then
        echo "[ERROR] No C++ compiler found (g++ or clang++ required)"
        echo "[INFO] Please install: sudo apt-get install g++ (Ubuntu/Debian) or brew install gcc (macOS)"
        exit 1
    else
        COMPILER="clang++"
    fi
else
    COMPILER="g++"
fi

echo "[INFO] Using C++ compiler: $COMPILER"

# Compile
echo "[COMPILE] Compiling adaptive_scheduler.cpp with $COMPILER..."
eval "$COMPILER -shared -fPIC \
    $INCLUDE_PATHS -I\"src/main/cpp/headers\" \
    -o \"$OUTPUT_LIB\" \
    src/main/cpp/adaptive_scheduler.cpp"

if [ $? -eq 0 ]; then
    echo "[SUCCESS] Native library compiled to $OUTPUT_LIB"
    echo "[INFO] Library size: $(ls -lh \"$OUTPUT_LIB\" | awk '{print $5}')"
    echo ""
    echo "[NEXT] To use the native library:"
    echo "       1. Set java.library.path to: $(pwd)/lib"
    echo "       2. Run integration test: java -Djava.library.path=$(pwd)/lib -cp ..."
    exit 0
else
    echo "[ERROR] Compilation failed"
    exit 1
fi
