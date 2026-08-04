@echo off
REM ============================================================================
REM Build script for adaptive_scheduler JNI native library (Windows)
REM Phase 2.1 C++ Native Implementation
REM ============================================================================

setlocal enabledelayedexpansion

echo [BUILD] Phase 2.1 Native Library - adaptive_scheduler
echo.

REM Check for Java installation
for /f "tokens=*" %%A in ('where java.exe 2^>nul') do set "JAVA_PATH=%%~dpA"
if not defined JAVA_PATH (
    echo [ERROR] Java not found in PATH. Please install Java Development Kit (JDK).
    exit /b 1
)

echo [INFO] Found Java in: !JAVA_PATH!

REM Determine JAVA_HOME
for /f "tokens=*" %%A in ('where javac.exe 2^>nul') do set "JAVAC_PATH=%%~dpA"
if defined JAVAC_PATH (
    set "JAVA_HOME=!JAVAC_PATH!.."
    echo [INFO] JAVA_HOME: !JAVA_HOME!
) else (
    echo [ERROR] javac not found. Please ensure JDK is properly installed.
    exit /b 1
)

REM Create output directory
if not exist "lib" mkdir lib
if not exist "build\native" mkdir build\native

echo [INFO] Output directory: lib\

REM Find MSVC compiler (cl.exe)
where /q cl.exe
if %errorlevel% equ 0 (
    echo [INFO] Using MSVC compiler
    
    REM Compile with MSVC
    for /f "tokens=*" %%A in ('where cl.exe 2^>nul') do set "CL_PATH=%%~dpA"
    
    echo [COMPILE] Compiling adaptive_scheduler.cpp with MSVC...
    cd src\main\cpp
    cl /nologo /EHsc /MD /I"!JAVA_HOME!\include" /I"!JAVA_HOME!\include\win32" /I"headers" ^
        /Fo"..\..\..\..\build\native\\" /Ld adaptive_scheduler.cpp ^
        /link /OUT:"..\..\..\..\lib\adaptive_scheduler.dll" /DLL
    
    if %errorlevel% equ 0 (
        echo [SUCCESS] Native library compiled to lib\adaptive_scheduler.dll
        cd ..\..\..\..
        exit /b 0
    ) else (
        echo [ERROR] Compilation failed with MSVC
        cd ..\..\..\..
        exit /b 1
    )
) else (
    echo [WARNING] MSVC compiler not found. Trying alternative methods...
    
    REM Try MinGW if available
    where /q g++.exe
    if %errorlevel% equ 0 (
        echo [INFO] Found MinGW g++ compiler
        
        echo [COMPILE] Compiling adaptive_scheduler.cpp with g++...
        g++ -shared -fPIC ^
            -I"!JAVA_HOME!\include" -I"!JAVA_HOME!\include\win32" -I"src\main\cpp\headers" ^
            src\main\cpp\adaptive_scheduler.cpp ^
            -o lib\adaptive_scheduler.dll
        
        if %errorlevel% equ 0 (
            echo [SUCCESS] Native library compiled to lib\adaptive_scheduler.dll
            exit /b 0
        ) else (
            echo [ERROR] Compilation failed with g++
            exit /b 1
        )
    ) else (
        echo [ERROR] No C++ compiler found (MSVC or MinGW required)
        echo [INFO] Please install:
        echo         - Microsoft Visual C++ Build Tools, or
        echo         - MinGW-w64
        exit /b 1
    )
)
