#include "GPUMemoryPool.h"

#include <cuda_runtime.h>
#include "CUDAError.h"
#include "AllocationTracker.h"
#include "Logger.h"

#include <mutex>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <chrono>

// MiniDump writer helper (runtime-load dbghelp so no link-time dependency)
#ifdef _WIN32
#include <windows.h>
#endif

static bool writeMiniDumpToFile(const std::string &path)
{
#ifdef _WIN32
    HMODULE hDbg = LoadLibraryW(L"Dbghelp.dll");
    if(!hDbg) {
        return false;
    }
    typedef BOOL (WINAPI *MiniDumpWriteDump_t)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
    MiniDumpWriteDump_t pFunc = (MiniDumpWriteDump_t)GetProcAddress(hDbg, "MiniDumpWriteDump");
    if(!pFunc) {
        FreeLibrary(hDbg);
        return false;
    }
    // Convert UTF-8 path to wide string
    std::wstring wpath;
    int len = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
    if(len > 0) {
        wpath.resize(len);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], len);
    }
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if(hFile == INVALID_HANDLE_VALUE) {
        // fallback to temp path
        char tempPath[MAX_PATH+1] = {0};
        DWORD tlen = GetTempPathA(MAX_PATH, tempPath);
        if(tlen == 0 || tlen > MAX_PATH) {
            FreeLibrary(hDbg);
            return false;
        }
        char filename[MAX_PATH+1];
        sprintf_s(filename, sizeof(filename), "%s\\minidump_fallback.dmp", tempPath);
        int llen = MultiByteToWideChar(CP_UTF8, 0, filename, -1, NULL, 0);
        if(llen > 0) {
            std::wstring wfile;
            wfile.resize(llen);
            MultiByteToWideChar(CP_UTF8, 0, filename, -1, &wfile[0], llen);
            hFile = CreateFileW(wfile.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if(hFile == INVALID_HANDLE_VALUE) {
                FreeLibrary(hDbg);
                return false;
            }
        } else {
            FreeLibrary(hDbg);
            return false;
        }
    }
    BOOL ok = pFunc(GetCurrentProcess(), GetCurrentProcessId(), hFile, /*MiniDumpWithFullMemory*/ 0x00000002, NULL, NULL, NULL);
    CloseHandle(hFile);
    FreeLibrary(hDbg);
    return ok == TRUE;
#else
    (void)path; return false;
#endif
}

#ifdef _WIN32
// Unhandled exception filter that writes a minidump when the process crashes
static LONG WINAPI AdaptiveUnhandledExceptionFilter(EXCEPTION_POINTERS* ExceptionInfo) {
    char cwdBuf[MAX_PATH] = {0};
    if(GetCurrentDirectoryA(MAX_PATH, cwdBuf) == 0) { strcpy_s(cwdBuf, MAX_PATH, "."); }
    char ts[64]; SYSTEMTIME st; GetLocalTime(&st);
    sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    std::string dumpPath = std::string(cwdBuf) + "\\native-engine\\dumps\\unhandled_" + ts + ".dmp";

    HMODULE hDbg = LoadLibraryW(L"Dbghelp.dll");
    if(hDbg) {
        typedef BOOL (WINAPI *MiniDumpWriteDump_t)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        MiniDumpWriteDump_t pFunc = (MiniDumpWriteDump_t)GetProcAddress(hDbg, "MiniDumpWriteDump");
        if(pFunc) {
            int len = MultiByteToWideChar(CP_UTF8, 0, dumpPath.c_str(), -1, NULL, 0);
            std::wstring wpath;
            if(len > 0) {
                wpath.resize(len);
                MultiByteToWideChar(CP_UTF8, 0, dumpPath.c_str(), -1, &wpath[0], len);
            }
            HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            if(hFile != INVALID_HANDLE_VALUE) {
#ifndef MINIDUMP_EXCEPTION_INFORMATION
                typedef struct _MINIDUMP_EXCEPTION_INFORMATION {
                    DWORD ThreadId;
                    PEXCEPTION_POINTERS ExceptionPointers;
                    BOOL ClientPointers;
                } MINIDUMP_EXCEPTION_INFORMATION, *PMINIDUMP_EXCEPTION_INFORMATION;
#endif
                MINIDUMP_EXCEPTION_INFORMATION mei;
                mei.ThreadId = GetCurrentThreadId();
                mei.ExceptionPointers = ExceptionInfo;
                mei.ClientPointers = TRUE;
                pFunc(GetCurrentProcess(), GetCurrentProcessId(), hFile, /*MiniDumpWithFullMemory*/ 0x00000002, &mei, NULL, NULL);
                CloseHandle(hFile);
            }
        }
        FreeLibrary(hDbg);
    }
    return EXCEPTION_EXECUTE_HANDLER;
}

void GPUMemoryPool::installUnhandledExceptionFilter() {
    SetUnhandledExceptionFilter(AdaptiveUnhandledExceptionFilter);
}
#endif



GPUMemoryPool::GPUMemoryPool()

{

    totalMemory = 0;

    initialized = false;

}



GPUMemoryPool::~GPUMemoryPool()

{

    shutdown();

}





void GPUMemoryPool::initialize(size_t bytes)

{

    if(initialized)
    {
        return;
    }


    std::lock_guard<std::mutex> lock(mutex);


    void* memory = nullptr;

    // Ensure base allocation is aligned to 256 bytes for suballocation arithmetic and driver expectations
    const size_t ALIGN = 256;
    size_t allocSize = bytes + ALIGN; // over-allocate so we can align within the buffer

    // Use tracked allocator
    cudaError_t _err = AllocationTracker::trackedCudaMalloc(&memory, allocSize, "GPUMemoryPool::initialize");
    CUDA_CHECK(_err);

    // record base allocation for later free
    allocations.push_back(memory);

    // ensure tracker initialized
    AllocationTracker::init();

    // Compute an aligned pointer inside the allocated region
    uintptr_t base = reinterpret_cast<uintptr_t>(memory);
    uintptr_t aligned = (base + (ALIGN - 1)) & ~(uintptr_t)(ALIGN - 1);
    size_t offset = static_cast<size_t>(aligned - base);
    if(offset + bytes > allocSize) {
    // Should not happen, but guard
    LOG_ERROR_STREAM("GPUMemoryPool: alignment computation failed — insufficient space");
    // fall back to using base as-is
    {
        Block b; b.pointer = memory; b.size = bytes; b.free = true; b.scratch = false; b.pendingEvents.clear(); b.guardSize = 0;
        blocks.push_back(b);
    }
    totalMemory = bytes;
    } else {
    void* alignedPtr = reinterpret_cast<void*>(aligned);
    if(alignedPtr != memory) {
        LOG_INFO_STREAM("GPUMemoryPool: base allocation " << memory << " aligned to " << alignedPtr << " (offset=" << offset << ")");
    }
    {
        Block b; b.pointer = alignedPtr; b.size = bytes; b.free = true; b.scratch = false; b.pendingEvents.clear(); b.guardSize = 0;
        blocks.push_back(b);
    }
    totalMemory = bytes;
    }


    initialized = true;


    LOG_INFO_STREAM("GPU Pool initialized: " << bytes / (1024 * 1024) << " MB");

}





void GPUMemoryPool::dumpBlocks(const std::vector<Block>& blocks) {
    std::string s = "[GPUMemoryPool] blocks:";
    for(size_t i=0;i<blocks.size();++i) {
        const auto &b = blocks[i];
        char buf[512];
        sprintf(buf, " idx=%zu ptr=%p size=%zu free=%d scratch=%d guard=%zu;", i, b.pointer, b.size, b.free?1:0, b.scratch?1:0, b.guardSize);
        s += buf;
    }
    LOG_INFO_STREAM(s);
}

// Scan all blocks and validate guard regions; capture state/dump on mismatch
void GPUMemoryPool::verifyAllGuards() {
    // Try to acquire the mutex, but if already held by caller, proceed without locking to avoid deadlock
    bool locked = false;
    std::unique_lock<std::mutex> ulock(mutex, std::defer_lock);
    if(ulock.try_lock()) locked = true;
    for(size_t i=0;i<blocks.size();++i) {
        const auto &b = blocks[i];
        if(b.guardSize == 0) continue;
        if(!checkGuardOfBlock(b)) {
            LOG_ERROR_STREAM("verifyAllGuards: guard check failed for block idx=" << i << " ptr=" << b.pointer << " size=" << b.size);
            try { serializeState(std::string("out/allocator_state_verify_failure.json")); } catch(...) {}
#ifdef _WIN32
            char cwdBuf[MAX_PATH] = {0};
            if(GetCurrentDirectoryA(MAX_PATH, cwdBuf) == 0) strcpy_s(cwdBuf, MAX_PATH, ".");
            char ts[64]; SYSTEMTIME st; GetLocalTime(&st);
            sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
            std::string dumpPath = std::string(cwdBuf) + "\\native-engine\\dumps\\verify_failure_" + ts + ".dmp";
            bool dumped = writeMiniDumpToFile(dumpPath);
            if(dumped) LOG_INFO_STREAM("verifyAllGuards: wrote minidump " << dumpPath); else LOG_WARN_STREAM("verifyAllGuards: failed to write minidump");
#endif
            // continue scanning; do not abort so we can collect more data
        }
    }
    if(locked) ulock.unlock();
}

// Internal helpers: write and verify canary guards for a block
static const uint64_t GUARD_PATTERN = 0xDEADBEEFDEADBEEFULL;

size_t GPUMemoryPool::computeGuardSize(size_t blockSize) {
    // Choose guard size as min(64, blockSize/8), but at least 8 bytes if possible
    size_t g = blockSize / 8;
    if(g > 512) g = 512; // increase guard cap for stronger detection
    if(g < 16) g = 0; // require at least 16 bytes to be useful
    return g;
}

bool GPUMemoryPool::writeGuardToBlock(Block &b) {
    if(b.guardSize == 0) return true;
    if(!b.pointer) {
        LOG_WARN_STREAM("writeGuardToBlock: null pointer");
        return false;
    }

    // Clamp and sanitize guard size to avoid absurd values
    size_t guard = b.guardSize;
    if(guard > 512) guard = 512;
    if(guard > b.size) guard = b.size; // don't exceed block
    if(guard > 0 && guard < 16) {
        // not useful — skip
        b.guardSize = 0;
        return true;
    }
    b.guardSize = guard;

    // Basic range sanity check using first block as base if available
    if(!blocks.empty()) {
        uintptr_t base = reinterpret_cast<uintptr_t>(blocks.front().pointer);
        uintptr_t top = base + totalMemory;
        uintptr_t p = reinterpret_cast<uintptr_t>(b.pointer);
        if(p < base || (p + b.size) > top) {
            char msg[256];
            _snprintf_s(msg, sizeof(msg), _TRUNCATE, "writeGuardToBlock: pointer out of pool range ptr=%p base=%p top=%p size=%zu\n", b.pointer, (void*)base, (void*)top, b.size);
            fprintf(stderr, "%s", msg);
            try { serializeState(std::string("out/allocator_state_writeguard_outofrange.json")); } catch(...) {}
            return false;
        }
    }

    // Detect common heap-free pattern which indicates corruption
    uintptr_t ip = reinterpret_cast<uintptr_t>(b.pointer);
    if(ip == 0xFEEEFEEEFEEEFEEEULL) {
        char msg[256];
        _snprintf_s(msg, sizeof(msg), _TRUNCATE, "writeGuardToBlock: pointer appears to be freed pattern: %p\n", b.pointer);
        fprintf(stderr, "%s", msg);
        try { serializeState(std::string("out/allocator_state_writeguard_freedptr.json")); } catch(...) {}
        return false;
    }

    // Verify pointer attributes before attempting device transfer
    cudaPointerAttributes pat;
    cudaError_t patErr = cudaPointerGetAttributes(&pat, b.pointer);
    if(patErr != cudaSuccess) {
        char msg[256]; _snprintf_s(msg, sizeof(msg), _TRUNCATE, "writeGuardToBlock: cudaPointerGetAttributes failed: %s\n", cudaGetErrorString(patErr)); fprintf(stderr, "%s", msg);
        try { serializeState(std::string("out/allocator_state_writeguard_ptrattr_fail.json")); } catch(...) {}
        return false;
    }

    // Prepare host buffer
    std::vector<uint8_t> buf;
    try {
        buf.resize(b.guardSize);
    } catch(...) {
        fprintf(stderr, "writeGuardToBlock: failed to allocate guard buffer of size %zu\n", b.guardSize);
        return false;
    }
    for(size_t i=0;i<b.guardSize;i+=8) {
        size_t nn = ((size_t)8 < (b.guardSize - i)) ? (size_t)8 : (b.guardSize - i);
        uint64_t v = GUARD_PATTERN;
        memcpy(&buf[i], &v, nn);
    }

    auto doMemcpyHostToDevice = [&](void* dst)->cudaError_t {
        // Double-check dst pointer attributes
        cudaPointerAttributes dstPat;
        if(cudaPointerGetAttributes(&dstPat, dst) != cudaSuccess) {
            return cudaErrorInvalidValue;
        }
        // Log pointer / guard info to help triage
        char pinfo[256]; _snprintf_s(pinfo, sizeof(pinfo), _TRUNCATE, "writeGuardToBlock: memcpy dst=%p guard=%zu\n", dst, b.guardSize); fprintf(stderr, "%s", pinfo);
        cudaError_t last = cudaErrorUnknown;
        for(int attempt=0; attempt<3; ++attempt) {
            cudaError_t e = cudaMemcpy(dst, buf.data(), b.guardSize, cudaMemcpyHostToDevice);
            if(e == cudaSuccess) return e;
            last = e;
            char msg[256]; _snprintf_s(msg, sizeof(msg), _TRUNCATE, "writeGuardToBlock: cudaMemcpy attempt %d failed: %s\n", attempt, cudaGetErrorString(e)); fprintf(stderr, "%s", msg);
            cudaDeviceSynchronize();
            cudaGetLastError();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        // After retries, serialize state and optionally abort to capture a dump for analysis
        try { serializeState(std::string("out/allocator_state_writeguard_memcpy_final_fail.json")); } catch(...) {}
        const char* envFail = std::getenv("ADAPTIVE_FAIL_ON_INVALID_MEMCPY");
        if(envFail && envFail[0] == '1') {
#ifdef _WIN32
            char cwdBuf[MAX_PATH] = {0}; if(GetCurrentDirectoryA(MAX_PATH, cwdBuf) == 0) strcpy_s(cwdBuf, MAX_PATH, ".");
            char ts[64]; SYSTEMTIME st; GetLocalTime(&st); sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
            std::string dumpPath = std::string(cwdBuf) + "\\native-engine\\dumps\\memcpy_invalid_" + ts + ".dmp";
            bool dumped = writeMiniDumpToFile(dumpPath);
            if(dumped) fprintf(stderr, "writeGuardToBlock: wrote minidump %s\n", dumpPath.c_str()); else fprintf(stderr, "writeGuardToBlock: failed to write minidump\n");
#endif
            abort();
        }
        return last;
    };

    if(b.size < b.guardSize) {
        char m2[256]; _snprintf_s(m2, sizeof(m2), _TRUNCATE, "writeGuardToBlock: block too small for guard size ptr=%p size=%zu guard=%zu\n", b.pointer, b.size, b.guardSize); fprintf(stderr, "%s", m2);
        try { serializeState(std::string("out/allocator_state_writeguard_blocktoosmall.json")); } catch(...) {}
        return false;
    }

    cudaError_t e1 = doMemcpyHostToDevice(b.pointer);
    if(e1 != cudaSuccess) {
        char m[256]; _snprintf_s(m, sizeof(m), _TRUNCATE, "writeGuardToBlock: cudaMemcpy start failed after retries: %s\n", cudaGetErrorString(e1)); fprintf(stderr, "%s", m);
        try { serializeState(std::string("out/allocator_state_writeguard_memcpy_start_fail.json")); } catch(...) {}
        return false;
    }

    void* endPtr = static_cast<char*>(b.pointer) + (b.size - b.guardSize);
    cudaError_t e2 = doMemcpyHostToDevice(endPtr);
    if(e2 != cudaSuccess) {
        char m3[256]; _snprintf_s(m3, sizeof(m3), _TRUNCATE, "writeGuardToBlock: cudaMemcpy end failed after retries: %s\n", cudaGetErrorString(e2)); fprintf(stderr, "%s", m3);
        try { serializeState(std::string("out/allocator_state_writeguard_memcpy_end_fail.json")); } catch(...) {}
        return false;
    }
    return true;
}

bool GPUMemoryPool::checkGuardOfBlock(const Block &b) {
    if(b.guardSize == 0) return true;
    if(!b.pointer) { LOG_WARN_STREAM("checkGuardOfBlock: null pointer"); return false; }
    if(!blocks.empty()) {
        uintptr_t base = reinterpret_cast<uintptr_t>(blocks.front().pointer);
        uintptr_t top = base + totalMemory;
        uintptr_t p = reinterpret_cast<uintptr_t>(b.pointer);
        if(p < base || (p + b.size) > top) {
            LOG_ERROR_STREAM("checkGuardOfBlock: pointer out of pool range ptr=" << b.pointer << " base=" << (void*)base << " top=" << (void*)top << " size=" << b.size);
            return false;
        }
    }
    // read start with retries
    std::vector<uint8_t> hostBuf(b.guardSize);
    auto doMemcpyDeviceToHost = [&](const void* src)->cudaError_t {
        cudaError_t last = cudaErrorUnknown;
        for(int attempt=0; attempt<3; ++attempt) {
            cudaError_t e = cudaMemcpy(hostBuf.data(), src, b.guardSize, cudaMemcpyDeviceToHost);
            if(e == cudaSuccess) return e;
            last = e;
            LOG_WARN_STREAM("checkGuardOfBlock: cudaMemcpy attempt " << attempt << " failed: " << cudaGetErrorString(e));
            cudaDeviceSynchronize();
            cudaGetLastError();
            cudaPointerAttributes pat;
                        if(cudaPointerGetAttributes(&pat, src) == cudaSuccess) {
                            LOG_INFO_STREAM("Pointer attributes: device=" << pat.device);
                        }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return last;
    };

    cudaError_t r1 = doMemcpyDeviceToHost(b.pointer);
    if(r1 != cudaSuccess) { LOG_WARN_STREAM("checkGuardOfBlock: cudaMemcpy start failed after retries: " << cudaGetErrorString(r1)); return false; }
    for(size_t i=0;i<b.guardSize;i+=8) {
        uint64_t v = 0; size_t nn = ((size_t)8 < (b.guardSize - i)) ? (size_t)8 : (b.guardSize - i); memcpy(&v, &hostBuf[i], nn); if(v != GUARD_PATTERN) { LOG_ERROR_STREAM("Guard mismatch at start of block ptr=" << b.pointer << " offset=" << i); try { serializeState(std::string("out/allocator_state_guard_mismatch.json")); } catch(...) {} return false; }
    }
    if(b.size < b.guardSize) { LOG_ERROR_STREAM("checkGuardOfBlock: block too small for guard size ptr="<< b.pointer); return false; }
    void* endPtr = static_cast<char*>(b.pointer) + (b.size - b.guardSize);
    cudaError_t r2 = doMemcpyDeviceToHost(endPtr);
    if(r2 != cudaSuccess) { LOG_WARN_STREAM("checkGuardOfBlock: cudaMemcpy end failed after retries: " << cudaGetErrorString(r2)); return false; }
    for(size_t i=0;i<b.guardSize;i+=8) {
        uint64_t v = 0; size_t nn = ((size_t)8 < (b.guardSize - i)) ? (size_t)8 : (b.guardSize - i); memcpy(&v, &hostBuf[i], nn); if(v != GUARD_PATTERN) { LOG_ERROR_STREAM("Guard mismatch at end of block ptr=" << b.pointer << " offset=" << i); try { serializeState(std::string("out/allocator_state_guard_mismatch.json")); } catch(...) {} return false; }
    }
    return true;
}

void* GPUMemoryPool::allocate(

    size_t bytes

)

{

    std::lock_guard<std::mutex> lock(mutex);

    LOG_INFO_STREAM("GPUMemoryPool::allocate requested bytes=" << bytes << " allocations_count=" << allocations.size());
    dumpBlocks(blocks);

    for(size_t i = 0; i < blocks.size(); i++)

    {

        Block& block = blocks[i];



        if(block.free && !block.scratch && block.size >= bytes)

        {

            void* result =
                block.pointer;



            if(block.size > bytes)

            {

                Block remaining;

                remaining.pointer = static_cast<char*>(block.pointer) + bytes;
                remaining.size = block.size - bytes;
                remaining.free = true;
                remaining.scratch = false;
                remaining.pendingEvents.clear();
                remaining.guardSize = 0;

                block.size = bytes;
                block.free = false;

                blocks.insert(blocks.begin() + i + 1, remaining);

            }

            else

            {

                block.free = false;

            }

            
                // set guard size and write canaries for the allocated portion
                // NOTE: Skip guard writes when running under compute-sanitizer because
                // derived pointers within a pool allocation may not be recognized by
                // compute-sanitizer as valid CUDA allocations, causing "Allocation not found" errors
                const char* isComputeSanitizer = std::getenv("COMPUTE_SANITIZER");
                const char* enableGuards = std::getenv("ADAPTIVE_ENABLE_GUARDS");
                block.guardSize = computeGuardSize(block.size);
                if(block.guardSize > 0 && !isComputeSanitizer && enableGuards && enableGuards[0] == '1') {
                    if(!writeGuardToBlock(block)) {
                        LOG_WARN_STREAM("Failed to write guard to newly allocated block ptr=" << block.pointer);
                    }
                    // perform a full verification pass immediately after allocation to detect early corruption
                    try {
                        verifyAllGuards();
                    } catch(...) {
                        LOG_WARN_STREAM("verifyAllGuards threw an exception during allocation verification");
                    }
                }

            LOG_INFO_STREAM("GPU allocate: " << bytes << " bytes at " << result);
            dumpBlocks(blocks);

            return result;

        }

    }


    LOG_ERROR_STREAM("GPU Pool Out of Memory. Requested: " << bytes << " bytes");
    dumpBlocks(blocks);



    return nullptr;

}


// Reserve a scratch partition at the end of the pool. If the pool has a free block
// large enough, split it so that the last 'bytes' become a scratch block.
void GPUMemoryPool::reserveScratch(size_t bytes)
{
    std::lock_guard<std::mutex> lock(mutex);
    if(bytes == 0) return;
    // Find last free block with sufficient size
    for(int i = static_cast<int>(blocks.size()) - 1; i >= 0; --i) {
        Block &b = blocks[i];
        if(b.free && !b.scratch && b.size >= bytes) {
            // Split into main + scratch
            size_t mainSize = b.size - bytes;
            void* scratchPtr = static_cast<char*>(b.pointer) + mainSize;
            b.size = mainSize;
            // insert scratch block after b
            Block scratch;
            scratch.pointer = scratchPtr;
            scratch.size = bytes;
            scratch.free = true;
            scratch.scratch = true;
            scratch.pendingEvents.clear();
            scratch.guardSize = 0;
            blocks.insert(blocks.begin() + i + 1, scratch);
            LOG_INFO_STREAM("GPUMemoryPool: reserved scratch " << bytes << " bytes at " << scratchPtr);
            return;
        }
    }
    LOG_WARN_STREAM("GPUMemoryPool: reserveScratch failed — not enough free contiguous memory");
}

// Allocate from the scratch partition only
void* GPUMemoryPool::allocateScratch(size_t bytes)
{
    std::lock_guard<std::mutex> lock(mutex);
    LOG_INFO_STREAM("GPUMemoryPool::allocateScratch requested bytes=" << bytes << " allocations_count=" << allocations.size());
    dumpBlocks(blocks);
    for(size_t i = 0; i < blocks.size(); ++i) {
        Block &block = blocks[i];
        if(block.free && block.scratch && block.size >= bytes) {
            void* result = block.pointer;
            if(block.size > bytes) {
                Block remaining;
                remaining.pointer = static_cast<char*>(block.pointer) + bytes;
                remaining.size = block.size - bytes;
                remaining.free = true;
                remaining.scratch = true;
                remaining.pendingEvents.clear();
                remaining.guardSize = 0;
                block.size = bytes;
                block.free = false;
                block.scratch = true;
                blocks.insert(blocks.begin() + i + 1, remaining);
            } else {
                block.free = false;
            }
            // set guard size and write canaries for scratch allocation
            // NOTE: Skip guard writes when running under compute-sanitizer (see note in allocate())
            const char* isComputeSanitizer = std::getenv("COMPUTE_SANITIZER");
            const char* enableGuards = std::getenv("ADAPTIVE_ENABLE_GUARDS");
            block.guardSize = computeGuardSize(block.size);
            if(block.guardSize > 0 && !isComputeSanitizer && enableGuards && enableGuards[0] == '1') {
                if(!writeGuardToBlock(block)) {
                    LOG_WARN_STREAM("Failed to write guard to scratch block ptr=" << block.pointer);
                }
            }
            LOG_INFO_STREAM("GPUMemoryPool allocateScratch: " << bytes << " bytes at " << result);
            dumpBlocks(blocks);
            return result;
        }
    }
    LOG_ERROR_STREAM("GPUMemoryPool scratch OOM. Requested: " << bytes << " bytes");
    dumpBlocks(blocks);
    return nullptr;
}

bool GPUMemoryPool::ownsPointer(void* ptr) const
{
    return findBlockIndex(const_cast<void*>(ptr)) >= 0;
}

int GPUMemoryPool::findBlockIndex(void* ptr) const
{
    if(!ptr) return -1;
    const auto* p = static_cast<const unsigned char*>(ptr);
    for(size_t i = 0; i < blocks.size(); i++) {
        const auto* start = static_cast<const unsigned char*>(blocks[i].pointer);
        const auto* end = start + blocks[i].size;
        if(p >= start && p < end) {
            return static_cast<int>(i);
        }
    }
    return -1;
}



void GPUMemoryPool::attachEventToBlock(void* ptr, cudaEvent_t event) {
    if(!ptr || !event) return;
    std::lock_guard<std::mutex> lock(mutex);
    int idx = findBlockIndex(ptr);
    if(idx >= 0) {
        blocks[idx].pendingEvents.push_back(event);
        LOG_INFO_STREAM("attachEventToBlock: attached event to block " << idx << " ptr=" << ptr);
    } else {
        // Unknown pointer — destroy event to avoid leaks
        LOG_WARN_STREAM("attachEventToBlock: unknown pointer, destroying event. ptr=" << ptr);
        cudaEventDestroy(event);
    }
}

void GPUMemoryPool::waitForBlockEvents(void* ptr) {
    if(!ptr) return;
int idx = findBlockIndex(ptr);
if(idx < 0) return;
auto &vec = blocks[idx].pendingEvents;
for(auto ev : vec) {
    if(ev) {
        cudaError_t e = cudaEventSynchronize(ev);
        if(e != cudaSuccess) {
            LOG_WARN_STREAM("cudaEventSynchronize failed: " << cudaGetErrorString(e));
        }
        cudaEventDestroy(ev);
    }
}
vec.clear();
}

void GPUMemoryPool::serializeState(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex);
    std::ofstream f(path);
    if(!f.is_open()) { LOG_WARN_STREAM("serializeState: failed to open " << path); return; }
    f << "{\"blocks\": [";
    for(size_t i=0;i<blocks.size();++i) {
        const auto &b = blocks[i];
        f << "{";
        f << "\"pointer\": \"" << b.pointer << "\",";
        f << "\"size\": " << b.size << ",";
        f << "\"free\": " << (b.free?1:0) << ",";
        f << "\"scratch\": " << (b.scratch?1:0) << ",";
        f << "\"pendingEvents\": " << b.pendingEvents.size();
        f << "}";
        if(i+1<blocks.size()) f << ",";
    }
    f << "] }\n";
    f.close();
    LOG_INFO_STREAM("GPUMemoryPool: serialized allocator state to " << path);
}



bool GPUMemoryPool::release(
    
    void* ptr
    
)
    
{
    
    std::lock_guard<std::mutex> lock(mutex);
    
    int idx = findBlockIndex(ptr);
    if(idx >= 0) {
        size_t i = static_cast<size_t>(idx);
        // Ensure any pending CUDA operations referencing this block have completed before marking it free
        waitForBlockEvents(ptr);
        // Optional guard verification is disabled by default to keep the allocator stable in this build.
        // Enable it explicitly by setting ADAPTIVE_ENABLE_GUARDS=1 when debugging corruption issues.
        const char* enableGuards = std::getenv("ADAPTIVE_ENABLE_GUARDS");
        if(enableGuards && enableGuards[0] == '1') {
            bool guardOk = checkGuardOfBlock(blocks[i]);
            if(!guardOk) {
                LOG_ERROR_STREAM("GPUMemoryPool::release detected guard corruption for ptr=" << ptr);
                // attempt to write a minidump for offline analysis
                #ifdef _WIN32
                char cwdBuf[MAX_PATH] = {0}; if(GetCurrentDirectoryA(MAX_PATH, cwdBuf) == 0) strcpy_s(cwdBuf, MAX_PATH, ".");
                char ts[64]; SYSTEMTIME st; GetLocalTime(&st); sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
                std::string dumpPath = std::string(cwdBuf) + "\\native-engine\\dumps\\guard_corrupt_" + ts + ".dmp";
                bool dumped = writeMiniDumpToFile(dumpPath);
                if(dumped) LOG_INFO_STREAM("Guard corruption minidump written: " << dumpPath);
                else LOG_WARN_STREAM("Failed to write guard corruption minidump");
                #endif
            }
        }
        blocks[i].free = true;

        
        
        /*
            Merge with next block
        */
        
        if(i + 1 < blocks.size()
            &&
           blocks[i+1].free)
        
        {
            
            blocks[i].size +=
blocks[i+1].size;
            
            
            
            blocks.erase(
        
blocks.begin() + i + 1
            
            );
        
        
        }
        
        
        /*
            Merge with previous block
        */
        
        if(i > 0
            &&
           blocks[i-1].free)
        
        {
            
            blocks[i-1].size +=
blocks[i].size;
            
            
            
            blocks.erase(
            
blocks.begin() + i
            
            );
        
        }
        
        
        
        return true;
    }
    
    
    
    LOG_WARN_STREAM("GPU release failed. Pointer not found: " << ptr);
    // If environment requests, abort on invalid free so ProcDump can capture a full minidump for analysis
    const char* envFail = std::getenv("ADAPTIVE_FAIL_ON_INVALID_FREE");
    if(envFail && envFail[0] == '1') {
        LOG_ERROR_STREAM("ADAPTIVE_FAIL_ON_INVALID_FREE=1 - aborting to generate minidump for debugging. Pointer:" << ptr);
        fflush(nullptr);
        // Attempt to write a minidump before aborting so ProcDump is not required
        {
#ifdef _WIN32
            char cwdBuf[MAX_PATH] = {0};
            if(GetCurrentDirectoryA(MAX_PATH, cwdBuf) != 0) {
                std::string cwdStr = cwdBuf;
                char ts[64];
                SYSTEMTIME st; GetLocalTime(&st);
                sprintf_s(ts, sizeof(ts), "%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
                std::string dumpPath = cwdStr + "\\native-engine\\dumps\\minidump_abort_" + ts + ".dmp";
                bool dumped = writeMiniDumpToFile(dumpPath);
                if(dumped) LOG_INFO_STREAM("Abort minidump written: " << dumpPath);
                else LOG_WARN_STREAM("Failed to write abort minidump to: " << dumpPath);
            }
#endif
        }
        abort();
    }

    // Defensive fallback: attempt tracked cudaFree if pointer appears to be a device allocation
    // Ensure any pending device work completes before inspecting pointer attributes
    cudaError_t preSync = cudaDeviceSynchronize();
    if(preSync != cudaSuccess) {
        LOG_WARN_STREAM("cudaDeviceSynchronize in release fallback failed: " << cudaGetErrorString(preSync));
    }
    cudaPointerAttributes attr;
    cudaError_t gerr = cudaPointerGetAttributes(&attr, ptr);
    if(gerr == cudaSuccess) {
        LOG_WARN_STREAM("GPUPool release fallback: pointer attributes found, attempting trackedCudaFree. type=" << attr.type << " device=" << attr.device);
        cudaError_t ferr = AllocationTracker::trackedCudaFree(ptr, "GPUMemoryPool::release fallback");
        if(ferr == cudaSuccess) { LOG_INFO_STREAM("GPUMemoryPool::release fallback freed ptr="<<ptr); return true; }
        else { LOG_ERROR_STREAM("GPUMemoryPool::release fallback free failed: "<< ferr); return false; }
    }
    return false;

}





void GPUMemoryPool::shutdown()

{

    std::lock_guard<std::mutex> lock(mutex);


    if(!initialized)
    {
        return;
    }


    LOG_INFO_STREAM("[GPUMemoryPool] shutdown: resetting pool state for " << allocations.size() << " base allocations");

    // Best-effort state reset for this lightweight reconstruction build.
    // Explicit CUDA free can be re-enabled once the allocator semantics are fully restored.
    allocations.clear();
    blocks.clear();

    totalMemory = 0;

    initialized = false;

    LOG_INFO_STREAM("GPU Pool shutdown");

}