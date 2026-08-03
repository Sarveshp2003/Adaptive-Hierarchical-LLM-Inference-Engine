#ifndef GPU_MEMORY_POOL_H
#define GPU_MEMORY_POOL_H


#include <cstddef>
#include <vector>
#include <mutex>
#include <cuda_runtime.h>
#include <string>
#include <fstream>



class GPUMemoryPool
{

private:


    struct Block
    {

        void* pointer;

        size_t size;

        bool free;
        bool scratch; // true if this block is reserved as scratch space for temporaries

        // Pending CUDA events that reference this block (async copies/kernels)
        std::vector<cudaEvent_t> pendingEvents;

        // Canary guard size in bytes written at start/end of block (0 if disabled)
        size_t guardSize;


    };



    std::vector<Block> blocks;

    // Track original allocated base pointers (call cudaFree on these only)
    std::vector<void*> allocations;

    size_t totalMemory;



    /*
        Tracks whether the pool owns
        allocated CUDA memory.
    */

    bool initialized;



    std::mutex mutex;



    // Diagnostic helper for logging block table (static so .cpp can define it)
    static void dumpBlocks(const std::vector<Block>& blocks);
    // Verify guards across all allocated blocks and capture state on failure
    void verifyAllGuards();

    // Canary helpers
    static size_t computeGuardSize(size_t blockSize);
    bool writeGuardToBlock(Block &b);
    bool checkGuardOfBlock(const Block &b);

    // Ownership check: returns true if pointer belongs to a block (allocated or free) in the pool
    bool ownsPointer(void* ptr) const;

    // Detailed ownership check: returns block index or -1 if not owned
    int findBlockIndex(void* ptr) const;

public:

    // Attach an async CUDA event to a block so the pool knows the block is in-use until the event completes
    void attachEventToBlock(void* ptr, cudaEvent_t event);

    // Wait for any pending events referencing a block and destroy those events
    void waitForBlockEvents(void* ptr);

    // Serialize allocator state (blocks/allocations/pending events) to JSON for post-mortem analysis
    void serializeState(const std::string& path);


    GPUMemoryPool();



    ~GPUMemoryPool();




    void initialize(

        size_t bytes

    );

    // Reserve a scratch partition within the pool for temporary allocations (e.g., cuBLAS)
    void reserveScratch(size_t bytes);

    // Allocate from scratch partition (returns nullptr if not enough scratch space)
    void* allocateScratch(size_t bytes);




    void* allocate(

        size_t bytes

    );




    bool release(

        void* ptr

    );




    void shutdown();

    // Install a top-level unhandled exception filter that writes a minidump on crash
    static void installUnhandledExceptionFilter();



};



#endif