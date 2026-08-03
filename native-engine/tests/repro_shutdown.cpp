#include "GPUMemoryPool.h"
#include <iostream>

int main()
{
    std::cout << "Repro: starting\n";
    GPUMemoryPool pool;
    pool.initialize(1024 * 1024); // 1 MB

    void* p1 = pool.allocate(256);
    void* p2 = pool.allocate(512);
    std::cout << "Allocated p1=" << p1 << " p2=" << p2 << "\n";

    pool.release(p1);
    std::cout << "Released p1\n";

    // allocate again
    void* p3 = pool.allocate(128);
    std::cout << "Allocated p3=" << p3 << "\n";

    // Normal shutdown
    std::cout << "Calling shutdown()\n";
    pool.shutdown();
    std::cout << "After shutdown()\n";

    // Call shutdown again to check idempotence
    std::cout << "Calling shutdown() again\n";
    pool.shutdown();
    std::cout << "After second shutdown()\n";

    return 0;
}
