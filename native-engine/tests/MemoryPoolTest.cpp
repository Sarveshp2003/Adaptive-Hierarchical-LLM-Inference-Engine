#include "GPUMemoryPool.h"

#include <iostream>



int main()

{

    std::cout
        << "GPU Memory Pool Test\n";



    GPUMemoryPool pool;



    pool.initialize(

        1024 * 1024 * 1024

    );



    void* a =
        pool.allocate(

            256 * 1024 * 1024

        );



    void* b =
        pool.allocate(

            128 * 1024 * 1024

        );



    if(a && b)
    {

        std::cout
            << "Allocation successful\n";

    }



    pool.release(a);

    pool.release(b);



    std::cout
        << "Memory released\n";


    pool.shutdown();



    return 0;

}