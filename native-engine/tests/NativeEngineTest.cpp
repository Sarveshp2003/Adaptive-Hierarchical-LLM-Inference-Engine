#include "NativeEngine.h"

#include <iostream>


int main()
{

    std::cout
        << "Native Engine Test\n";


    NativeEngine engine;


    engine.initialize(0);


    engine.addLayer(512);


    // Tensor shapes are [batch, seq, hidden]
    Tensor input(
        {1, 1, 512},
        DataType::FP32
    );


    Tensor output(
        {1, 1, 512},
        DataType::FP32
    );


    input.allocateCPU();
    input.allocateGPU();

    output.allocateCPU();
    output.allocateGPU();


    std::cerr << "NativeEngineTest: input.rank=" << input.rank() << " input.shape=";
    for(auto s : input.shape()) std::cerr << s << ",";
    std::cerr << std::endl;

    engine.executeLayer(

        0,

        input,

        output

    );


    engine.shutdown();



    std::cout
        << "ENGINE TEST PASSED\n";


    return 0;

}