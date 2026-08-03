#include "NativeEngine.h"
#include "RuntimeMemory.h"
#include "CUDAStream.h"
#include "../../loader/gguf_reader.h"
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>
#include <vector>
#include <filesystem>

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: run_gguf_inference <model.gguf> [num_layers] [poolMB] [hiddenDim]" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    int runLayers = 4;
    if(argc >= 3) runLayers = std::stoi(argv[2]);

    NativeEngine engine;
    try {
        unsigned long long gpuPoolBytes = 0x40000000ULL; // default 1GB
        if(argc >= 4) {
            long long poolMB = std::stoll(argv[3]);
            if(poolMB > 0) gpuPoolBytes = static_cast<unsigned long long>(poolMB) * 1024ULL * 1024ULL;
        }
        engine.initialize(0, gpuPoolBytes);
    } catch(const std::exception &e) {
        std::cerr << "Engine init failed: " << e.what() << std::endl;
        return 1;
    }

    // Use NativeEngine's GGUF loader to stream per-layer tensors (avoids materializing full weights in RAM)
    if(!engine.loadModelGGUF(modelPath)) {
        std::cerr << "Failed to open GGUF via engine loader" << std::endl;
        engine.shutdown();
        return 1;
    }

    int numLayers = engine.getNumLayers();
    int hidden = engine.getHiddenDim();
    // Allow explicit override of hidden dimension via CLI arg 4 (if provided)
    if(argc >= 5) {
        try {
            hidden = std::stoi(argv[4]);
            std::cout << "Overriding hidden dim via CLI: " << hidden << std::endl;
        } catch(...) {
            std::cerr << "Invalid hiddenDim argument, using guessed value " << hidden << std::endl;
        }
    }
    std::cout << "Archive layers=" << numLayers << " guessed_hidden=" << engine.getHiddenDim() << " effective_hidden=" << hidden << std::endl;

    int toRun = std::min(numLayers, runLayers);

    // Ensure metrics output directory exists (relative to build/Release executable)
    std::filesystem::path metricsDir = std::filesystem::path("..") / "out";
    std::error_code ec;
    std::filesystem::create_directories(metricsDir, ec);
    if(ec) std::cerr << "Warning: failed to create metrics dir: " << ec.message() << std::endl;
    std::ofstream out((metricsDir / "gguf_inference_metrics.txt").string(), std::ios::out);
    if(!out.is_open()) std::cerr << "Warning: failed to open metrics file for writing\n";
    out << "Model=" << modelPath << "\n";
    out << "archive_layers_total=" << numLayers << " guessed_hidden=" << hidden << "\n";

    // Stream layer-by-layer using engine loader. Each load will allocate pinned buffers and copy needed tensors to GPU.
    for(int layerId=0; layerId<toRun; ++layerId) {
        auto s = std::chrono::high_resolution_clock::now();
        bool ok = engine.loadLayerFromGGUF(layerId);
        auto e = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(e-s).count();
        out << "layer_streamed="<<layerId<<" time_ms="<<ms<<" success="<<(ok?1:0)<<"\n";
        out.flush();
        std::cout << "Layer "<<layerId<<" streamed in "<<ms<<" ms (ok="<<(ok?1:0)<<")"<<std::endl;
        // If streaming failed, skip this layer and continue
        if(!ok) {
            std::cerr << "Warning: streaming layer "<<layerId<<" failed, skipping execution"<<std::endl;
            // Ensure any partial allocations for this layer are released
            try { engine.releaseLayer(layerId); } catch(...) {}
            continue;
        }

        // Add runtime layer wrapper (transformer structure)
        try {
            engine.addLayer(hidden);
        } catch(const std::exception &e) {
            std::cerr << "Error adding runtime layer "<<layerId<<": "<<e.what()<<" - freeing layer and continuing"<<std::endl;
            try { engine.releaseLayer(layerId); } catch(...) {}
            continue;
        }

        // Execute this layer immediately to avoid holding multiple layers in GPU memory.
        try {
            Tensor input({1,1,hidden}, DataType::FP32);
            Tensor output({1,1,hidden}, DataType::FP32);
            input.allocateCPU(); input.allocateGPU();
            output.allocateCPU(); output.allocateGPU();
            // fill input with small values
            for(size_t i=0;i<input.elements();++i) input.cpu()[i] = 0.01f * (i%10);
            input.upload();

            out << "\nForward runs for layer " << layerId << ":\n";
            int runs = 2; // shorter per-layer runs during streaming test
            for(int r=0;r<runs;++r) {
                auto s = std::chrono::high_resolution_clock::now();
                try {
                    engine.executeLayer(static_cast<int>(layerId), input, output);
                } catch(const std::exception &e) {
                    std::cerr << "Error executing layer "<<layerId<<" run "<<r<<": "<<e.what()<<"\n";
                    break;
                }
                auto e = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(e-s).count();
                out << "layer_exec="<<layerId<<" run="<<r<<" time_ms="<<ms<<"\n";
                std::cout<<"Layer exec "<<layerId<<" run "<<r<<" took "<<ms<<" ms\n";
                std::swap(input, output);
            }
            out.flush();
        } catch(const std::exception &e) {
            std::cerr << "Allocation/Execution error for layer "<<layerId<<": "<<e.what()<<" - releasing and continuing"<<std::endl;
        } catch(...) {
            std::cerr << "Unknown error during allocation/execution for layer "<<layerId<<" - releasing and continuing"<<std::endl;
        }

        // Remove runtime layer to free its GPU resources and release loader-side tensors for this layer
        try { engine.removeLastLayer(); } catch(...) {}
        try { engine.releaseLayer(layerId); } catch(...) {}
    }

    std::cout<<"Streaming and single-layer execution complete.\n";
    // Cleanup
    try {
        engine.shutdown();
    } catch (const std::exception &e) {
        std::cerr << "Engine shutdown error: " << e.what() << std::endl;
    } catch(...) {
        std::cerr << "Engine shutdown unknown error" << std::endl;
    }
    out.flush();
    out.close();
    std::cout<<"Inference pass complete. Metrics in ..\\out\\gguf_inference_metrics.txt\n";
    return 0;
}
