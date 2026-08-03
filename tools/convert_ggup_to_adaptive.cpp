#include <iostream>
#include <filesystem>
#include <fstream>
#include "../loader/custom_model_loader.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: convert_ggup_to_adaptive <input.ggup|.gguf> [output.adaptive]" << std::endl;
        return 2;
    }
    std::filesystem::path inpath = argv[1];
    std::filesystem::path outpath = (argc > 2) ? std::filesystem::path(argv[2]) : inpath.parent_path() / "model.adaptive";

    try {
        auto archive = loader::CustomModelLoader::load(inpath.string());
        std::ofstream out(outpath.string());
        out << archive.name << "\n";
        out << archive.format << "\n";
        out << archive.version << "\n";
        out << archive.layers.size() << "\n";
        // Prepare external weights file and index
        std::filesystem::path weights_bin = outpath.parent_path() / (outpath.stem().string() + std::string(".weights"));
        std::filesystem::path weights_idx = weights_bin.string() + ".idx";
        std::ofstream wbin(weights_bin, std::ios::binary);
        std::ofstream widx(weights_idx);
        size_t current_offset = 0;
        for (const auto &l : archive.layers) {
            out << l.layer_id << "\n";
            out << l.name << "\n";
            // shape as comma list
            for (size_t i = 0; i < l.shape.size(); ++i) {
                if (i) out << ",";
                out << l.shape[i];
            }
            out << "\n";
            // write weights to external binary and index
            size_t float_count = l.weights.size();
            if (float_count > 0) {
                // write raw floats
                wbin.write(reinterpret_cast<const char*>(l.weights.data()), float_count * sizeof(float));
                // write index entry: layer_id offset_bytes float_count
                widx << l.layer_id << " " << current_offset << " " << float_count << "\n";
                current_offset += float_count * sizeof(float);
            } else {
                widx << l.layer_id << " " << current_offset << " " << 0 << "\n";
            }
            // in the adaptive archive keep an empty weights line to maintain format
            out << "\n";
        }
        wbin.close();
        widx.close();
        out.close();
        std::cout << "Wrote adaptive archive: " << outpath.string() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Conversion failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
