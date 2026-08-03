#include "jni_interface.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: runtime_bridge_cli <start|stop|request> [layer_id]\n";
        return 2;
    }

    std::string cmd = argv[1];
    if (cmd == "start") {
        runtime_start();
        return 0;
    }
    if (cmd == "stop") {
        runtime_stop();
        return 0;
    }
    if (cmd == "request") {
        int layer = argc > 2 ? std::stoi(argv[2]) : 0;
        runtime_request_layer(layer);
        return 0;
    }

    std::cerr << "unknown command: " << cmd << "\n";
    return 2;
}
