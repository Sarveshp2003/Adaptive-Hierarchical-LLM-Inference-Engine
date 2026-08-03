#include "jni_interface.h"
#include <iostream>

// Minimal stubs. Real implementation will manage the runtime instance and threads.
void runtime_start() {
    std::cout << "runtime_start() called\n";
}

void runtime_stop() {
    std::cout << "runtime_stop() called\n";
}

void runtime_request_layer(int layer_id) {
    std::cout << "runtime_request_layer(" << layer_id << ") called\n";
}
