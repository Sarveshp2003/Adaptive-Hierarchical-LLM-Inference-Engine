#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Simple C API for embedding/integration. These are convenience stubs
// Java JNI will wrap these later.

void runtime_start();
void runtime_stop();
void runtime_request_layer(int layer_id);

#ifdef __cplusplus
}
#endif
