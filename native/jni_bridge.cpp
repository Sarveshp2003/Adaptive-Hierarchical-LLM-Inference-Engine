#include "jni_bridge.h"
#include "jni_interface.h"
#include <iostream>

#if defined(__has_include)
#if __has_include(<jni.h>)
#include <jni.h>
#else
// Provide a minimal JNI-compatible declaration if jni.h is unavailable.
#ifndef JNIEXPORT
#define JNIEXPORT
#endif
#ifndef JNICALL
#define JNICALL
#endif
#ifndef JNI_VERSION_1_8
#define JNI_VERSION_1_8 0x00010008
#endif
using jint = int;
struct JNIEnv_;
using JNIEnv = JNIEnv_;
struct _jobject;
using jobject = _jobject*;
#endif
#endif

extern "C" {

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_startRuntime(JNIEnv*, jobject) {
    runtime_start();
}

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_stopRuntime(JNIEnv*, jobject) {
    runtime_stop();
}

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_requestLayer(JNIEnv*, jobject, jint layerId) {
    runtime_request_layer(layerId);
}

}
