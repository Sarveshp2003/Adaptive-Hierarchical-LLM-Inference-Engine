#pragma once

#include <cstdint>

#if defined(__has_include)
#if __has_include(<jni.h>)
#include <jni.h>
#else
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

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_startRuntime(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_stopRuntime(JNIEnv *, jobject);
JNIEXPORT void JNICALL Java_com_adaptivellm_nativeengine_NativeEngine_requestLayer(JNIEnv *, jobject, jint layerId);

#ifdef __cplusplus
}
#endif
