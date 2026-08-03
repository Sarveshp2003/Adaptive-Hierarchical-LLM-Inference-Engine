package com.adaptivellm.nativeengine;

/**
 * Java interface to the native runtime bridge.
 */
public final class NativeEngine {
    static {
        try {
            System.loadLibrary("adaptive_engine");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native library not loaded: " + e.getMessage());
        }
    }

    public native void startRuntime();
    public native void stopRuntime();
    public native void requestLayer(int layerId);
}
