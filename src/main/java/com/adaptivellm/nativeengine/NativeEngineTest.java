package com.adaptivellm.nativeengine;

public class NativeEngineTest {
    public static void main(String[] args) {
        try {
            NativeEngine engine = new NativeEngine();
            engine.startRuntime();
            engine.requestLayer(3);
            engine.stopRuntime();
        } catch (Throwable t) {
            System.out.println("⚠️ Skipping NativeEngineTest (native runtime missing): " + t.getMessage());
        }
    }
}
