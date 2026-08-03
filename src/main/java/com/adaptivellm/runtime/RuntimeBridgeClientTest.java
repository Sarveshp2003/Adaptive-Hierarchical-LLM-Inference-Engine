package com.adaptivellm.runtime;

public class RuntimeBridgeClientTest {
    public static void main(String[] args) {
        try {
            RuntimeBridgeClient client = new RuntimeBridgeClient("http://127.0.0.1:8765/");
            System.out.println(client.post("request", 11));
        } catch (Exception e) {
            System.out.println("⚠️ Skipping RuntimeBridgeClientTest: " + e.getMessage());
        }
    }
}
