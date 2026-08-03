package com.adaptivellm.runtime;

import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class RuntimeBridgeClient {
    private final String baseUrl;

    public RuntimeBridgeClient(String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
    }

    public String post(String action, int layerId) throws Exception {
        String payload = String.format("{\"action\":\"%s\",\"layer_id\":%d}", action, layerId);
        URL url = new URL(baseUrl);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(payload.getBytes(StandardCharsets.UTF_8));
        }
        int code = conn.getResponseCode();
        byte[] body = conn.getInputStream().readAllBytes();
        return new String(body, StandardCharsets.UTF_8);
    }
}
