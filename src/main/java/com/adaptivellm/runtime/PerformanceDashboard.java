package com.adaptivellm.runtime;

import com.sun.net.httpserver.HttpServer;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.Executors;

public class PerformanceDashboard {

    public static void main(String[] args) throws IOException {
        int port = 8080;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/status", new StatusHandler());
        server.createContext("/dashboard", new DashboardHandler());
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("PerformanceDashboard running on http://localhost:" + port + "/dashboard");
    }

    private static String loadStatus() {
        try {
            byte[] bytes = Files.readAllBytes(Paths.get("CURRENT_STATUS.md"));
            return new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
        } catch (IOException e) {
            return "CURRENT_STATUS.md not found: " + e.getMessage();
        }
    }

    static class StatusHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            String body = loadStatus();
            byte[] resp = body.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "text/markdown; charset=utf-8");
            exchange.sendResponseHeaders(200, resp.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(resp);
            }
        }
    }

    static class DashboardHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1);
                return;
            }
            String status = loadStatus();
            String html = "<!doctype html><html><head><meta charset='utf-8'><title>Performance Dashboard</title>" +
                    "<meta name='viewport' content='width=device-width,initial-scale=1'>" +
                    "<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;}pre{white-space:pre-wrap;background:#f6f8fa;padding:12px;border:1px solid #ddd;}</style>" +
                    "</head><body><h1>Performance Dashboard</h1>" +
                    "<p><a href='/status'>View raw status</a></p>" +
                    "<pre>" + escapeHtml(status) + "</pre></body></html>";
            byte[] resp = html.getBytes(java.nio.charset.StandardCharsets.UTF_8);
            exchange.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
            exchange.sendResponseHeaders(200, resp.length);
            try (OutputStream os = exchange.getResponseBody()) {
                os.write(resp);
            }
        }
    }

    private static String escapeHtml(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) {
            switch (c) {
                case '&': sb.append("&amp;"); break;
                case '<': sb.append("&lt;"); break;
                case '>': sb.append("&gt;"); break;
                case '"': sb.append("&quot;"); break;
                default: sb.append(c);
            }
        }
        return sb.toString();
    }
}
