package tour.recsys.serve;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;

/**
 * 迷你推荐 HTTP 服务:零第三方依赖,直接用 JDK 内置 HttpServer。
 *
 * <p>接口:
 * <ul>
 *   <li>{@code GET /recommend?userId=42&n=10} —— 默认 FM 精排推荐;</li>
 *   <li>{@code GET /ab?userId=42&n=10} —— 按 userId 哈希分流到 FM 或 BPR 桶;</li>
 *   <li>{@code GET /health} —— 健康检查。</li>
 * </ul>
 */
public final class RecServer {

    private final RecService service;
    private final int port;

    public RecServer(RecService service, int port) {
        this.service = service;
        this.port = port;
    }

    public void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/recommend", new RecommendHandler(service, false));
        server.createContext("/ab", new RecommendHandler(service, true));
        server.createContext("/health", new HealthHandler());
        server.setExecutor(null);
        server.start();
        System.out.println("RecServer listening on http://localhost:" + port);
    }

    private static final class RecommendHandler implements HttpHandler {

        private final RecService service;
        private final boolean ab;

        RecommendHandler(RecService service, boolean ab) {
            this.service = service;
            this.ab = ab;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                send(exchange, 405, "{\"error\":\"method not allowed\"}");
                return;
            }
            String query = exchange.getRequestURI().getQuery();
            long userId = parseLongParam(query, "userId", -1);
            int n = (int) parseLongParam(query, "n", 10);
            if (userId < 0 || n <= 0 || n > 100) {
                send(exchange, 400, "{\"error\":\"invalid userId or n\"}");
                return;
            }

            String variant = ab ? service.abVariant(userId) : "fm";
            long t0 = System.nanoTime();
            List<Long> items = service.recommend(userId, n, variant);
            long millis = (System.nanoTime() - t0) / 1_000_000;

            StringBuilder sb = new StringBuilder();
            sb.append("{\"userId\":").append(userId)
              .append(",\"variant\":\"").append(variant).append("\"")
              .append(",\"latencyMs\":").append(millis)
              .append(",\"items\":[");
            for (int i = 0; i < items.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(items.get(i));
            }
            sb.append("]}");
            send(exchange, 200, sb.toString());
        }
    }

    private static final class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            send(exchange, 200, "{\"status\":\"ok\"}");
        }
    }

    private static void send(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=UTF-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static long parseLongParam(String query, String key, long defaultValue) {
        if (query == null || query.isEmpty()) {
            return defaultValue;
        }
        for (String pair : query.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0 && pair.substring(0, eq).equals(key)) {
                try {
                    return Long.parseLong(pair.substring(eq + 1));
                } catch (NumberFormatException e) {
                    return defaultValue;
                }
            }
        }
        return defaultValue;
    }

    public static void main(String[] args) throws IOException {
        String modelDir = args.length > 0 ? args[0] : "models";
        int port = args.length > 1 ? Integer.parseInt(args[1]) : 8080;
        RecService service = RecService.load(modelDir, 200);
        new RecServer(service, port).start();
    }
}
