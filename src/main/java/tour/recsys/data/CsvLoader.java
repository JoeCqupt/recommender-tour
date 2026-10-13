package tour.recsys.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 读取 MovieLens 的 ratings.csv 与 movies.csv。 */
public final class CsvLoader {

    private CsvLoader() {
    }

    /** ratings.csv 列:userId,movieId,rating,timestamp。字段里不含逗号,可直接切分。 */
    public static List<Rating> loadRatings(String path) throws IOException {
        List<Rating> out = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path, StandardCharsets.UTF_8))) {
            String line = br.readLine();
            if (line == null) {
                return out;
            }
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                String[] f = line.split(",");
                out.add(new Rating(
                        Long.parseLong(f[0]),
                        Long.parseLong(f[1]),
                        Double.parseDouble(f[2]),
                        Long.parseLong(f[3])));
            }
        }
        return out;
    }

    /** movies.csv 列:movieId,title,genres。title 含逗号,按引号规则解析。 */
    public static Map<Long, Movie> loadMovies(String path) throws IOException {
        Map<Long, Movie> out = new LinkedHashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path, StandardCharsets.UTF_8))) {
            br.readLine();
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) {
                    continue;
                }
                List<String> f = parseCsvLine(line);
                long id = Long.parseLong(f.get(0));
                out.put(id, new Movie(id, f.get(1), f.get(2)));
            }
        }
        return out;
    }

    /** 手写 CSV 解析:movies.csv 片名形如 "Toy Story (1995)" 甚至含逗号,必须处理引号包裹与转义。 */
    static List<String> parseCsvLine(String line) {
        List<String> out = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        boolean inQuotes = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (inQuotes) {
                if (c == '"') {
                    if (i + 1 < line.length() && line.charAt(i + 1) == '"') {
                        sb.append('"');
                        i++;
                    } else {
                        inQuotes = false;
                    }
                } else {
                    sb.append(c);
                }
            } else if (c == '"') {
                inQuotes = true;
            } else if (c == ',') {
                out.add(sb.toString());
                sb.setLength(0);
            } else {
                sb.append(c);
            }
        }
        out.add(sb.toString());
        return out;
    }
}
