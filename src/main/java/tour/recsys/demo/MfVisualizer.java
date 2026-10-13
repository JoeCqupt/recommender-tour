package tour.recsys.demo;

import tour.recsys.data.CsvLoader;
import tour.recsys.data.Dataset;
import tour.recsys.data.Movie;
import tour.recsys.data.Rating;
import tour.recsys.model.MatrixFactorization;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 用 k=2 的矩阵分解把用户和电影画在一张二维图上。
 *
 * <p>输出:
 * <ul>
 *   <li>控制台:训练后的 μ、几个用户/电影的坐标、一个完整的"逐步预测"示例</li>
 *   <li>CSV 文件:{@code data/mf-2d-embedding.csv},供 Python 脚本画 SVG</li>
 * </ul>
 */
public final class MfVisualizer {

    private static final List<String> GENRES = List.of("Action", "Comedy", "Drama", "Romance", "Horror", "Animation");
    private static final List<String> GENRE_COLORS = List.of(
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#f781bf");

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "data/ml-latest-small";
        String outCsv = args.length > 1 ? args[1] : "data/mf-2d-embedding.csv";

        Dataset all = new Dataset(
                CsvLoader.loadRatings(dataDir + "/ratings.csv"),
                CsvLoader.loadMovies(dataDir + "/movies.csv"));

        System.out.println("=== 训练 k=2 的矩阵分解 ===");
        System.out.printf("数据集: %d 用户, %d 物品, %d 评分%n", all.userCount(), all.itemCount(), all.ratingCount());

        // k=2, epochs=100, lr=0.01, reg=0.02;维度低,多迭代让坐标更稳定
        MatrixFactorization mf = new MatrixFactorization(2, 100, 0.01, 0.02);
        mf.fit(all);

        double mu = mf.getMu();
        Map<Long, double[]> users = mf.userEmbeddings();
        Map<Long, double[]> items = mf.itemEmbeddings();
        Map<Long, Double> uBias = mf.userBiases();
        Map<Long, Double> iBias = mf.itemBiases();

        System.out.printf("%n训练完成。全局均值 μ = %.4f%n%n", mu);

        // 选电影:每个类型里被评次数最多的 3 部
        Map<String, String> genreColorMap = new LinkedHashMap<>();
        for (int i = 0; i < GENRES.size(); i++) {
            genreColorMap.put(GENRES.get(i), GENRE_COLORS.get(i));
        }
        List<Embedding> movieEmbeddings = selectMoviesByGenre(all, items, iBias, genreColorMap, 3);

        // 选用户:互动最多的前 30 人
        List<Embedding> userEmbeddings = selectTopUsers(all, users, uBias, 30);

        // 导出 CSV
        writeCsv(outCsv, movieEmbeddings, userEmbeddings);
        System.out.println("CSV 已导出: " + outCsv);
        System.out.printf("其中电影 %d 部,用户 %d 人%n%n", movieEmbeddings.size(), userEmbeddings.size());

        // 打印几个坐标示例
        System.out.println("=== 电影坐标示例(前 6 部) ===");
        for (int i = 0; i < Math.min(6, movieEmbeddings.size()); i++) {
            Embedding e = movieEmbeddings.get(i);
            System.out.printf("%-40s 坐标=(%7.4f, %7.4f)  偏置=%7.4f  类型=%s%n",
                    e.label(), e.x(), e.y(), e.bias(), e.tag());
        }

        System.out.println("\n=== 用户坐标示例(前 5 人) ===");
        for (int i = 0; i < Math.min(5, userEmbeddings.size()); i++) {
            Embedding e = userEmbeddings.get(i);
            System.out.printf("user %-5d  坐标=(%7.4f, %7.4f)  偏置=%7.4f%n",
                    e.id(), e.x(), e.y(), e.bias());
        }

        // 逐步预测:选一个活跃用户,挑一部他评过的动作片和一部爱情片
        long sampleUser = userEmbeddings.get(0).id();
        List<Rating> userRatings = all.ratingsOfUser(sampleUser);
        long actionMovie = findRatedMovieOfGenre(userRatings, all.movies(), "Action");
        long romanceMovie = findRatedMovieOfGenre(userRatings, all.movies(), "Romance");

        System.out.println("\n=== 一个完整预测的逐步拆解(user=" + sampleUser + ") ===");
        printPredictionStep(mu, sampleUser, actionMovie, all.movies(), users, items, uBias, iBias, userRatings);
        if (romanceMovie != -1 && romanceMovie != actionMovie) {
            printPredictionStep(mu, sampleUser, romanceMovie, all.movies(), users, items, uBias, iBias, userRatings);
        }

        System.out.println("\n=== 下一步 ===");
        System.out.println("运行: python3 scripts/plot_mf_2d.py " + outCsv + " docs/images/mf-2d.svg");
    }

    private record Embedding(long id, String label, double x, double y, double bias, String tag, String color) {
    }

    private static List<Embedding> selectMoviesByGenre(Dataset ds, Map<Long, double[]> items,
                                                       Map<Long, Double> iBias, Map<String, String> genreColors, int nPerGenre) {
        // 用 movies.csv 里列出的第一个 genre 作为主类型,避免"True Lies 被分到 Romance"这种反直觉情况
        Map<String, List<Long>> primaryGroups = new LinkedHashMap<>();
        for (Movie m : ds.movies().values()) {
            String primary = m.genres().split("\\|")[0];
            primaryGroups.computeIfAbsent(primary, k -> new ArrayList<>()).add(m.movieId());
        }

        List<Embedding> out = new ArrayList<>();
        for (String genre : genreColors.keySet()) {
            List<Long> candidates = primaryGroups.getOrDefault(genre, List.of());
            candidates.sort((a, b) -> Integer.compare(ds.ratingsOfItem(b).size(), ds.ratingsOfItem(a).size()));
            int taken = 0;
            for (long mid : candidates) {
                if (taken >= nPerGenre) {
                    break;
                }
                double[] v = items.get(mid);
                if (v == null) {
                    continue;
                }
                Movie m = ds.movies().get(mid);
                out.add(new Embedding(mid, m.title(), v[0], v[1], iBias.getOrDefault(mid, 0.0), genre, genreColors.get(genre)));
                taken++;
            }
        }
        return out;
    }

    private static List<Embedding> selectTopUsers(Dataset ds, Map<Long, double[]> users,
                                                  Map<Long, Double> uBias, int n) {
        List<Long> userIds = new ArrayList<>(ds.users());
        userIds.sort((a, b) -> Integer.compare(ds.ratingsOfUser(b).size(), ds.ratingsOfUser(a).size()));
        List<Embedding> out = new ArrayList<>();
        for (int i = 0; i < Math.min(n, userIds.size()); i++) {
            long uid = userIds.get(i);
            double[] v = users.get(uid);
            out.add(new Embedding(uid, "user" + uid, v[0], v[1], uBias.getOrDefault(uid, 0.0), "", "#333333"));
        }
        return out;
    }

    private static long findRatedMovieOfGenre(List<Rating> ratings, Map<Long, Movie> movies, String genre) {
        for (Rating r : ratings) {
            Movie m = movies.get(r.movieId());
            if (m != null && m.genres().contains(genre)) {
                return r.movieId();
            }
        }
        return -1;
    }

    private static void printPredictionStep(double mu, long uid, long mid, Map<Long, Movie> movies,
                                            Map<Long, double[]> users, Map<Long, double[]> items,
                                            Map<Long, Double> uBias, Map<Long, Double> iBias,
                                            List<Rating> userRatings) {
        double[] pu = users.get(uid);
        double[] qi = items.get(mid);
        double bu = uBias.getOrDefault(uid, 0.0);
        double bi = iBias.getOrDefault(mid, 0.0);
        double actual = userRatings.stream().filter(r -> r.movieId() == mid).findFirst().map(Rating::rating).orElse(Double.NaN);

        double dot = pu[0] * qi[0] + pu[1] * qi[1];
        double pred = mu + bu + bi + dot;

        System.out.printf("%n电影: %s%n", movies.get(mid).title());
        System.out.printf("  μ      = %.4f   (所有人打分的平均水平)%n", mu);
        System.out.printf("  b_u    = %+.4f  (user %d 的打分偏置)%n", bu, uid);
        System.out.printf("  b_i    = %+.4f  (这部电影的口碑偏置)%n", bi);
        System.out.printf("  p_u    = [%.4f, %.4f]%n", pu[0], pu[1]);
        System.out.printf("  q_i    = [%.4f, %.4f]%n", qi[0], qi[1]);
        System.out.printf("  p_u·q_i= %.4f   (口味匹配分:坐标点积)%n", dot);
        System.out.printf("  r̂      = %.4f + %+.4f + %+.4f + %.4f = %.4f%n", mu, bu, bi, dot, pred);
        if (!Double.isNaN(actual)) {
            System.out.printf("  真实评分 = %.1f,  误差 = %+.4f%n", actual, actual - pred);
        }
    }

    private static void writeCsv(String path, List<Embedding> movies, List<Embedding> users) throws IOException {
        try (PrintWriter pw = new PrintWriter(new FileWriter(path, StandardCharsets.UTF_8))) {
            pw.println("type,id,label,x,y,bias,tag,color");
            for (Embedding e : movies) {
                pw.printf("movie,%d,\"%s\",%.6f,%.6f,%.6f,%s,%s%n",
                        e.id(), e.label().replace("\"", "\"\""), e.x(), e.y(), e.bias(), e.tag(), e.color());
            }
            for (Embedding e : users) {
                pw.printf("user,%d,\"%s\",%.6f,%.6f,%.6f,%s,%s%n",
                        e.id(), e.label(), e.x(), e.y(), e.bias(), e.tag(), e.color());
            }
        }
    }
}
