package tour.recsys;

import tour.recsys.data.CsvLoader;
import tour.recsys.data.Dataset;
import tour.recsys.data.TrainTestSplitter;
import tour.recsys.model.BprMF;
import tour.recsys.model.FmRanker;
import tour.recsys.serve.RecService;

import java.util.List;

/**
 * 阶段五主入口:模型导出 + 在线服务的最小闭环。
 *
 * <p>它做三件事:
 * <ol>
 *   <li>像阶段四一样训练 BPR 召回向量和 FM 精排;</li>
 *   <li>把训练产物({@code EmbeddingStore}、{@code FmRanker}、热门列表)导出到目录;</li>
 *   <li>从目录加载回来,用 {@link RecService} 做推荐,验证"离线训练 → 在线 serving"打通。</li>
 * </ol>
 *
 * <p>用法:
 * <pre>
 *   # 1. 训练并导出模型
 *   java -cp target/classes tour.recsys.Stage5Main data/ml-latest-small models
 *
 *   # 2. 启动 HTTP 服务
 *   java -cp target/classes tour.recsys.serve.RecServer models 8080
 *
 *   # 3. 请求推荐
 *   curl "http://localhost:8080/recommend?userId=1&n=10"
 *   curl "http://localhost:8080/ab?userId=1&n=10"
 * </pre>
 */
public final class Stage5Main {

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "data/ml-latest-small";
        String modelDir = args.length > 1 ? args[1] : "models";
        int candidateSize = 200;

        Dataset all = new Dataset(
                CsvLoader.loadRatings(dataDir + "/ratings.csv"),
                CsvLoader.loadMovies(dataDir + "/movies.csv"));
        Dataset implicit = all.filterRatings(4.0);

        TrainTestSplitter.Split split = TrainTestSplitter.timeSplit(implicit, 0.2, 5);
        Dataset train = split.train();
        System.out.printf("训练集: %d 用户, %d 物品, %d 条交互%n",
                train.userCount(), train.itemCount(), train.ratingCount());

        // 1. 训练
        BprMF bpr = new BprMF(100, 50, 0.02, 0.01, 5);
        long t0 = System.currentTimeMillis();
        bpr.fit(train);
        System.out.printf("BPR 训练耗时 %dms%n", System.currentTimeMillis() - t0);

        FmRanker fm = new FmRanker(32, 60, 0.01, 0.01, 10, true);
        t0 = System.currentTimeMillis();
        fm.fit(train);
        System.out.printf("FM 训练耗时 %dms; 特征空间 %d 维, k=%d%n%n",
                System.currentTimeMillis() - t0, fm.numFeatures(), fm.dim());

        // 2. 导出
        RecService.save(modelDir,
                tour.recsys.serve.EmbeddingStore.fromBpr(bpr),
                fm,
                train.itemsByPopularity());
        System.out.println("模型已导出到: " + modelDir);

        // 3. 加载并验证
        RecService service = RecService.load(modelDir, candidateSize);

        long testUser = train.users().iterator().next();
        System.out.printf("%n为用户 %d 推荐 Top-10 (FM):%n", testUser);
        List<Long> fmRecs = service.recommend(testUser, 10);
        System.out.println(fmRecs);

        String variant = service.abVariant(testUser);
        System.out.printf("A/B 分流桶: %s, 同用户推荐:%n", variant);
        List<Long> abRecs = service.recommend(testUser, 10, variant);
        System.out.println(abRecs);

        // 冷启动用户(训练集里没出现过)
        long coldUser = 999_999_999L;
        System.out.printf("%n冷启动用户 %d 的热门兜底 Top-10:%n", coldUser);
        List<Long> coldRecs = service.recommend(coldUser, 10);
        System.out.println(coldRecs);

        System.out.println("\n下一步: java -cp target/classes tour.recsys.serve.RecServer " + modelDir + " 8080");
    }
}
