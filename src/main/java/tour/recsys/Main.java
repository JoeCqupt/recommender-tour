package tour.recsys;

import tour.recsys.data.CsvLoader;
import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;
import tour.recsys.data.TrainTestSplitter;
import tour.recsys.eval.Evaluation;
import tour.recsys.eval.Metrics;
import tour.recsys.model.ItemCF;
import tour.recsys.model.MatrixFactorization;
import tour.recsys.model.PopularRecommender;
import tour.recsys.model.Recommender;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 阶段一主入口:加载 MovieLens -> leave-one-out 切分 -> 训练三个模型 -> 离线评估对比。
 *
 * <p>用法: java -cp target/classes tour.recsys.Main [dataDir] [topN]
 */
public final class Main {

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "data/ml-latest-small";
        int topN = args.length > 1 ? Integer.parseInt(args[1]) : 10;

        Dataset all = new Dataset(
                CsvLoader.loadRatings(dataDir + "/ratings.csv"),
                CsvLoader.loadMovies(dataDir + "/movies.csv"));

        System.out.printf("数据集: %d 用户, %d 物品, %d 评分, 全局均值 %.3f%n",
                all.userCount(), all.itemCount(), all.ratingCount(), all.globalMean());

        TrainTestSplitter.Split split = TrainTestSplitter.leaveOneOut(all, 5);
        Dataset train = split.train();
        Map<Long, Rating> test = split.testByUser();
        System.out.printf("切分(leave-one-out, 每用户交互>=5): 训练 %d 条, 测试 %d 用户%n%n",
                train.ratingCount(), test.size());

        // 1) 热门 baseline
        Recommender popular = new PopularRecommender(train);

        // 2) ItemCF:topK 邻居 = 50,热门惩罚 alpha = 0(见 README:把 alpha 调到 >0 反而会摧毁召回)
        Recommender itemCF = new ItemCF(train, 50, 0.0);

        // 3) 带偏置矩阵分解(k=100, epochs=50, lr=0.01, reg=0.02)
        MatrixFactorization mf = new MatrixFactorization(100, 50, 0.01, 0.02);
        long t0 = System.currentTimeMillis();
        mf.fit(train);
        long mfMillis = System.currentTimeMillis() - t0;

        System.out.printf("评估指标 @K=%d%n", topN);
        System.out.println("-----------------------------------------------------------");
        System.out.printf("%-12s %10s %10s %10s%n", "模型", "HR@" + topN, "NDCG@" + topN, "覆盖率");
        printRow(popular, train, test, topN);
        printRow(itemCF, train, test, topN);
        printRow(mf, train, test, topN);
        System.out.println("-----------------------------------------------------------");

        // MF 的评分预测质量(RMSE):和排序指标一起看,体会"准"与"靠前"的区别
        List<double[]> pa = new ArrayList<>();
        for (Map.Entry<Long, Rating> e : test.entrySet()) {
            pa.add(new double[]{mf.predict(e.getKey(), e.getValue().movieId()), e.getValue().rating()});
        }
        System.out.printf("%nMF 评分预测 RMSE = %.4f  (训练耗时 %d ms)%n", Metrics.rmse(pa), mfMillis);

        // 给一个用户看看推荐了什么,直观感受两个模型的差异
        long sampleUser = test.keySet().iterator().next();
        System.out.printf("%n用户 %d 的推荐样例(去掉了训练集已看过的电影):%n", sampleUser);
        Set<Long> seen = train.itemsOfUser(sampleUser);
        System.out.println("  真实留出: " + train.titleOf(test.get(sampleUser).movieId()));
        printRecs("ItemCF", itemCF, train, sampleUser, seen, topN);
        printRecs("MF", mf, train, sampleUser, seen, topN);
    }

    private static void printRow(Recommender rec, Dataset train, Map<Long, Rating> test, int k) {
        Evaluation.RankResult r = Evaluation.evaluateRanking(rec, train, test, k);
        System.out.printf("%-12s %10.4f %10.4f %9.1f%%%n",
                rec.name(), r.hr(), r.ndcg(), r.coverage() * 100);
    }

    private static void printRecs(String label, Recommender rec, Dataset train,
                                  long user, Set<Long> seen, int topN) {
        List<Long> recs = rec.recommend(user, topN, seen);
        System.out.println("  " + label + ":");
        for (int i = 0; i < recs.size(); i++) {
            System.out.printf("    %2d. %s%n", i + 1, train.titleOf(recs.get(i)));
        }
    }
}
