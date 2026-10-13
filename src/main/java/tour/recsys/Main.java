package tour.recsys;

import tour.recsys.data.CsvLoader;
import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;
import tour.recsys.data.TrainTestSplitter;
import tour.recsys.eval.Evaluation;
import tour.recsys.model.BprMF;
import tour.recsys.model.ItemCF;
import tour.recsys.model.PopularRecommender;
import tour.recsys.model.Recommender;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 阶段二主入口:隐式反馈 + BPR-MF。
 *
 * <p>切分从 leave-one-out 改为按时间比例切分(训练 80%,测试 20%),每个用户有多个测试物品,
 * 更适合评估排序模型。对比 Popular、ItemCF、评分 MF、BPR-MF。
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

        // 阶段二:把显式评分转成隐式反馈 —— 只有 >=4.0 才算"喜欢"
        Dataset implicit = all.filterRatings(4.0);

        System.out.printf("原始评分: %d 用户, %d 物品, %d 条评分%n",
                all.userCount(), all.itemCount(), all.ratingCount());
        System.out.printf("隐式正样本(>=4.0): %d 用户, %d 物品, %d 条交互%n",
                implicit.userCount(), implicit.itemCount(), implicit.ratingCount());

        // 按时间比例切分:每用户前 80% 训练,后 20% 测试
        TrainTestSplitter.Split split = TrainTestSplitter.timeSplit(implicit, 0.2, 5);
        Dataset train = split.train();
        Map<Long, List<Rating>> test = split.testByUser();
        int testItems = test.values().stream().mapToInt(List::size).sum();
        System.out.printf("切分(time-split, 每用户交互>=5, 测试比例 20%%): 训练 %d 条, 测试 %d 用户 / %d 条%n%n",
                train.ratingCount(), test.size(), testItems);

        // 1) 热门 baseline
        Recommender popular = new PopularRecommender(train);

        // 2) ItemCF
        Recommender itemCF = new ItemCF(train, 50, 0.0);

        // 3) BPR-MF:专门学习"正样本 > 负样本"的排序模型
        BprMF bpr = new BprMF(100, 50, 0.02, 0.01, 5);
        long t0 = System.currentTimeMillis();
        bpr.fit(train);
        long bprMillis = System.currentTimeMillis() - t0;

        System.out.printf("评估指标 @K=%d (隐式反馈, 多个测试物品取平均)%n", topN);
        System.out.println("---------------------------------------------------------------");
        System.out.printf("%-14s %10s %10s %10s %10s%n", "模型", "HR@" + topN, "NDCG@" + topN, "覆盖率", "训练耗时");
        printRow(popular, train, test, topN, -1);
        printRow(itemCF, train, test, topN, -1);
        printRow(bpr, train, test, topN, bprMillis);
        System.out.println("---------------------------------------------------------------");

        // 给一个用户看推荐样例
        long sampleUser = test.keySet().iterator().next();
        System.out.printf("%n用户 %d 的推荐样例(去掉了训练集已看过的电影):%n", sampleUser);
        Set<Long> seen = train.itemsOfUser(sampleUser);
        System.out.println("  真实测试: " + titles(train, test.get(sampleUser)));
        printRecs("ItemCF", itemCF, train, sampleUser, seen, topN);
        printRecs("BPR-MF", bpr, train, sampleUser, seen, topN);
    }

    private static void printRow(Recommender rec, Dataset train, Map<Long, List<Rating>> test, int k, long millis) {
        Evaluation.RankResult r = Evaluation.evaluateRanking(rec, train, test, k);
        String time = millis < 0 ? "-" : millis + "ms";
        System.out.printf("%-14s %10.4f %10.4f %9.1f%%%10s%n",
                rec.name(), r.hr(), r.ndcg(), r.coverage() * 100, time);
    }

    private static String titles(Dataset train, List<Rating> rs) {
        StringBuilder sb = new StringBuilder();
        for (Rating r : rs) {
            if (!sb.isEmpty()) {
                sb.append("; ");
            }
            sb.append(train.titleOf(r.movieId()));
        }
        return sb.toString();
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
