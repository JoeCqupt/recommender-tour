package tour.recsys;

import tour.recsys.data.CsvLoader;
import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;
import tour.recsys.data.TrainTestSplitter;
import tour.recsys.eval.Evaluation;
import tour.recsys.model.BprMF;
import tour.recsys.model.ItemCF;
import tour.recsys.model.PopularRecommender;
import tour.recsys.model.Ranker;
import tour.recsys.model.Recommender;
import tour.recsys.recall.MultiRecall;
import tour.recsys.recall.PopularRecall;
import tour.recsys.recall.Recall;
import tour.recsys.recall.RecommenderRecall;
import tour.recsys.recall.TwoStageRecommender;
import tour.recsys.recall.VectorRecall;
import tour.recsys.serve.EmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 阶段三主入口:两级推荐架构(召回 → 精排)。
 *
 * <p>四张表层层递进:
 * <ol>
 *   <li>表1 各路召回的召回率天花板;</li>
 *   <li>表2 精排=召回(同一套向量)时,候选集开多大其实不影响最终 Top-K —— 两级只省算力,不改质量;</li>
 *   <li>表3 精排≠召回(换成弱精排"热门分")时,候选集越大 HR 反而越低 —— 揭示"两级效果取决于召回与精排的搭配";</li>
 *   <li>表4 单级 vs 两级总账。</li>
 * </ol>
 *
 * <p>用法: java -cp target/classes tour.recsys.Stage3Main [dataDir] [topN]
 */
public final class Stage3Main {

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "data/ml-latest-small";
        int topN = args.length > 1 ? Integer.parseInt(args[1]) : 10;

        int channelN = 200;   // 每一路召回的候选数
        int mergedN = 600;    // 多路合并后的候选上限
        int[] sweep = {50, 100, 200, 500, 1000};

        Dataset all = new Dataset(
                CsvLoader.loadRatings(dataDir + "/ratings.csv"),
                CsvLoader.loadMovies(dataDir + "/movies.csv"));
        Dataset implicit = all.filterRatings(4.0);

        TrainTestSplitter.Split split = TrainTestSplitter.timeSplit(implicit, 0.2, 5);
        Dataset train = split.train();
        Map<Long, List<Rating>> test = split.testByUser();
        int testItems = test.values().stream().mapToInt(List::size).sum();
        System.out.printf("隐式正样本(>=4.0): %d 用户, %d 物品, %d 条交互%n",
                implicit.userCount(), implicit.itemCount(), implicit.ratingCount());
        System.out.printf("切分(time-split, 每用户交互>=5, 测试比例 20%%): 训练 %d 条, 测试 %d 用户 / %d 条%n%n",
                train.ratingCount(), test.size(), testItems);

        // ---- 训练 ----
        ItemCF itemCF = new ItemCF(train, 50, 0.0);
        BprMF bpr = new BprMF(100, 50, 0.02, 0.01, 5);
        long t0 = System.currentTimeMillis();
        bpr.fit(train);
        long bprMillis = System.currentTimeMillis() - t0;

        // 训练产物导出为"线上形态"(只有向量,没有训练逻辑)
        EmbeddingStore store = EmbeddingStore.fromBpr(bpr);

        // ---- 三路召回 ----
        Recall popularRecall = new PopularRecall(train);
        Recall itemRecall = new RecommenderRecall(itemCF);
        Recall vectorRecall = new VectorRecall(store);
        Recall multiRecall = new MultiRecall(
                List.of(popularRecall, itemRecall, vectorRecall), channelN);

        // ============ 表 1:召回率天花板 ============
        System.out.println("表1 召回率天花板 —— 候选集里命中的真实物品比例");
        System.out.println("------------------------------------------------------------------");
        System.out.printf("%-18s %13s %14s%n", "召回通道", "召回率", "平均候选数");
        printRecallRow(popularRecall, train, test, channelN);
        printRecallRow(itemRecall, train, test, channelN);
        printRecallRow(vectorRecall, train, test, channelN);
        printRecallRow(multiRecall, train, test, mergedN);
        System.out.println("------------------------------------------------------------------");
        System.out.printf("每个单路召回只取 %d 个候选;多路合并取 %d 个,天然含更多候选,召回率更高。%n", channelN, mergedN);
        System.out.println("教训:召回率是精排的天花板——候选集里没有的东西,精排永远排不出来。\n");

        // ============ 表 2:候选集大小扫描(精排=召回) ============
        System.out.println("表2 候选集开多大?(召回=向量,BPR 精排用同一套向量)");
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-12s %12s %9s %9s %11s%n", "候选数", "召回率", "HR@" + topN, "NDCG@" + topN, "评估耗时");
        for (int n : sweep) {
            Evaluation.RecallResult rr = Evaluation.evaluateRecall(vectorRecall, train, test, n);
            Recommender twoStage = new TwoStageRecommender(vectorRecall, bpr, n);
            long s0 = System.currentTimeMillis();
            Evaluation.RankResult r = Evaluation.evaluateRanking(twoStage, train, test, topN);
            long ms = System.currentTimeMillis() - s0;
            System.out.printf("%-12d %11.1f%% %9.4f %9.4f %9dms%n",
                    n, rr.rate() * 100, r.hr(), r.ndcg(), ms);
        }
        System.out.println("--------------------------------------------------------------------");
        System.out.println("注意 HR 几乎不变!原因:召回和精排用的是同一套向量,精排给候选打的排序 =");
        System.out.println("召回时的顺序,所以候选集大小只影响召回率与耗时,不影响最终 Top-K。\n");

        // ============ 表 3:换个"和召回不一样"的精排 ============
        // 用"热门分"当精排(一个与向量召回完全不同的排序信号)
        List<Long> popOrder = train.itemsByPopularity();
        Map<Long, Integer> popRank = new HashMap<>();
        for (int i = 0; i < popOrder.size(); i++) {
            popRank.put(popOrder.get(i), i);
        }
        Ranker popRanker = (u, item) -> {
            Integer r = popRank.get(item);
            return r == null ? -1_000_000 : -r;
        };

        System.out.println("表3 把精排换成'热门分'(和向量召回不一致的弱精排)");
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-12s %12s %9s %9s %11s%n", "候选数", "召回率", "HR@" + topN, "NDCG@" + topN, "评估耗时");
        for (int n : sweep) {
            Evaluation.RecallResult rr = Evaluation.evaluateRecall(vectorRecall, train, test, n);
            Recommender twoStage = new TwoStageRecommender(vectorRecall, popRanker, n);
            long s0 = System.currentTimeMillis();
            Evaluation.RankResult r = Evaluation.evaluateRanking(twoStage, train, test, topN);
            long ms = System.currentTimeMillis() - s0;
            System.out.printf("%-12d %11.1f%% %9.4f %9.4f %9dms%n",
                    n, rr.rate() * 100, r.hr(), r.ndcg(), ms);
        }
        System.out.println("--------------------------------------------------------------------");
        System.out.println("这次 HR 反而随候选数下降:候选池越大越接近全库,'热门精排'就越无力做个性化,");
        System.out.println("n=1000 时跌回 Popular 基线的 0.0295 —— 相当于两级退化成纯推热门。");
        System.out.println("这说明:召回与精排是'接力',两棒要匹配。");
        System.out.println("  · 弱精排:靠小而准的候选集来兜个性化(候选越小越好);");
        System.out.println("  · 强精排:需要大候选池把用户真实喜欢的物品捞进来才排得出(候选越大越好)。");
        System.out.println("所以'两级一定更好'是错觉——精排比召回强,两级才有效;否则只是白绕一圈。\n");

        // ============ 表 4:单级 vs 两级 ============
        Recommender popular = new PopularRecommender(train);
        Recommender twoStageMulti = new TwoStageRecommender(multiRecall, bpr, mergedN);
        Recommender twoStageVector = new TwoStageRecommender(vectorRecall, bpr, channelN);

        System.out.printf("表4 单级 vs 两级 @K=%d%n", topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-32s %9s %9s %8s %10s%n", "方案", "HR@" + topN, "NDCG@" + topN, "覆盖率", "评估耗时");
        printRow(popular, train, test, topN);
        printRow(itemCF, train, test, topN);
        printRow(bpr, train, test, topN); // 单级:BPR 直接给全库打分
        printRow(twoStageMulti, train, test, topN);
        printRow(twoStageVector, train, test, topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("BPR 训练耗时: %dms%n", bprMillis);
        System.out.println("本数据集物品才 6千+,两级省下的算力看不明显;物品上亿时,这就是能不能上线的分水岭。");
        System.out.println("下一步(阶段四):把精排换成'真正不同的模型'——带用户/物品/上下文交叉特征的 CTR 估计,");
        System.out.println("让两级不再冗余。这正是工业推荐系统效果提升的主战场。");
    }

    private static void printRecallRow(Recall recall, Dataset train, Map<Long, List<Rating>> test, int n) {
        Evaluation.RecallResult r = Evaluation.evaluateRecall(recall, train, test, n);
        System.out.printf("%-18s %12.1f%% %14.1f%n", recall.name(), r.rate() * 100, r.avgCandidates());
    }

    private static void printRow(Recommender rec, Dataset train, Map<Long, List<Rating>> test, int k) {
        long t0 = System.currentTimeMillis();
        Evaluation.RankResult r = Evaluation.evaluateRanking(rec, train, test, k);
        long millis = System.currentTimeMillis() - t0;
        System.out.printf("%-32s %9.4f %9.4f %7.1f%% %8dms%n",
                rec.name(), r.hr(), r.ndcg(), r.coverage() * 100, millis);
    }
}
