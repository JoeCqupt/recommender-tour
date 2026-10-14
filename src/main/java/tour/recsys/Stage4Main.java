package tour.recsys;

import tour.recsys.data.CsvLoader;
import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;
import tour.recsys.data.TrainTestSplitter;
import tour.recsys.eval.Evaluation;
import tour.recsys.model.BprMF;
import tour.recsys.model.FmRanker;
import tour.recsys.model.Ranker;
import tour.recsys.model.Recommender;
import tour.recsys.recall.MultiRecall;
import tour.recsys.recall.PopularRecall;
import tour.recsys.recall.Recall;
import tour.recsys.recall.TwoStageRecommender;
import tour.recsys.recall.VectorRecall;
import tour.recsys.serve.EmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 阶段四主入口:CTR 预估精排(特征交叉)。
 *
 * <p>阶段三的教训是"精排必须比召回强,两级才有效"。阶段三的"精排"其实复用了召回的
 * 同一套向量,所以两级是冗余的。阶段四换成真正不同的精排信号——把 (用户 × 物品 × 品类)
 * 铺成稀疏特征、用 FM 学交叉——精排才可能把召回候选重新排好、把漏在候选里的真实物品往前提。
 *
 * <p>四张表:
 * <ol>
 *   <li>表1 精排器横评:同一路向量召回,换不同精排(热门分 / BPR 向量分 / FM);</li>
 *   <li>表2 特征消融:FM 关掉品类交叉 vs 打开,看"多一类交叉特征"值多少;</li>
 *   <li>表3 召回路的影响:好精排在不同召回路下的表现;</li>
 *   <li>表4 单级 vs 两级总账。</li>
 * </ol>
 *
 * <p>用法: java -cp target/classes tour.recsys.Stage4Main [dataDir] [topN]
 */
public final class Stage4Main {

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "data/ml-latest-small";
        int topN = args.length > 1 ? Integer.parseInt(args[1]) : 10;

        int channelN = 200;   // 每一路召回的候选数
        int mergedN = 600;    // 多路合并后的候选上限

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

        // ---- 训练:召回用 BPR 向量;精排用 FM ----
        BprMF bpr = new BprMF(100, 50, 0.02, 0.01, 5);
        long t0 = System.currentTimeMillis();
        bpr.fit(train);
        long bprMillis = System.currentTimeMillis() - t0;
        EmbeddingStore store = EmbeddingStore.fromBpr(bpr);

        FmRanker fm = new FmRanker(32, 60, 0.01, 0.01, 10, true);
        long t1 = System.currentTimeMillis();
        fm.fit(train);
        long fmMillis = System.currentTimeMillis() - t1;
        System.out.printf("训练耗时:BPR-MF %dms,FM %dms;FM 特征空间 %d 维(k=%d, 负采样=10)%n%n",
                bprMillis, fmMillis, fm.numFeatures(), fm.dim());

        Recall vectorRecall = new VectorRecall(store);
        Recall popularRecall = new PopularRecall(train);
        Recall multiRecall = new MultiRecall(List.of(popularRecall, vectorRecall), channelN);

        // 一个"和向量召回不一致"的弱精排(热门分),作为对照底线
        List<Long> popOrder = train.itemsByPopularity();
        Map<Long, Integer> popRank = new HashMap<>();
        for (int i = 0; i < popOrder.size(); i++) {
            popRank.put(popOrder.get(i), i);
        }
        Ranker popRanker = (u, item) -> {
            Integer r = popRank.get(item);
            return r == null ? -1_000_000 : -r;
        };

        // ============ 表 1:精排器横评(同一路向量召回 200) ============
        System.out.printf("表1 精排器横评 —— 同一路向量召回(候选 %d),换不同精排 @K=%d%n", channelN, topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-30s %9s %9s %8s %10s%n", "精排器", "HR@" + topN, "NDCG@" + topN, "覆盖率", "评估耗时");
        printRow("热门分", new TwoStageRecommender(vectorRecall, popRanker, channelN), train, test, topN);
        printRow("BPR 向量分", new TwoStageRecommender(vectorRecall, bpr, channelN), train, test, topN);
        printRow("FM(品类交叉)", new TwoStageRecommender(vectorRecall, fm, channelN), train, test, topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.println("BPR 精排用的是和召回同一套向量,排序 ≈ 召回顺序,两级冗余;热门精排完全丢了个性化;");
        System.out.println("FM 精排引入'用户 × 物品 × 品类'的交叉特征,是独立于召回的信号,HR/NDCG 都超过了 BPR,");
        System.out.println("但覆盖率更低 —— 它更偏好大众口味,推的东西更集中。效果与多样性是要权衡的。\n");

        // ============ 表 2:特征消融 ============
        System.out.println("表2 特征消融 —— '多一类交叉特征'到底值多少?(同为 FM 精排)");
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-30s %9s %9s %8s %10s%n", "FM 变体", "HR@" + topN, "NDCG@" + topN, "覆盖率", "评估耗时");
        FmRanker fmNoGenre = new FmRanker(32, 60, 0.01, 0.01, 10, false);
        fmNoGenre.fit(train);
        printRow("FM(仅用户×物品)", new TwoStageRecommender(vectorRecall, fmNoGenre, channelN), train, test, topN);
        printRow("FM(加品类交叉)", new TwoStageRecommender(vectorRecall, fm, channelN), train, test, topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.println("只有 (用户×物品) 交叉时,FM 退化成一个带线性偏置的 MF;加上品类交叉后,模型能利用");
        System.out.println("'这个用户爱看什么类型'的先验,把候选里同类型的物品往前提 —— 这就是特征工程的价值。\n");

        // ============ 表 3:召回路的影响 ============
        System.out.println("表3 好精排需要好召回 —— 同一 FM 精排,换不同召回路");
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-30s %9s %9s %8s %10s%n", "召回 → FM 精排", "HR@" + topN, "NDCG@" + topN, "覆盖率", "评估耗时");
        printRow("热门召回 → FM", new TwoStageRecommender(popularRecall, fm, channelN), train, test, topN);
        printRow("向量召回 → FM", new TwoStageRecommender(vectorRecall, fm, channelN), train, test, topN);
        printRow("多路召回 → FM", new TwoStageRecommender(multiRecall, fm, mergedN), train, test, topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.println("精排再强也救不回召回没捞到的物品:召回率是天花板。多路召回把上限抬高,精排才有空间发挥。\n");

        // ============ 表 4:单级 vs 两级 ============
        System.out.printf("表4 单级 vs 两级 @K=%d%n", topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.printf("%-32s %9s %9s %8s %10s%n", "方案", "HR@" + topN, "NDCG@" + topN, "覆盖率", "评估耗时");
        printRow("BPR 单级(全库打分)", bpr, train, test, topN);
        printRow("向量召回 → BPR 精排", new TwoStageRecommender(vectorRecall, bpr, channelN), train, test, topN);
        printRow("多路召回 → FM 精排", new TwoStageRecommender(multiRecall, fm, mergedN), train, test, topN);
        System.out.println("--------------------------------------------------------------------");
        System.out.println("阶段四结论:两级架构的收益,来自'召回捞得全' + '精排用不同的、更细的特征重排'。");
        System.out.println("下一步(阶段五):把 FM 这样的手工交叉,换成神经网络自动学交叉(DeepFM),并引入序列建模(DIN)。");
    }

    private static void printRow(String label, Recommender rec, Dataset train, Map<Long, List<Rating>> test, int k) {
        long t0 = System.currentTimeMillis();
        Evaluation.RankResult r = Evaluation.evaluateRanking(rec, train, test, k);
        long millis = System.currentTimeMillis() - t0;
        System.out.printf("%-30s %9.4f %9.4f %7.1f%% %8dms%n",
                label, r.hr(), r.ndcg(), r.coverage() * 100, millis);
    }
}
