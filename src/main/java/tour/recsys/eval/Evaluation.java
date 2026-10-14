package tour.recsys.eval;

import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;
import tour.recsys.model.Recommender;
import tour.recsys.recall.Recall;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 离线评估流程:对每个测试用户,去掉他训练集见过的物品后取 Top-K,和留出的真实物品比对。 */
public final class Evaluation {

    private Evaluation() {
    }

    /**
     * @param coverage 被推荐过的不同物品数 / 训练集物品总数 —— 衡量"是不是总推那几部热门片"
     */
    public record RankResult(double hr, double ndcg, double coverage, int testCount, int userCount) {
    }

    public static RankResult evaluateRanking(Recommender rec, Dataset train,
                                             Map<Long, List<Rating>> testByUser, int k) {
        double hr = 0;
        double ndcg = 0;
        Set<Long> recommended = new HashSet<>();
        int n = 0;
        int users = 0;
        for (Map.Entry<Long, List<Rating>> e : testByUser.entrySet()) {
            long user = e.getKey();
            users++;
            Set<Long> exclude = train.itemsOfUser(user);
            List<Long> recs = rec.recommend(user, k, exclude);
            recommended.addAll(recs);
            for (Rating r : e.getValue()) {
                long truth = r.movieId();
                hr += Metrics.hitAtK(recs, truth, k);
                ndcg += Metrics.ndcgAtK(recs, truth, k);
                n++;
            }
        }
        double cov = train.itemCount() == 0 ? 0 : (double) recommended.size() / train.itemCount();
        return new RankResult(hr / n, ndcg / n, cov, n, users);
    }

    /** 召回评估结果:召回率 = 真实测试物品落在候选集里的比例;avgCandidates = 平均候选数。 */
    public record RecallResult(double rate, double avgCandidates) {
    }

    /**
     * 评估召回层:真实测试物品有多大概率出现在召回候选里。
     *
     * <p>这是"召回率天花板"——精排只能在候选集里排序,召回没捞到的东西,精排永远排不出来。
     * 所以召回率必须显著高于最终的 HR@K,否则瓶颈在召回而不是排序。
     */
    public static RecallResult evaluateRecall(Recall recall, Dataset train,
                                              Map<Long, List<Rating>> testByUser, int n) {
        int total = 0;
        int hit = 0;
        long candSum = 0;
        int users = 0;
        for (Map.Entry<Long, List<Rating>> e : testByUser.entrySet()) {
            long user = e.getKey();
            users++;
            Set<Long> exclude = train.itemsOfUser(user);
            List<Long> cands = recall.recall(user, n, exclude);
            Set<Long> candSet = new HashSet<>(cands);
            candSum += cands.size();
            for (Rating r : e.getValue()) {
                total++;
                if (candSet.contains(r.movieId())) {
                    hit++;
                }
            }
        }
        double rate = total == 0 ? 0 : (double) hit / total;
        double avg = users == 0 ? 0 : (double) candSum / users;
        return new RecallResult(rate, avg);
    }
}
