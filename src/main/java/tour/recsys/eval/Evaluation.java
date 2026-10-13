package tour.recsys.eval;

import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;
import tour.recsys.model.Recommender;

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
}
