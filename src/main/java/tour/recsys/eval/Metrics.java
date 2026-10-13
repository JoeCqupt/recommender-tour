package tour.recsys.eval;

import java.util.List;

/**
 * 离线评估指标。
 *
 * <p>Top-N 推荐与"评分预测"是两种任务,指标也不同:
 * <ul>
 *   <li>排序任务看 HR@K / NDCG@K —— 关心的东西排进前 K 了吗、排得靠前吗</li>
 *   <li>评分任务看 RMSE —— 预测分和真实分差多少</li>
 * </ul>
 */
public final class Metrics {

    private Metrics() {
    }

    /** 命中位置(1-based);未命中返回 -1。 */
    public static int rankOf(List<Long> recs, long truth) {
        for (int i = 0; i < recs.size(); i++) {
            if (recs.get(i) == truth) {
                return i + 1;
            }
        }
        return -1;
    }

    /**
     * Hit Ratio@K:命中返回 1,否则 0。
     * leave-one-out 下每个用户只有 1 个相关物品,此时 Recall@K 与 HR@K 等价。
     */
    public static double hitAtK(List<Long> recs, long truth, int k) {
        int rank = rankOf(recs, truth);
        return (rank >= 1 && rank <= k) ? 1.0 : 0.0;
    }

    /**
     * NDCG@K:单个相关物品时 IDCG=1,故等于 1/log2(rank+1)(rank 命中且 <=K),否则 0。
     * 它比 HR 更严格:排第 1 和第 5 都算命中,但前者得分更高。
     */
    public static double ndcgAtK(List<Long> recs, long truth, int k) {
        int rank = rankOf(recs, truth);
        if (rank < 1 || rank > k) {
            return 0.0;
        }
        return 1.0 / (Math.log(rank + 1) / Math.log(2));
    }

    public static double rmse(List<double[]> predActual) {
        if (predActual.isEmpty()) {
            return Double.NaN;
        }
        double s = 0;
        for (double[] pa : predActual) {
            double d = pa[0] - pa[1];
            s += d * d;
        }
        return Math.sqrt(s / predActual.size());
    }
}
