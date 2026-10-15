package tour.recsys.serve;

import tour.recsys.model.Ranker;

/**
 * 用 {@link EmbeddingStore} 里的 BPR 向量做精排打分。
 *
 * <p>它把 EmbeddingStore 从"召回工具"扩展成"召回 + 精排"两用:线上可以直接拿它
 * 和 FM 精排做 A/B 对比,看特征交叉到底有没有带来收益。
 */
public final class BprRanker implements Ranker {

    private final EmbeddingStore store;

    public BprRanker(EmbeddingStore store) {
        this.store = store;
    }

    @Override
    public double score(long userId, long movieId) {
        double[] u = store.userVector(userId);
        int idx = store.itemIndex(movieId);
        if (u == null || idx < 0) {
            return Double.NEGATIVE_INFINITY;
        }
        return store.score(u, idx);
    }
}
