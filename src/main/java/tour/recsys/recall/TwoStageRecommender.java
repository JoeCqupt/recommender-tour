package tour.recsys.recall;

import tour.recsys.model.Ranker;
import tour.recsys.model.Recommender;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 两级推荐:召回(Recall)→ 精排(Ranker)→ Top-K。
 *
 * <p>这是工业界推荐服务的主干结构。相比"模型直接给全库打分",
 * 它把昂贵的精排限制在几百个候选上,从而支持亿级物品。
 *
 * <p>它实现了 {@link Recommender},所以能直接塞进现有评估框架,和单级模型同表对比。
 */
public final class TwoStageRecommender implements Recommender {

    private final Recall recall;
    private final Ranker ranker;
    private final int candidateSize;

    public TwoStageRecommender(Recall recall, Ranker ranker, int candidateSize) {
        this.recall = recall;
        this.ranker = ranker;
        this.candidateSize = candidateSize;
    }

    @Override
    public List<Long> recommend(long userId, int topN, Set<Long> exclude) {
        List<Long> candidates = recall.recall(userId, candidateSize, exclude);
        if (candidates.isEmpty()) {
            return List.of();
        }
        List<Map.Entry<Long, Double>> scored = new ArrayList<>(candidates.size());
        for (long item : candidates) {
            scored.add(Map.entry(item, ranker.score(userId, item)));
        }
        scored.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        List<Long> out = new ArrayList<>(topN);
        for (int i = 0; i < scored.size() && i < topN; i++) {
            out.add(scored.get(i).getKey());
        }
        return out;
    }

    public int candidateSize() {
        return candidateSize;
    }

    @Override
    public String name() {
        return "两级(" + recall.name() + "→" + candidateSize + "→精排)";
    }
}
