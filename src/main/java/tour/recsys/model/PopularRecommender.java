package tour.recsys.model;

import tour.recsys.data.Dataset;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 热门 baseline:直接推荐被评分次数最多的物品。
 * 任何个性化算法的效果都必须超过它,否则说明模型没学到东西。
 */
public final class PopularRecommender implements Recommender {

    private final List<Long> ranked;

    public PopularRecommender(Dataset train) {
        this.ranked = train.itemsByPopularity();
    }

    @Override
    public List<Long> recommend(long userId, int topN, Set<Long> exclude) {
        List<Long> out = new ArrayList<>(topN);
        for (long item : ranked) {
            if (exclude != null && exclude.contains(item)) {
                continue;
            }
            out.add(item);
            if (out.size() >= topN) {
                break;
            }
        }
        return out;
    }

    @Override
    public String name() {
        return "Popular";
    }
}
