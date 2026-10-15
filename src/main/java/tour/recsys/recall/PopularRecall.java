package tour.recsys.recall;

import tour.recsys.data.Dataset;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 热门召回:直接按物品热度取前 n 个。
 *
 * <p>它是召回的"地基"——个性化召回调参前必须先有一条热门兜底,
 * 否则新用户/冷门场景会召回空集。也是所有多路召回里必挂的一路。
 */
public final class PopularRecall implements Recall {

    private final List<Long> ranked;

    public PopularRecall(Dataset train) {
        this(train.itemsByPopularity());
    }

    public PopularRecall(List<Long> ranked) {
        this.ranked = ranked;
    }

    @Override
    public List<Long> recall(long userId, int n, Set<Long> exclude) {
        List<Long> out = new ArrayList<>(n);
        for (long item : ranked) {
            if (exclude != null && exclude.contains(item)) {
                continue;
            }
            out.add(item);
            if (out.size() >= n) {
                break;
            }
        }
        return out;
    }

    @Override
    public String name() {
        return "PopularRecall";
    }
}
