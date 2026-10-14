package tour.recsys.recall;

import tour.recsys.model.Recommender;

import java.util.List;
import java.util.Set;

/**
 * 适配器:把任意一个 {@link Recommender} 当成一路召回用。
 *
 * <p>它点破一件事——<b>"单级模型"和"一路召回"只是同一个东西的两种视角</b>。
 * 一个能吐 Top-N 的模型(ItemCF、热门、甚至别的 MF),把它的 Top-N 拿来当候选集,
 * 它就是一路召回。阶段一/二写的模型不用改写,直接挂进多路召回里。
 */
public final class RecommenderRecall implements Recall {

    private final Recommender delegate;

    public RecommenderRecall(Recommender delegate) {
        this.delegate = delegate;
    }

    @Override
    public List<Long> recall(long userId, int n, Set<Long> exclude) {
        return delegate.recommend(userId, n, exclude);
    }

    @Override
    public String name() {
        return delegate.name() + "召回";
    }
}
