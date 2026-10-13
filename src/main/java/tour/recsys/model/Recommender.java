package tour.recsys.model;

import java.util.List;
import java.util.Set;

/** 所有推荐算法的统一出口:给一个用户,返回 topN 个物品(按分数从高到低)。 */
public interface Recommender {

    /**
     * @param userId  目标用户
     * @param topN    返回条数
     * @param exclude 需要排除的物品(通常是用户在训练集中已交互过的),不应出现在结果里
     * @return 推荐的 movieId 列表,长度 <= topN
     */
    List<Long> recommend(long userId, int topN, Set<Long> exclude);

    default String name() {
        return getClass().getSimpleName();
    }
}
