package tour.recsys.recall;

import java.util.List;
import java.util.Set;

/**
 * 召回层接口:从全库物品里快速捞出一批候选,交给精排层。
 *
 * <p>召回和排序的目标不同:
 * <ul>
 *   <li>召回要"快 + 全":宁可多捞一些不相关的,也不能漏掉真正该推的(召回率天花板)。</li>
 *   <li>排序要"准":在候选集里精细打分,决定最终顺序。</li>
 * </ul>
 * 因此用一个独立的接口表达召回,和 {@link tour.recsys.model.Recommender} 区分开。
 */
public interface Recall {

    /**
     * @param userId  目标用户
     * @param n       召回多少条候选
     * @param exclude 需要排除的物品(通常是用户训练集已交互过的)
     * @return 候选 movieId 列表,长度 <= n
     */
    List<Long> recall(long userId, int n, Set<Long> exclude);

    default String name() {
        return getClass().getSimpleName();
    }
}
