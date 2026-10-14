package tour.recsys.model;

/**
 * 精排打分器:给单个 (用户, 物品) 返回一个偏好分,分数越高越该推荐。
 *
 * <p>召回层负责"从全库捞候选",精排层负责"给候选精细打分"——两者职责不同,
 * 所以用两个接口分开。{@link Recommender} 是端到端出口,{@link Ranker} 是精排这一级的抽象。
 */
public interface Ranker {

    double score(long userId, long movieId);
}
