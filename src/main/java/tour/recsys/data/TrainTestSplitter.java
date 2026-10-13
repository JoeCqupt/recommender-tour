package tour.recsys.data;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Leave-one-out 切分:每个用户按时间排序,留最新一条作为测试集,其余作为训练集。
 *
 * <p>为什么用 leave-one-out 而不是随机切分?Top-N 推荐评估的是"能不能把用户真正会喜欢的
 * 东西排进前 K",用最新一次交互当测试最贴近线上场景,也是该领域的标准做法。
 */
public final class TrainTestSplitter {

    /** train:训练集;testByUser:userId -> 被留出的评分列表。 */
    public record Split(Dataset train, Map<Long, List<Rating>> testByUser) {
    }

    /**
     * Leave-one-out:每个用户按时间排序,留最新一条作为测试,其余训练。
     *
     * <p>为什么用 leave-one-out 而不是随机切分?Top-N 推荐评估的是"能不能把用户真正会喜欢的
     * 东西排进前 K",用最新一次交互当测试最贴近线上场景,也是该领域的标准做法之一。
     *
     * @param ds         完整数据集
     * @param minRatings 交互数太少的用户在倒排索引里噪声很大,低于该阈值直接剔除
     */
    public static Split leaveOneOut(Dataset ds, int minRatings) {
        List<Rating> train = new ArrayList<>();
        Map<Long, List<Rating>> test = new HashMap<>();

        for (long userId : ds.users()) {
            List<Rating> rs = new ArrayList<>(ds.ratingsOfUser(userId));
            if (rs.size() < minRatings) {
                continue;
            }
            rs.sort(Comparator.comparingLong(Rating::timestamp));
            Rating latest = rs.get(rs.size() - 1);
            List<Rating> trainRs = new ArrayList<>(rs);
            trainRs.remove(latest);
            test.put(userId, List.of(latest));
            train.addAll(trainRs);
        }
        return new Split(new Dataset(train, ds.movies()), test);
    }

    /**
     * 按时间比例切分:每个用户的交互按时间排序,前 (1-testRatio) 训练,后 testRatio 测试。
     *
     * <p>这是隐式反馈/排序任务更常用的切法:测试集包含用户近期多个交互,能更稳定地估计
     * HR@K 和 NDCG@K;也避免了 leave-one-out 下每个用户只有 1 个测试样本的方差问题。
     *
     * @param ds             完整数据集
     * @param testRatio      每个用户留多少比例作为测试,例如 0.2
     * @param minInteractions 交互数太少的用户直接剔除
     */
    public static Split timeSplit(Dataset ds, double testRatio, int minInteractions) {
        List<Rating> train = new ArrayList<>();
        Map<Long, List<Rating>> test = new HashMap<>();

        for (long userId : ds.users()) {
            List<Rating> rs = new ArrayList<>(ds.ratingsOfUser(userId));
            if (rs.size() < minInteractions) {
                continue;
            }
            rs.sort(Comparator.comparingLong(Rating::timestamp));
            int nTest = Math.max(1, (int) Math.round(rs.size() * testRatio));
            int splitAt = rs.size() - nTest;
            train.addAll(rs.subList(0, splitAt));
            test.put(userId, new ArrayList<>(rs.subList(splitAt, rs.size())));
        }
        return new Split(new Dataset(train, ds.movies()), test);
    }
}
