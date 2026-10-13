package tour.recsys.data;

import java.util.ArrayList;
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

    /** train:训练集;testByUser:userId -> 被留出的那条评分。 */
    public record Split(Dataset train, Map<Long, Rating> testByUser) {
    }

    /**
     * @param ds         完整数据集
     * @param minRatings 交互数太少的用户在倒排索引里噪声很大,低于该阈值直接剔除
     */
    public static Split leaveOneOut(Dataset ds, int minRatings) {
        List<Rating> train = new ArrayList<>();
        Map<Long, Rating> test = new HashMap<>();

        for (long userId : ds.users()) {
            List<Rating> rs = new ArrayList<>(ds.ratingsOfUser(userId));
            if (rs.size() < minRatings) {
                continue;
            }
            Rating latest = rs.get(0);
            for (Rating r : rs) {
                if (r.timestamp() > latest.timestamp()) {
                    latest = r;
                }
            }
            test.put(userId, latest);
            for (Rating r : rs) {
                if (r != latest) {
                    train.add(r);
                }
            }
        }
        return new Split(new Dataset(train, ds.movies()), test);
    }
}
