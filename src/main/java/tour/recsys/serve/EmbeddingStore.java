package tour.recsys.serve;

import tour.recsys.model.BprMF;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 离线训练产物的"线上形态":只保留用户/物品向量和物品偏置,不带任何训练逻辑。
 *
 * <p>为什么单独抽一层?因为真实系统里"训练"和"服务"是两件事:
 * 训练任务跑完把向量写进这里(或推给特征/向量服务),线上只做查表和点积。
 * 这一步是"离线训练 → 导出 → 在线 serving"的最小示范。
 *
 * <p>物品向量用数组按下标存储(连续内存,便于向量化/建 ANN 索引),
 * 用户向量用 Map(在线按 userId 随机查)。
 */
public final class EmbeddingStore {

    private final int dim;
    private final Map<Long, double[]> userVectors = new HashMap<>();
    private final long[] itemIds;
    private final double[][] itemVectors;
    private final double[] itemBias;

    private EmbeddingStore(int dim, long[] itemIds, double[][] itemVectors, double[] itemBias) {
        this.dim = dim;
        this.itemIds = itemIds;
        this.itemVectors = itemVectors;
        this.itemBias = itemBias;
    }

    /** 从训练好的 BPR-MF 导出向量,模拟"离线训练 → 导出"这一步。 */
    public static EmbeddingStore fromBpr(BprMF model) {
        int ni = model.itemCount();
        long[] itemIds = model.itemIds();
        double[][] itemVectors = new double[ni][];
        double[] itemBias = new double[ni];
        for (int i = 0; i < ni; i++) {
            itemVectors[i] = model.itemVector(i).clone();
            itemBias[i] = model.itemBias(i);
        }
        EmbeddingStore store = new EmbeddingStore(model.dim(), itemIds, itemVectors, itemBias);
        for (long u : model.userIds()) {
            store.userVectors.put(u, model.userVector(u).clone());
        }
        return store;
    }

    public int dim() {
        return dim;
    }

    public int itemCount() {
        return itemIds.length;
    }

    public long itemId(int idx) {
        return itemIds[idx];
    }

    public List<Long> itemIdList() {
        List<Long> out = new java.util.ArrayList<>(itemIds.length);
        for (long id : itemIds) {
            out.add(id);
        }
        return out;
    }

    public double[] userVector(long userId) {
        return userVectors.get(userId);
    }

    /** 单条物品打分:u·q + b_i。召回层逐个物品算它,精排层对候选算它。 */
    public double score(double[] userVector, int itemIdx) {
        double[] q = itemVectors[itemIdx];
        double s = itemBias[itemIdx];
        for (int f = 0; f < dim; f++) {
            s += userVector[f] * q[f];
        }
        return s;
    }
}
