package tour.recsys.recall;

import tour.recsys.serve.EmbeddingStore;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * 向量召回:把用户向量当查询,找点积最高的 n 个物品向量。
 *
 * <p>这是"把 MF/BPR 训练出的 embedding 拿来召回"的最小实现——<b>暴力全扫</b>:
 * 逐个物品算 u·q + b_i,用一个小顶堆维护 Top-N。
 *
 * <p>复杂度 O(物品数 × k)。它很准,但不快——物品上亿时就要换成 ANN 近似索引
 * (IVF/LSH/HNSW),用一点点召回率损失换数量级的加速。见 docs 阶段三章节。
 *
 * <p>工程细节:用"大小为 n 的小顶堆"一次扫描求出 Top-N,是处理流式候选的标准做法,
 * 复杂度 O(物品数 × log n),比"全排序再截断"的 O(物品数 × log 物品数)更省。
 */
public final class VectorRecall implements Recall {

    private final EmbeddingStore store;

    public VectorRecall(EmbeddingStore store) {
        this.store = store;
    }

    @Override
    public List<Long> recall(long userId, int n, Set<Long> exclude) {
        double[] u = store.userVector(userId);
        if (u == null) {
            return List.of();
        }
        int ni = store.itemCount();
        // 小顶堆:堆顶是当前 Top-N 里分数最低的,便于淘汰
        PriorityQueue<long[]> heap = new PriorityQueue<>((a, b) -> Double.compare(
                Double.longBitsToDouble(a[1]), Double.longBitsToDouble(b[1])));
        for (int i = 0; i < ni; i++) {
            long item = store.itemId(i);
            if (exclude != null && exclude.contains(item)) {
                continue;
            }
            double s = store.score(u, i);
            if (heap.size() < n) {
                heap.offer(new long[]{item, Double.doubleToLongBits(s)});
            } else if (s > Double.longBitsToDouble(heap.peek()[1])) {
                heap.poll();
                heap.offer(new long[]{item, Double.doubleToLongBits(s)});
            }
        }
        List<long[]> all = new ArrayList<>(heap);
        all.sort((a, b) -> Double.compare(
                Double.longBitsToDouble(b[1]), Double.longBitsToDouble(a[1])));
        List<Long> out = new ArrayList<>(all.size());
        for (long[] e : all) {
            out.add(e[0]);
        }
        return out;
    }

    @Override
    public String name() {
        return "VectorRecall";
    }
}
