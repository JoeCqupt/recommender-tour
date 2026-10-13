package tour.recsys.model;

import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Item-based Collaborative Filtering(物品协同过滤)。
 *
 * <p>核心思想:两个物品被同一批用户喜欢,它们就相似。给用户推荐时,拿他历史喜欢的物品,
 * 找到这些物品的相似物品,累加相似度作为候选分数。
 *
 * <p>两个工程要点:
 * <ul>
 *   <li>相似度用倒排索引累加:遍历每个用户评过的物品两两配对,而不是 O(items^2) 盲扫。</li>
 *   <li>对热门物品做惩罚:热门物品和谁都有共同用户,相似度虚高(similarity shrinkage)。</li>
 * </ul>
 */
public final class ItemCF implements Recommender {

    private final int topKNeighbors;
    private final double popAlpha;

    /** item -> (neighbor item -> similarity),只保留每个 item 最相似的 topKNeighbors 个。 */
    private final Map<Long, Map<Long, Double>> sim = new HashMap<>();
    /** userId -> 该用户在训练集里交互过的物品集合,推荐时作为种子。 */
    private final Map<Long, Set<Long>> userHistory = new HashMap<>();

    public ItemCF(Dataset train, int topKNeighbors, double popAlpha) {
        this.topKNeighbors = topKNeighbors;
        this.popAlpha = popAlpha;
        trainModel(train);
    }

    private void trainModel(Dataset train) {
        Map<Long, Set<Long>> itemUsers = new HashMap<>();
        for (long userId : train.users()) {
            Set<Long> hist = new HashSet<>();
            for (Rating r : train.ratingsOfUser(userId)) {
                itemUsers.computeIfAbsent(r.movieId(), k -> new HashSet<>()).add(userId);
                hist.add(r.movieId());
            }
            userHistory.put(userId, hist);
        }

        Map<Long, Double> itemNorm = new HashMap<>();
        for (Map.Entry<Long, Set<Long>> e : itemUsers.entrySet()) {
            itemNorm.put(e.getKey(), Math.sqrt(e.getValue().size()));
        }

        // 分子 |U_i ∩ U_j|:遍历每个用户评过的物品两两配对累加
        Map<Long, Map<Long, Integer>> coCount = new HashMap<>();
        for (Set<Long> hist : userHistory.values()) {
            List<Long> items = new ArrayList<>(hist);
            for (int a = 0; a < items.size(); a++) {
                for (int b = a + 1; b < items.size(); b++) {
                    long i = items.get(a);
                    long j = items.get(b);
                    coCount.computeIfAbsent(i, k -> new HashMap<>()).merge(j, 1, Integer::sum);
                    coCount.computeIfAbsent(j, k -> new HashMap<>()).merge(i, 1, Integer::sum);
                }
            }
        }

        // 余弦相似度 + 热门惩罚,取每个物品的 topK 邻居
        for (Map.Entry<Long, Map<Long, Integer>> e : coCount.entrySet()) {
            long i = e.getKey();
            double ni = itemNorm.get(i);
            List<Map.Entry<Long, Double>> scored = new ArrayList<>();
            for (Map.Entry<Long, Integer> c : e.getValue().entrySet()) {
                long j = c.getKey();
                double cos = c.getValue() / (ni * itemNorm.get(j));
                double shrink = 1.0 / (1.0 + popAlpha * (itemNorm.get(j) - 1.0));
                scored.add(Map.entry(j, cos * shrink));
            }
            scored.sort((p, q) -> Double.compare(q.getValue(), p.getValue()));
            Map<Long, Double> topN = new HashMap<>();
            for (int idx = 0; idx < Math.min(topKNeighbors, scored.size()); idx++) {
                topN.put(scored.get(idx).getKey(), scored.get(idx).getValue());
            }
            sim.put(i, topN);
        }
    }

    @Override
    public List<Long> recommend(long userId, int topN, Set<Long> exclude) {
        Set<Long> history = userHistory.get(userId);
        if (history == null || history.isEmpty()) {
            return List.of();
        }
        Map<Long, Double> score = new HashMap<>();
        for (long seed : history) {
            Map<Long, Double> neighbors = sim.get(seed);
            if (neighbors == null) {
                continue;
            }
            for (Map.Entry<Long, Double> nb : neighbors.entrySet()) {
                long cand = nb.getKey();
                if (exclude != null && exclude.contains(cand)) {
                    continue;
                }
                score.merge(cand, nb.getValue(), Double::sum);
            }
        }
        List<Map.Entry<Long, Double>> ranked = new ArrayList<>(score.entrySet());
        ranked.sort((p, q) -> Double.compare(q.getValue(), p.getValue()));
        List<Long> out = new ArrayList<>(topN);
        for (int i = 0; i < ranked.size() && i < topN; i++) {
            out.add(ranked.get(i).getKey());
        }
        return out;
    }

    @Override
    public String name() {
        return "ItemCF";
    }
}
