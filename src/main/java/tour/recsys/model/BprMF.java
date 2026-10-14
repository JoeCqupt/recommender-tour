package tour.recsys.model;

import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * BPR-MF(Bayesian Personalized Ranking with Matrix Factorization)。
 *
 * <p>核心转变:不再预测"评几分",而是学习"用户喜欢的物品 > 用户没交互过的物品"。
 * 对每一个观测到的正样本 (u,i),随机采样一个该用户没见过的负样本 (u,j),让模型学会
 * x_ui 的分数高于 x_uj。
 *
 * <p>这是解决阶段一 MF "评分预测准、但 Top-N 排序弱"的关键一步。
 */
public final class BprMF implements Recommender, Ranker {

    private final int k;
    private final int epochs;
    private final double lr;
    private final double reg;
    private final int negSamples;

    private final Map<Long, Integer> userIdx = new HashMap<>();
    private final Map<Long, Integer> itemIdx = new HashMap<>();
    private long[] userIds;
    private long[] itemIds;
    private double[][] p;
    private double[][] q;
    private double[] bI;
    /** 用户在训练集里交互过的物品,负采样时用来排除。 */
    private final Map<Long, Set<Long>> userHistory = new HashMap<>();

    public BprMF(int k, int epochs, double lr, double reg, int negSamples) {
        this.k = k;
        this.epochs = epochs;
        this.lr = lr;
        this.reg = reg;
        this.negSamples = negSamples;
    }

    public void fit(Dataset train) {
        int nu = train.userCount();
        int ni = train.itemCount();
        userIds = new long[nu];
        itemIds = new long[ni];
        for (long u : train.users()) {
            userIdx.put(u, userIdx.size());
            userIds[userIdx.get(u)] = u;
        }
        for (long i : train.items()) {
            itemIdx.put(i, itemIdx.size());
            itemIds[itemIdx.get(i)] = i;
        }
        p = new double[nu][k];
        q = new double[ni][k];
        bI = new double[ni];

        Random rnd = new Random(42);
        for (int u = 0; u < nu; u++) {
            for (int f = 0; f < k; f++) {
                p[u][f] = rnd.nextGaussian() * 0.01;
            }
        }
        for (int i = 0; i < ni; i++) {
            for (int f = 0; f < k; f++) {
                q[i][f] = rnd.nextGaussian() * 0.01;
            }
        }

        // 记录每个用户的正样本集合,负采样时排除
        for (long u : train.users()) {
            Set<Long> hist = new HashSet<>();
            for (Rating r : train.ratingsOfUser(u)) {
                hist.add(r.movieId());
            }
            userHistory.put(u, hist);
        }

        List<Rating> positives = new ArrayList<>(train.ratings());
        int nItems = ni;
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(positives, rnd);
            for (Rating r : positives) {
                int u = userIdx.get(r.userId());
                int i = itemIdx.get(r.movieId());
                Set<Long> hist = userHistory.get(r.userId());
                for (int s = 0; s < negSamples; s++) {
                    // 均匀负采样:从用户没交互过的物品里随机挑
                    int j;
                    do {
                        j = rnd.nextInt(nItems);
                    } while (hist.contains(itemIds[j]));
                    update(u, i, j);
                }
            }
        }
    }

    private void update(int u, int i, int j) {
        double z = (bI[i] - bI[j]) + dotDiff(p[u], q[i], q[j]);
        // σ(-z) = 1 / (1 + exp(z)),表示"i 应该比 j 高,但实际差 z"的概率
        double sigmoidNeg = 1.0 / (1.0 + Math.exp(z));

        bI[i] += lr * (sigmoidNeg - reg * bI[i]);
        bI[j] -= lr * (sigmoidNeg - reg * bI[j]);

        double[] pu = p[u];
        double[] qi = q[i];
        double[] qj = q[j];
        for (int f = 0; f < k; f++) {
            double puf = pu[f];
            double qif = qi[f];
            double qjf = qj[f];
            // 梯度方向:让 p_u 靠近 q_i、远离 q_j
            pu[f] += lr * (sigmoidNeg * (qif - qjf) - reg * puf);
            qi[f] += lr * (sigmoidNeg * puf - reg * qif);
            qj[f] -= lr * (sigmoidNeg * puf - reg * qjf);
        }
    }

    private static double dotDiff(double[] pu, double[] qi, double[] qj) {
        double s = 0;
        for (int f = 0; f < pu.length; f++) {
            s += pu[f] * (qi[f] - qj[f]);
        }
        return s;
    }

    @Override
    public List<Long> recommend(long userId, int topN, Set<Long> exclude) {
        Integer u = userIdx.get(userId);
        if (u == null) {
            return List.of();
        }
        List<Map.Entry<Long, Double>> scored = new ArrayList<>(itemIds.length);
        for (int i = 0; i < itemIds.length; i++) {
            long item = itemIds[i];
            if (exclude != null && exclude.contains(item)) {
                continue;
            }
            double s = bI[i] + dot(p[u], q[i]);
            scored.add(Map.entry(item, s));
        }
        scored.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        List<Long> out = new ArrayList<>(topN);
        for (int i = 0; i < scored.size() && i < topN; i++) {
            out.add(scored.get(i).getKey());
        }
        return out;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int f = 0; f < a.length; f++) {
            s += a[f] * b[f];
        }
        return s;
    }

    /** 精排打分:给单个 (用户, 物品) 算偏好分。未见过用户/物品返回负无穷,排序时自然垫底。 */
    @Override
    public double score(long userId, long movieId) {
        Integer u = userIdx.get(userId);
        Integer i = itemIdx.get(movieId);
        if (u == null || i == null) {
            return Double.NEGATIVE_INFINITY;
        }
        return bI[i] + dot(p[u], q[i]);
    }

    public int dim() {
        return k;
    }

    public int itemCount() {
        return itemIds.length;
    }

    /** 物品 id 数组(按下标对齐 itemVector / itemBias)。用于导出向量。 */
    public long[] itemIds() {
        return itemIds.clone();
    }

    public long[] userIds() {
        return userIds.clone();
    }

    public double[] itemVector(int idx) {
        return q[idx];
    }

    public double itemBias(int idx) {
        return bI[idx];
    }

    public double[] userVector(long userId) {
        Integer u = userIdx.get(userId);
        return u == null ? null : p[u];
    }

    @Override
    public String name() {
        return "BPR-MF(k=" + k + ")";
    }
}
