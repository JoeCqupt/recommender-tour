package tour.recsys.model;

import tour.recsys.data.Dataset;
import tour.recsys.data.Rating;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * 带偏置的矩阵分解(Biased MF),用 SGD 训练。
 *
 * <p>预测式:r_hat = mu + b_u + b_i + p_u · q_i
 * <ul>
 *   <li>mu 全局均值,b_u 用户偏置(苛刻/宽容),b_i 物品偏置(叫好/叫座)</li>
 *   <li>p_u、q_i 是隐向量,维度 k,刻画"用户偏好"与"物品属性"的潜在匹配</li>
 * </ul>
 *
 * <p>它同时是"评分预测"模型(用 RMSE 评估)和"排序"模型(对所有未交互物品打分排 Top-N),
 * 正好让你对比"预测得准"和"排得靠前"是两回事。
 */
public final class MatrixFactorization implements Recommender {

    private static final double INIT_STD = 0.01;

    private final int k;
    private final int epochs;
    private final double lr;
    private final double reg;

    private double mu;
    private final Map<Long, Integer> userIdx = new HashMap<>();
    private final Map<Long, Integer> itemIdx = new HashMap<>();
    private long[] userIds;
    private long[] itemIds;
    private double[] bU;
    private double[] bI;
    private double[][] p;
    private double[][] q;
    /** userId -> 训练集里交互过的物品,排序时作为种子(此处仅用于 fallback 判断,排除交由调用方)。 */
    private final Map<Long, Set<Long>> userHistory = new HashMap<>();

    public MatrixFactorization(int k, int epochs, double lr, double reg) {
        this.k = k;
        this.epochs = epochs;
        this.lr = lr;
        this.reg = reg;
    }

    public void fit(Dataset train) {
        mu = train.globalMean();
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
        bU = new double[nu];
        bI = new double[ni];
        p = new double[nu][k];
        q = new double[ni][k];

        Random rnd = new Random(42);
        for (int u = 0; u < nu; u++) {
            for (int f = 0; f < k; f++) {
                p[u][f] = rnd.nextGaussian() * INIT_STD;
            }
        }
        for (int i = 0; i < ni; i++) {
            for (int f = 0; f < k; f++) {
                q[i][f] = rnd.nextGaussian() * INIT_STD;
            }
        }
        for (long u : train.users()) {
            userHistory.put(u, train.itemsOfUser(u));
        }

        List<Rating> data = new ArrayList<>(train.ratings());
        for (int epoch = 0; epoch < epochs; epoch++) {
            Collections.shuffle(data, rnd);
            for (Rating r : data) {
                int u = userIdx.get(r.userId());
                int i = itemIdx.get(r.movieId());
                double pred = mu + bU[u] + bI[i] + dot(p[u], q[i]);
                double e = r.rating() - pred;
                bU[u] += lr * (e - reg * bU[u]);
                bI[i] += lr * (e - reg * bI[i]);
                double[] pu = p[u];
                double[] qi = q[i];
                for (int f = 0; f < k; f++) {
                    double puf = pu[f];
                    double qif = qi[f];
                    pu[f] += lr * (e * qif - reg * puf);
                    qi[f] += lr * (e * puf - reg * qif);
                }
            }
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int f = 0; f < a.length; f++) {
            s += a[f] * b[f];
        }
        return s;
    }

    /** 预测用户对某物品的评分;训练集中没见过的用户/物品退化为全局均值 + 已见偏置。 */
    public double predict(long userId, long movieId) {
        Integer u = userIdx.get(userId);
        Integer i = itemIdx.get(movieId);
        if (u == null && i == null) {
            return mu;
        }
        double bu = u == null ? 0 : bU[u];
        double bi = i == null ? 0 : bI[i];
        double pq = (u == null || i == null) ? 0 : dot(p[u], q[i]);
        return mu + bu + bi + pq;
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
            double s = mu + bU[u] + bI[i] + dot(p[u], q[i]);
            scored.add(Map.entry(item, s));
        }
        scored.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        List<Long> out = new ArrayList<>(topN);
        for (int i = 0; i < scored.size() && i < topN; i++) {
            out.add(scored.get(i).getKey());
        }
        return out;
    }

    @Override
    public String name() {
        return "MF(k=" + k + ")";
    }

    /** 全局均值 μ,用于把预测值拉回整体评分水平。 */
    public double getMu() {
        return mu;
    }

    /** 每个用户的隐向量坐标(复制),key 是 userId。 */
    public Map<Long, double[]> userEmbeddings() {
        Map<Long, double[]> out = new LinkedHashMap<>(userIds.length);
        for (int i = 0; i < userIds.length; i++) {
            out.put(userIds[i], p[i].clone());
        }
        return out;
    }

    /** 每个物品的隐向量坐标(复制),key 是 movieId。 */
    public Map<Long, double[]> itemEmbeddings() {
        Map<Long, double[]> out = new LinkedHashMap<>(itemIds.length);
        for (int i = 0; i < itemIds.length; i++) {
            out.put(itemIds[i], q[i].clone());
        }
        return out;
    }

    /** 用户偏置 b_u。 */
    public Map<Long, Double> userBiases() {
        Map<Long, Double> out = new LinkedHashMap<>(userIds.length);
        for (int i = 0; i < userIds.length; i++) {
            out.put(userIds[i], bU[i]);
        }
        return out;
    }

    /** 物品偏置 b_i。 */
    public Map<Long, Double> itemBiases() {
        Map<Long, Double> out = new LinkedHashMap<>(itemIds.length);
        for (int i = 0; i < itemIds.length; i++) {
            out.put(itemIds[i], bI[i]);
        }
        return out;
    }
}
