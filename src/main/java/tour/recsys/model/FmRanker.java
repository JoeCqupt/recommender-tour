package tour.recsys.model;

import tour.recsys.data.Dataset;
import tour.recsys.data.Movie;
import tour.recsys.data.Rating;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * 因子分解机(Factorization Machine),手写实现,用作两级架构的精排器。
 *
 * <p>为什么到阶段四要引入 FM?因为前几级的模型只知道"用户向量 × 物品向量"这一个交叉,
 * 而 CTR 预估的核心是<b>特征交叉</b>:用户 U × 物品 I × 品类 G × 时段……
 * 暴力做二阶交叉的参数是 O(特征数²),FM 用"每个特征配一个隐向量、交叉项的系数用两向量内积
 * 表示"把参数量压到 O(特征数 × k),还能对<b>从来没见过</b>的特征组合给出合理估计。
 *
 * <p>模型(取值为 1 的稀疏 one-hot / multi-hot 特征):
 * <pre>
 *   ŷ = w0 + Σ_i w_i x_i + Σ_{i&lt;j} ⟨v_i, v_j⟩ x_i x_j
 * </pre>
 * 这里 x_i ∈ {0,1},所以只对"激活的特征集合 S"求和:
 * <pre>
 *   ŷ = w0 + Σ_{f∈S} w_f + Σ_{f&lt;g, f,g∈S} ⟨v_f, v_g⟩
 * </pre>
 * 交叉项用完全平方恒等式简化,复杂度从 O(|S|²k) 降到 O(|S|k):
 * <pre>
 *   Σ_{f&lt;g} ⟨v_f, v_g⟩ = ½ ( ‖Σ_f v_f‖² - Σ_f ‖v_f‖² )
 * </pre>
 *
 * <p>特征设计(就是"把一条样本铺成 one-hot"):
 * <ul>
 *   <li>用户 one-hot:学"这个用户的偏好偏置";</li>
 *   <li>物品 one-hot:学"这个物品的受欢迎偏置";</li>
 *   <li>品类 multi-hot:一部片属于多个 genre,每个 genre 一个特征。</li>
 * </ul>
 * 用户特征与物品/品类特征之间的二阶交叉,就是 FM 自动学出来的"用户 × 物品""用户 × 品类"。
 * 注意:FM 里也含 ⟨v_用户, v_物品⟩ 这一项 —— 它把前面的 MF 收编成了自己的一个特例,再加交叉。
 *
 * <p>训练:CTR 二分类的 log loss,标签 1 = 正样本(隐式正反馈),0 = 负样本(采样得到)。
 * 这就是"隐式反馈 → CTR 预估"的桥:推荐里没有真实负例,用负采样造。
 */
public final class FmRanker implements Ranker {

    private final int k;
    private final int epochs;
    private final double lr;
    private final double reg;
    private final int negSamples;
    private final boolean useGenreCross; // 特征消融开关:关掉就只剩"用户×物品"的交叉

    private final Map<Long, Integer> userIdx = new HashMap<>();
    private final Map<Long, Integer> itemIdx = new HashMap<>();
    private final Map<String, Integer> genreIdx = new HashMap<>();
    private long[] userIds;
    private long[] itemIds;

    private int userOffset;   // 用户特征起始下标(0)
    private int itemOffset;   // 物品特征起始下标
    private int genreOffset;  // 品类特征起始下标
    private int numFeatures;

    private double w0;
    private double[] w;
    private double[][] v;

    /** 每个物品激活的特征下标(物品 one-hot [+ 品类 multi-hot]),训练/打分反复用,预计算。 */
    private int[][] itemFeatures;
    /** 用户在训练集里交互过的物品,负采样时排除。 */
    private final Map<Long, Set<Long>> userHistory = new HashMap<>();

    public FmRanker(int k, int epochs, double lr, double reg, int negSamples) {
        this(k, epochs, lr, reg, negSamples, true);
    }

    public FmRanker(int k, int epochs, double lr, double reg, int negSamples, boolean useGenreCross) {
        this.k = k;
        this.epochs = epochs;
        this.lr = lr;
        this.reg = reg;
        this.negSamples = negSamples;
        this.useGenreCross = useGenreCross;
    }

    /** 供 loadModel 使用的私有构造:只初始化服务所需字段,训练超参数置 0(不会再用)。 */
    private FmRanker(int k, boolean useGenreCross) {
        this.k = k;
        this.useGenreCross = useGenreCross;
        this.epochs = 0;
        this.lr = 0.0;
        this.reg = 0.0;
        this.negSamples = 0;
    }

    public void fit(Dataset train) {
        int nu = train.userCount();
        int ni = train.itemCount();

        userIds = new long[nu];
        for (long u : train.users()) {
            userIdx.put(u, userIdx.size());
            userIds[userIdx.get(u)] = u;
        }
        itemIds = new long[ni];
        for (long i : train.items()) {
            itemIdx.put(i, itemIdx.size());
            itemIds[itemIdx.get(i)] = i;
        }

        // 只在 train 里出现过的物品上收集品类,保证特征空间一致
        Map<Long, Movie> movies = train.movies();
        for (long i : train.items()) {
            Movie m = movies.get(i);
            if (m == null) {
                continue;
            }
            for (String g : m.genres().split("\\|")) {
                if (!g.isEmpty() && !genreIdx.containsKey(g)) {
                    genreIdx.put(g, genreIdx.size());
                }
            }
        }

        userOffset = 0;
        itemOffset = nu;
        genreOffset = nu + ni;
        numFeatures = nu + ni + genreIdx.size();

        w0 = 0.0;
        w = new double[numFeatures];
        v = new double[numFeatures][k];
        Random rnd = new Random(42);
        for (double[] row : v) {
            for (int f = 0; f < k; f++) {
                row[f] = rnd.nextGaussian() * 0.01;
            }
        }

        // 预计算每个物品的特征下标列表
        itemFeatures = new int[ni][];
        for (long i : train.items()) {
            int idx = itemIdx.get(i);
            Movie m = movies.get(i);
            List<Integer> feats = new ArrayList<>();
            feats.add(itemOffset + idx);
            if (useGenreCross && m != null) {
                for (String g : m.genres().split("\\|")) {
                    Integer gi = genreIdx.get(g);
                    if (gi != null) {
                        feats.add(genreOffset + gi);
                    }
                }
            }
            int[] arr = new int[feats.size()];
            for (int t = 0; t < arr.length; t++) {
                arr[t] = feats.get(t);
            }
            itemFeatures[idx] = arr;
        }

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
                long uid = r.userId();
                long iid = r.movieId();
                update(userIdx.get(uid), itemIdx.get(iid), 1.0);
                Set<Long> hist = userHistory.get(uid);
                for (int s = 0; s < negSamples; s++) {
                    long negItem;
                    do {
                        negItem = itemIds[rnd.nextInt(nItems)];
                    } while (hist.contains(negItem));
                    update(userIdx.get(uid), itemIdx.get(negItem), 0.0);
                }
            }
        }
    }

    /** 一次 SGD 更新:先算前向 ŷ,再按 log loss 的梯度回传 w0/w/v。 */
    private void update(int u, int i, double label) {
        int userFeat = userOffset + u;
        int[] iFeats = itemFeatures[i];

        // 激活特征集合 S = {用户特征} ∪ 物品特征列表
        int[] s = new int[iFeats.length + 1];
        s[0] = userFeat;
        System.arraycopy(iFeats, 0, s, 1, iFeats.length);

        double x = predict(s);
        double p = 1.0 / (1.0 + Math.exp(-x));
        double g = p - label; // log loss 对 ŷ 的梯度

        // 交叉项梯度需要 Σ_{g∈S} v_g,先算一次(必须用"旧值")
        double[] sum = new double[k];
        for (int f : s) {
            double[] vf = v[f];
            for (int d = 0; d < k; d++) {
                sum[d] += vf[d];
            }
        }

        w0 -= lr * (g + reg * w0);
        for (int f : s) {
            double[] vf = v[f];
            w[f] -= lr * (g + reg * w[f]);
            // ∂ŷ/∂v_f = Σ_{g∈S, g≠f} v_g = sum - v_f
            for (int d = 0; d < k; d++) {
                double vfd = vf[d];
                vf[d] -= lr * (g * (sum[d] - vfd) + reg * vfd);
            }
        }
    }

    private double predict(int[] s) {
        double x = w0;
        double[] sum = new double[k];
        double[] sumSq = new double[k];
        for (int f : s) {
            x += w[f];
            double[] vf = v[f];
            for (int d = 0; d < k; d++) {
                sum[d] += vf[d];
                sumSq[d] += vf[d] * vf[d];
            }
        }
        double cross = 0;
        for (int d = 0; d < k; d++) {
            cross += sum[d] * sum[d] - sumSq[d];
        }
        return x + 0.5 * cross;
    }

    /** 精排打分:未见过用户/物品返回负无穷,排序时自然垫底。 */
    @Override
    public double score(long userId, long movieId) {
        Integer u = userIdx.get(userId);
        Integer i = itemIdx.get(movieId);
        if (u == null || i == null) {
            return Double.NEGATIVE_INFINITY;
        }
        int[] iFeats = itemFeatures[i];
        int[] s = new int[iFeats.length + 1];
        s[0] = userOffset + u;
        System.arraycopy(iFeats, 0, s, 1, iFeats.length);
        return predict(s);
    }

    public int dim() {
        return k;
    }

    public int numFeatures() {
        return numFeatures;
    }

    public String name() {
        return "FM(k=" + k + (useGenreCross ? "" : ",无品类交叉") + ")";
    }

    /** 把训练好的 FM 模型序列化到输出流,用于"离线训练 → 导出模型"。 */
    public void saveModel(OutputStream out) throws IOException {
        DataOutputStream dos = new DataOutputStream(out);
        dos.writeInt(0x464D524B); // magic "FMRK"
        dos.writeInt(1);          // version
        dos.writeInt(k);
        dos.writeBoolean(useGenreCross);
        dos.writeInt(numFeatures);
        dos.writeInt(userOffset);
        dos.writeInt(itemOffset);
        dos.writeInt(genreOffset);
        dos.writeDouble(w0);

        dos.writeInt(userIds.length);
        for (long u : userIds) {
            dos.writeLong(u);
        }
        dos.writeInt(itemIds.length);
        for (long i : itemIds) {
            dos.writeLong(i);
        }

        dos.writeInt(genreIdx.size());
        for (Map.Entry<String, Integer> e : genreIdx.entrySet()) {
            dos.writeUTF(e.getKey());
            dos.writeInt(e.getValue());
        }

        for (int f = 0; f < numFeatures; f++) {
            dos.writeDouble(w[f]);
        }
        for (int f = 0; f < numFeatures; f++) {
            for (int d = 0; d < k; d++) {
                dos.writeDouble(v[f][d]);
            }
        }

        dos.writeInt(itemFeatures.length);
        for (int[] feats : itemFeatures) {
            dos.writeInt(feats.length);
            for (int f : feats) {
                dos.writeInt(f);
            }
        }
        dos.flush();
    }

    /** 从输入流反序列化 FM 模型,用于"在线服务加载模型"。 */
    public static FmRanker loadModel(InputStream in) throws IOException {
        DataInputStream dis = new DataInputStream(in);
        int magic = dis.readInt();
        if (magic != 0x464D524B) {
            throw new IOException("invalid FmRanker magic: " + Integer.toHexString(magic));
        }
        int version = dis.readInt();
        if (version != 1) {
            throw new IOException("unsupported FmRanker version: " + version);
        }
        int k = dis.readInt();
        boolean useGenreCross = dis.readBoolean();
        FmRanker fm = new FmRanker(k, useGenreCross);

        fm.numFeatures = dis.readInt();
        fm.userOffset = dis.readInt();
        fm.itemOffset = dis.readInt();
        fm.genreOffset = dis.readInt();
        fm.w0 = dis.readDouble();

        int nu = dis.readInt();
        fm.userIds = new long[nu];
        fm.userIdx.clear();
        for (int i = 0; i < nu; i++) {
            fm.userIds[i] = dis.readLong();
            fm.userIdx.put(fm.userIds[i], i);
        }

        int ni = dis.readInt();
        fm.itemIds = new long[ni];
        fm.itemIdx.clear();
        for (int i = 0; i < ni; i++) {
            fm.itemIds[i] = dis.readLong();
            fm.itemIdx.put(fm.itemIds[i], i);
        }

        int ng = dis.readInt();
        fm.genreIdx.clear();
        for (int i = 0; i < ng; i++) {
            String name = dis.readUTF();
            int idx = dis.readInt();
            fm.genreIdx.put(name, idx);
        }

        fm.w = new double[fm.numFeatures];
        for (int f = 0; f < fm.numFeatures; f++) {
            fm.w[f] = dis.readDouble();
        }
        fm.v = new double[fm.numFeatures][fm.k];
        for (int f = 0; f < fm.numFeatures; f++) {
            for (int d = 0; d < fm.k; d++) {
                fm.v[f][d] = dis.readDouble();
            }
        }

        int itemFeatLen = dis.readInt();
        fm.itemFeatures = new int[itemFeatLen][];
        for (int i = 0; i < itemFeatLen; i++) {
            int len = dis.readInt();
            int[] feats = new int[len];
            for (int t = 0; t < len; t++) {
                feats[t] = dis.readInt();
            }
            fm.itemFeatures[i] = feats;
        }
        return fm;
    }
}
