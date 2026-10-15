package tour.recsys.serve;

import tour.recsys.model.BprMF;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
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
    private final Map<Long, Integer> itemIndex = new HashMap<>();
    private final double[][] itemVectors;
    private final double[] itemBias;

    private EmbeddingStore(int dim, long[] itemIds, double[][] itemVectors, double[] itemBias) {
        this.dim = dim;
        this.itemIds = itemIds;
        this.itemVectors = itemVectors;
        this.itemBias = itemBias;
        for (int i = 0; i < itemIds.length; i++) {
            this.itemIndex.put(itemIds[i], i);
        }
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

    /** 按 itemId 查下标,不存在返回 -1(供精排层用)。 */
    public int itemIndex(long itemId) {
        Integer idx = itemIndex.get(itemId);
        return idx == null ? -1 : idx;
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

    /** 把向量库序列化到输出流,用于"离线训练 → 导出模型"。 */
    public void save(OutputStream out) throws IOException {
        DataOutputStream dos = new DataOutputStream(out);
        dos.writeInt(0x454D5354); // magic "EMST"
        dos.writeInt(1);          // version
        dos.writeInt(dim);
        dos.writeInt(userVectors.size());
        for (Map.Entry<Long, double[]> e : userVectors.entrySet()) {
            dos.writeLong(e.getKey());
            for (double v : e.getValue()) {
                dos.writeDouble(v);
            }
        }
        dos.writeInt(itemIds.length);
        for (int i = 0; i < itemIds.length; i++) {
            dos.writeLong(itemIds[i]);
            dos.writeDouble(itemBias[i]);
            for (double v : itemVectors[i]) {
                dos.writeDouble(v);
            }
        }
        dos.flush();
    }

    /** 从输入流反序列化向量库,用于"在线服务加载模型"。 */
    public static EmbeddingStore load(InputStream in) throws IOException {
        DataInputStream dis = new DataInputStream(in);
        int magic = dis.readInt();
        if (magic != 0x454D5354) {
            throw new IOException("invalid EmbeddingStore magic: " + Integer.toHexString(magic));
        }
        int version = dis.readInt();
        if (version != 1) {
            throw new IOException("unsupported EmbeddingStore version: " + version);
        }
        int dim = dis.readInt();
        int nu = dis.readInt();
        Map<Long, double[]> userVectors = new HashMap<>(nu * 2);
        for (int i = 0; i < nu; i++) {
            long uid = dis.readLong();
            double[] vec = new double[dim];
            for (int f = 0; f < dim; f++) {
                vec[f] = dis.readDouble();
            }
            userVectors.put(uid, vec);
        }
        int ni = dis.readInt();
        long[] itemIds = new long[ni];
        double[] itemBias = new double[ni];
        double[][] itemVectors = new double[ni][dim];
        for (int i = 0; i < ni; i++) {
            itemIds[i] = dis.readLong();
            itemBias[i] = dis.readDouble();
            for (int f = 0; f < dim; f++) {
                itemVectors[i][f] = dis.readDouble();
            }
        }
        EmbeddingStore store = new EmbeddingStore(dim, itemIds, itemVectors, itemBias);
        store.userVectors.putAll(userVectors);
        return store;
    }
}
