package tour.recsys.serve;

import tour.recsys.model.FmRanker;
import tour.recsys.recall.PopularRecall;
import tour.recsys.recall.TwoStageRecommender;
import tour.recsys.recall.VectorRecall;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * 迷你推荐服务:加载训练产物,对外提供推荐接口。
 *
 * <p>这是"离线训练 → 在线 serving"的最小闭环。它封装了:
 * <ul>
 *   <li>向量召回({@link VectorRecall}) —— 从全库物品里快速捞候选;</li>
 *   <li>FM 精排({@link FmRanker}) —— 对候选做特征交叉打分;</li>
 *   <li>BPR 向量精排({@link BprRanker}) —— 用于 A/B 对比的基线;</li>
 *   <li>热门兜底({@link PopularRecall}) —— 冷启动用户没有向量时的退路。</li>
 * </ul>
 */
public final class RecService {

    private final EmbeddingStore store;
    private final FmRanker fm;
    private final BprRanker bprRanker;
    private final PopularRecall popularRecall;
    private final VectorRecall vectorRecall;
    private final TwoStageRecommender fmPipeline;
    private final TwoStageRecommender bprPipeline;
    private final int candidateSize;

    public RecService(EmbeddingStore store, FmRanker fm, List<Long> popularItems, int candidateSize) {
        this.store = store;
        this.fm = fm;
        this.bprRanker = new BprRanker(store);
        this.popularRecall = new PopularRecall(popularItems);
        this.vectorRecall = new VectorRecall(store);
        this.fmPipeline = new TwoStageRecommender(vectorRecall, fm, candidateSize);
        this.bprPipeline = new TwoStageRecommender(vectorRecall, bprRanker, candidateSize);
        this.candidateSize = candidateSize;
    }

    /** 默认用 FM 精排推荐;冷启动用户走热门兜底。 */
    public List<Long> recommend(long userId, int topN) {
        return recommend(userId, topN, "fm");
    }

    /** 指定实验桶:fm 或 bpr。冷启动用户统一走热门兜底。 */
    public List<Long> recommend(long userId, int topN, String variant) {
        if (store.userVector(userId) == null) {
            return popularRecall.recall(userId, topN, Set.of());
        }
        TwoStageRecommender pipeline = "bpr".equals(variant) ? bprPipeline : fmPipeline;
        return pipeline.recommend(userId, topN, Set.of());
    }

    /** 按 userId 哈希决定进哪个实验桶(简单 A/B 分流)。 */
    public String abVariant(long userId) {
        return (userId % 2 == 0) ? "fm" : "bpr";
    }

    public int candidateSize() {
        return candidateSize;
    }

    /** 把整个服务依赖的模型和热门列表保存到目录。 */
    public static void save(String dir, EmbeddingStore store, FmRanker fm, List<Long> popularItems) throws IOException {
        java.io.File d = new java.io.File(dir);
        if (!d.exists()) {
            d.mkdirs();
        }
        try (OutputStream out = new FileOutputStream(dir + "/embedding.store")) {
            store.save(out);
        }
        try (OutputStream out = new FileOutputStream(dir + "/fm.ranker")) {
            fm.saveModel(out);
        }
        try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(dir + "/popular.items"))) {
            dos.writeInt(popularItems.size());
            for (long id : popularItems) {
                dos.writeLong(id);
            }
        }
    }

    /** 从目录加载服务。 */
    public static RecService load(String dir, int candidateSize) throws IOException {
        EmbeddingStore store;
        try (InputStream in = new FileInputStream(dir + "/embedding.store")) {
            store = EmbeddingStore.load(in);
        }
        FmRanker fm;
        try (InputStream in = new FileInputStream(dir + "/fm.ranker")) {
            fm = FmRanker.loadModel(in);
        }
        List<Long> popularItems;
        try (DataInputStream dis = new DataInputStream(new FileInputStream(dir + "/popular.items"))) {
            int n = dis.readInt();
            popularItems = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                popularItems.add(dis.readLong());
            }
        }
        return new RecService(store, fm, popularItems, candidateSize);
    }
}
