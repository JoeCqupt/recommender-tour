package tour.recsys.recall;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 多路召回:把若干路召回的候选合并成一路。
 *
 * <p>为什么需要多路?单路召回总有盲区——热门召回记不住长尾,向量召回对没看过的
 * 新用户无从下手,ItemCF 召不回没共现过的物品。工业界标配是"多路并行 + 合并去重",
 * 让不同的召回互相补盲。
 *
 * <p>合并策略:轮询(round-robin)从各路的第 1 名、第 2 名……依次取,保证每一路都有
 * 机会贡献候选,而不是被某一路刷屏。去重后取前 n。
 */
public final class MultiRecall implements Recall {

    private final List<Recall> sources;
    private final int perSource;

    public MultiRecall(List<Recall> sources, int perSource) {
        this.sources = sources;
        this.perSource = perSource;
    }

    @Override
    public List<Long> recall(long userId, int n, Set<Long> exclude) {
        List<List<Long>> lists = new ArrayList<>(sources.size());
        int maxLen = 0;
        for (Recall r : sources) {
            List<Long> l = r.recall(userId, perSource, exclude);
            lists.add(l);
            maxLen = Math.max(maxLen, l.size());
        }
        LinkedHashSet<Long> merged = new LinkedHashSet<>();
        for (int i = 0; i < maxLen && merged.size() < n; i++) {
            for (List<Long> l : lists) {
                if (i < l.size()) {
                    merged.add(l.get(i));
                    if (merged.size() >= n) {
                        break;
                    }
                }
            }
        }
        return new ArrayList<>(merged);
    }

    @Override
    public String name() {
        return "MultiRecall";
    }
}
