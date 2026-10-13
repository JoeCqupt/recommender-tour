package tour.recsys.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 对评分集合的只读视图,预先按用户/物品建立倒排索引。
 * 倒排索引是后面所有算法的地基:ItemCF 要"哪些用户评过这个物品",MF 要"某用户评过什么"。
 */
public final class Dataset {

    private final List<Rating> ratings;
    private final Map<Long, Movie> movies;
    private final Map<Long, List<Rating>> byUser = new HashMap<>();
    private final Map<Long, List<Rating>> byItem = new HashMap<>();
    private final double globalMean;
    private final long maxTimestamp;

    public Dataset(List<Rating> ratings, Map<Long, Movie> movies) {
        this.ratings = List.copyOf(ratings);
        this.movies = movies == null ? Map.of() : movies;
        double sum = 0;
        long maxTs = Long.MIN_VALUE;
        for (Rating r : this.ratings) {
            byUser.computeIfAbsent(r.userId(), k -> new ArrayList<>()).add(r);
            byItem.computeIfAbsent(r.movieId(), k -> new ArrayList<>()).add(r);
            sum += r.rating();
            maxTs = Math.max(maxTs, r.timestamp());
        }
        this.globalMean = this.ratings.isEmpty() ? 0.0 : sum / this.ratings.size();
        this.maxTimestamp = maxTs;
    }

    public List<Rating> ratings() {
        return ratings;
    }

    public Map<Long, Movie> movies() {
        return movies;
    }

    public List<Rating> ratingsOfUser(long userId) {
        return byUser.getOrDefault(userId, List.of());
    }

    public List<Rating> ratingsOfItem(long movieId) {
        return byItem.getOrDefault(movieId, List.of());
    }

    /** 用户评过分的物品集合(评估时作为"已见要去掉"的集合)。 */
    public Set<Long> itemsOfUser(long userId) {
        List<Rating> rs = ratingsOfUser(userId);
        Set<Long> out = new java.util.HashSet<>(rs.size() * 2);
        for (Rating r : rs) {
            out.add(r.movieId());
        }
        return out;
    }

    public Set<Long> users() {
        return byUser.keySet();
    }

    public Set<Long> items() {
        return byItem.keySet();
    }

    public int userCount() {
        return byUser.size();
    }

    public int itemCount() {
        return byItem.size();
    }

    public int ratingCount() {
        return ratings.size();
    }

    public double globalMean() {
        return globalMean;
    }

    public long maxTimestamp() {
        return maxTimestamp;
    }

    public String titleOf(long movieId) {
        Movie m = movies.get(movieId);
        return m == null ? ("<unknown " + movieId + ">") : m.title();
    }

    /** 按物品被评分次数降序返回,热门 baseline 直接复用。 */
    public List<Long> itemsByPopularity() {
        List<Long> items = new ArrayList<>(byItem.keySet());
        Collections.sort(items, (a, b) -> Integer.compare(byItem.get(b).size(), byItem.get(a).size()));
        return items;
    }
}
