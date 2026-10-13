package tour.recsys.data;

/** 一条显式评分:用户 userId 对电影 movieId 打了 rating 分(0.5~5.0),发生于 timestamp。 */
public record Rating(long userId, long movieId, double rating, long timestamp) {
}
