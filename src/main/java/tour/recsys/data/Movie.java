package tour.recsys.data;

/** 一部电影。title 里可能含逗号(CSV 中被引号包裹),所以加载器不能简单按逗号切分。 */
public record Movie(long movieId, String title, String genres) {
}
