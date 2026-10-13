# recommender-tour · 阶段二

从零手写推荐算法,不含任何第三方依赖(mvn + JDK 17)。

**阶段二目标**:把阶段一的评分 MF 改成 **BPR-MF**,从"预测评几分"转向"学习用户喜欢的物品应该排前面",并在隐式反馈(≥4.0)上验证排序指标超过 ItemCF。

> 📘 **算法详解文档**:[`docs/算法实现详解.md`](docs/算法实现详解.md) —— ItemCF / 评分 MF / BPR-MF 的设计思想、逐段代码走查、评估指标、踩坑复盘与参考资料。  
> 🎨 **MF 可视化教程**:[`docs/MF完全入门教程.md`](docs/MF完全入门教程.md) —— 从"填表格"到"品味空间",含 k=2 散点图和一次真实预测的五步拆解。

## 运行

```bash
./scripts/run.sh          # 默认推荐 Top-10
./scripts/run.sh 20       # 自定义 K
# 或手动
mvn -q compile
java -cp target/classes tour.recsys.Main data/ml-latest-small 10
```

## 数据集

MovieLens `ml-latest-small`:610 用户 / 9724 物品 / 100836 条评分(0.5~5.0)。
数据已在 `data/ml-latest-small/`,原始下载地址:`https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`

阶段二把显式评分转成隐式反馈:只保留 ≥4.0 的评分作为正样本。

## 代码结构

```
src/main/java/tour/recsys/
  data/   Rating / Movie / CsvLoader / Dataset / TrainTestSplitter
  model/  Recommender(接口) / PopularRecommender / ItemCF / MatrixFactorization / BprMF
  eval/   Metrics(HR@K, NDCG@K, RMSE) / Evaluation
  Main.java  串起全流程(阶段二)
```

## 三个模型

| 模型 | 做什么 | 怎么打分 |
|---|---|---|
| `PopularRecommender` | 兜底 baseline | 按物品被评分次数排序,个性化算法必须超过它 |
| `ItemCF` | Top-N 排序 | 两个物品被同一批用户喜欢就相似;用户历史物品的相似物品相似度累加 |
| `MatrixFactorization` | 评分预测(阶段一) | `r_hat = mu + b_u + b_i + p_u·q_i`,SGD 训练 |
| `BprMF` | Top-N 排序(阶段二) | 对正样本(i)和负样本(j)学 `x_ui > x_uj`,SGD 训练 |

**切分**:time-split 0.2 —— 每用户按时间排序,前 80% 训练,后 20% 测试(交互数 < 5 剔除)。
**评估**:去掉训练集已见物品后取 Top-K,和留出的真实物品比对;测试集每用户多条,指标取平均。

## 当前结果(K=10)

```
原始评分: 610 用户, 9724 物品, 100836 条评分
隐式正样本(>=4.0): 609 用户, 6298 物品, 48580 条交互
切分(time-split, 每用户交互>=5, 测试比例 20%): 训练 38858 条, 测试 603 用户 / 9704 条

评估指标 @K=10 (隐式反馈, 多个测试物品取平均)
---------------------------------------------------------------
模型                  HR@10    NDCG@10        覆盖率       训练耗时
Popular            0.0295     0.0149       1.0%         -
ItemCF             0.0396     0.0186      10.8%         -
BPR-MF(k=100)      0.0438     0.0205      15.3%    2167ms
---------------------------------------------------------------
```

## 阶段二最重要的结论

### 1. 目标函数对准任务,HR@10 才能上去

阶段一的评分 MF 把 RMSE 降到 0.97,但 HR@10 只有 2%。因为它最小化的是"已观测评分的平方误差",从来没学过"喜欢的 > 不喜欢的"这个排序关系。

BPR 把损失换成:

```
L = -ln σ(x_ui - x_uj)
```

对每一个正样本 `(u,i)` 采样若干负样本 `(u,j)`,让模型直接学习 `i` 排在 `j` 前面。结果 HR@10 从 MF 的 0.0197 提升到 0.0438,超过 ItemCF 的 0.0396。

**教训**:离线指标好看的模型,上线未必排序好。做 Top-N 推荐,就要用排序损失训练。

### 2. 隐式正样本的定义是业务问题

一开始把全部评分(包括 1~3 分)当正样本,BPR 的 HR@10 只有 0.0246,反而不如 ItemCF。改成只保留 ≥4.0 的评分后,HR@10 提升到 0.0438。

**原因**:低分评分不代表"喜欢",把它们当正样本会污染训练信号。真实业务里,正样本往往要结合点击、播放完成率、停留时长等行为共同定义。

## 阶段标签

本项目每个阶段都有 git tag,方便你切换回看:

```bash
git checkout stage-1   # 阶段一:ItemCF + 评分 MF
# 当前工作区即 stage-2
```

## 下一步(阶段三)

1. **召回 + 精排的两级架构**:用 ItemCF 或向量近似检索做候选召回,BPR-MF 做 Top-K 精排,解决 BPR-MF 全量打分 O(物品数×k) 太慢的问题。
2. **更真实的隐式反馈建模**:引入曝光/点击日志、处理位置偏置、尝试 hard negative 采样。
3. **模型服务化**:把训练好的用户/物品向量导出,用向量检索引擎(Faiss/HNSW)做线上推荐。

## 已埋的扩展点

- `Recommender` 接口是统一出口,新算法只要实现 `recommend(userId, topN, exclude)` 就能直接进评估表。
- `TrainTestSplitter` 同时提供 `leaveOneOut` 与 `timeSplit`,方便对比不同任务下的评估方式。
- `Evaluation.evaluateRanking` 与模型解耦,换算法不用改评估代码。
