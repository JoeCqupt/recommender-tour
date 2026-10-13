# recommender-tour · 阶段一

从零手写推荐算法,不含任何第三方依赖(mvn + JDK 17)。

**阶段一目标**:用最小可跑的代码建立三件事的直觉 —— 召回/排序的分数是怎么来的、离线怎么评估、以及"评分预测"和"Top-N 排序"是两种不同的任务。

> 📘 **算法详解文档**:[`docs/阶段一-算法实现详解.md`](docs/阶段一-算法实现详解.md) —— ItemCF / MF 的设计思想、逐段代码走查、评估指标、踩坑复盘与参考资料。  
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

## 代码结构

```
src/main/java/tour/recsys/
  data/   Rating / Movie / CsvLoader / Dataset / TrainTestSplitter
  model/  Recommender(接口) / PopularRecommender / ItemCF / MatrixFactorization
  eval/   Metrics(HR@K, NDCG@K, RMSE) / Evaluation
  Main.java  串起全流程
```

## 三个模型

| 模型 | 做什么 | 怎么打分 |
|---|---|---|
| `PopularRecommender` | 兜底 baseline | 按物品被评分次数排序,个性化算法必须超过它 |
| `ItemCF` | Top-N 排序 | 两个物品被同一批用户喜欢就相似;用户历史物品的相似物品相似度累加 |
| `MatrixFactorization` | 评分预测(也能排 Top-N) | `r_hat = mu + b_u + b_i + p_u·q_i`,SGD 训练 |

**切分**:leave-one-out —— 每个用户留最新一条评分作测试,其余训练(用户交互数 < 5 剔除)。
**评估**:去掉训练集已见物品后取 Top-K,和留出的真实物品比对。

## 当前结果(K=10)

```
模型              HR@10    NDCG@10     覆盖率
Popular         0.0410     0.0193      1.3%
ItemCF          0.0541     0.0259      4.5%   ← 排序最好,但不会评分预测
MF(k=100)       0.0197     0.0097      6.8%   ← 排序弱

MF 评分预测 RMSE = 0.9664   (全局均值 baseline RMSE = 1.1329)
```

## 阶段一最重要的两个结论(踩坑记录)

### 1. 热门惩罚参数用错量纲,会把 ItemCF 的召回直接清零

`ItemCF` 里 `popAlpha` 是热门惩罚系数:越热门的物品相似度打折越狠。我一开始默认 `popAlpha=10`,结果 **HR@10 = 0.0000**,比什么都不做的 Popular 还差。参数扫描:

| alpha | topK | HR@10 | 覆盖率 |
|---|---|---|---|
| **0.0** | 50 | **0.0541** | 4.5% |
| 0.5 | 50 | 0.0000 | 8.3% |
| 1.0 | 50 | 0.0016 | 2.9% |
| 10.0 | 200 | 0.0000 | 1.7% |

**原因**:惩罚公式写成了 `1 / (1 + alpha * (sqrt(n_j) - 1))`。`n_j` 是物品被评分次数,开根后仍不小(热门片能到 ~17),乘上 alpha 后分母爆炸,只有"极少人评过"的冷门片能拿到非零相似度 —— 于是模型永远只推冷门噪声片,留出的热门片一条都不命中。

**教训**:任何"惩罚/归一化"项,先确认它的量纲和取值范围再设系数;一个系数就能把信号全压没。稀疏数据上宁可先关掉惩罚(`alpha=0`),再逐步加。

### 2. "评分预测准"和"排序排得前"是两回事

MF 把 RMSE 从 1.1329(只会说全局均值)降到 0.9664,说明它**确实学到了用户/物品偏置和隐向量**;但它的 HR@10 只有 0.0197,远低于 ItemCF。

原因:MF 的损失函数是最小化**已观测评分**的平方误差,它从没见过"负样本",因此排序未观测物品时区分度很差。这是推荐系统里最经典的认知鸿沟 —— **rating prediction 的优化目标 ≠ Top-N ranking 的优化目标**。

正则系数 `reg` 把这个矛盾演示得很清楚:

| reg | RMSE | HR@10 |
|---|---|---|
| 0.00 | 0.9896 | 0.0197 |
| 0.02 | 0.9664 | 0.0197 |
| 0.05 | 0.9556 | 0.0082 |
| 0.10 | 0.9567 | 0.0000 |

正则越强,评分越"保守安全"、RMSE 越低,但隐向量被压平,排序能力反而归零。

**教训**:看 MF 的 RMSE 好看,不代表它适合做线上排序。要让 MF 排序强,需要换目标 —— 用隐式反馈 / 成对损失(BPR)训练。这正是阶段二要做的。

## 下一步(阶段二)

1. 把 MF 换成 **BPR(成对排序)** 训练,观察 HR@10 能否追平/超过 ItemCF。
2. 引入**负采样**和 **Recall@K**(测试集每用户多留几条)。
3. 从评分数据集转向**隐式反馈**(只保留"看过/点击"),贴近真实业务。

## 已埋的扩展点

- `Recommender` 接口是统一出口,新算法只要实现 `recommend(userId, topN, exclude)` 就能直接进评估表。
- `Evaluation.evaluateRanking` 与模型解耦,换算法不用改评估代码。
