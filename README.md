# recommender-tour · 阶段三

从零手写推荐算法,不含任何第三方依赖(mvn + JDK 17)。

**阶段三目标**:把阶段二那个"扫全库才打得动分"的 BPR-MF,改造成能上线的 **召回 + 精排两级架构**;顺便搞清两件容易被忽略的事——**召回率是精排的天花板**、**两级不一定比单级好**。

> 📖 **《推荐系统入门:从一行代码到一个能上线的系统》** —— 全书目录:[`docs/book/index.md`](docs/book/index.md)
>
> 一本按"数学 → 初-中-高级算法 → 系统搭建"组织的路线图式教程,共 7 部 + 4 附录。建议从 [序言](docs/book/part-00-preface.md) 开始按顺序读。
>
> | 部 | 章节 | 技术象限 |
> |---|---|---|
> | 1 | [数学够用就好](docs/book/part-01-math.md) | 🔵 |
> | 2 | [推荐第一课:ItemCF](docs/book/part-02-itemcf.md) | 🟢 |
> | 3 | [从统计到学习:矩阵分解 MF](docs/book/part-03-mf.md) | 🔵🟣 |
> | 4 | [隐式反馈与排序学习:BPR-MF](docs/book/part-04-bpr.md) | 🔵🟣 |
> | 5 | [评估:怎么判断推荐好坏](docs/book/part-05-evaluation.md) | ⚙️ |
> | 6 | [召回 + 精排:两级架构](docs/book/part-06-two-stage.md) | ⚙️ |
> | 7 | [进阶路线图](docs/book/part-07-roadmap.md) | 🟠 |
> | 附录 | [A 可视化](docs/book/appendix/A-mf-visualization.md) · [B 踩坑](docs/book/appendix/B-pitfalls.md) · [C 自测](docs/book/appendix/C-self-test.md) · [D 参考](docs/book/appendix/D-references.md) |  |
>
> 🧮 一看到公式就发怵?从 [第 1 部](docs/book/part-01-math.md) 读起 —— 只讲"够用"的那点数学(向量/点积/Σ/损失/梯度/sigmoid),每个符号配人话和代码行,20 分钟读完。

## 运行

```bash
./scripts/run.sh          # 阶段二:单级模型对比(Top-10)
./scripts/run.sh 20       # 自定义 K
# 或手动
mvn -q compile
java -cp target/classes tour.recsys.Main data/ml-latest-small 10        # 阶段二:单级对比
java -cp target/classes tour.recsys.Stage3Main data/ml-latest-small 10  # 阶段三:两级架构实验
```

## 数据集

MovieLens `ml-latest-small`:610 用户 / 9724 物品 / 100836 条评分(0.5~5.0)。
数据已在 `data/ml-latest-small/`,原始下载地址:`https://files.grouplens.org/datasets/movielens/ml-latest-small.zip`

阶段二起把显式评分转成隐式反馈:只保留 ≥4.0 的评分作为正样本。

## 代码结构

```
src/main/java/tour/recsys/
  data/    Rating / Movie / CsvLoader / Dataset / TrainTestSplitter
  model/   Recommender(接口) / PopularRecommender / ItemCF / MatrixFactorization / BprMF
           Ranker(精排接口)
  recall/  Recall(接口) / PopularRecall / RecommenderRecall / VectorRecall / MultiRecall
           TwoStageRecommender(两级主干)
  serve/   EmbeddingStore(训练产物 → 线上形态)
  eval/    Metrics(HR@K, NDCG@K, RMSE) / Evaluation
  Main.java       串起单级全流程(阶段二)
  Stage3Main.java 两级架构实验(阶段三)
```

## 模型 / 组件

| 象限 | 组件 | 做什么 | 怎么打分 |
|---|---|---|---|
| 🟢 统计/规则 | `PopularRecommender` | 兜底 baseline | 按物品被评分次数排序,个性化算法必须超过它 |
| 🟢 统计/规则 | `ItemCF` | Top-N 排序 | 两个物品被同一批用户喜欢就相似;用户历史物品的相似物品相似度累加 |
| 🔵 机器学习 · 🟣 神经网络 | `MatrixFactorization` | 评分预测(阶段一) | `r_hat = mu + b_u + b_i + p_u·q_i`,SGD 训练(无隐藏层的最小神经网络) |
| 🔵 机器学习 · 🟣 神经网络 | `BprMF` | Top-N 排序(阶段二) | 对正样本(i)和负样本(j)学 `x_ui > x_uj`,SGD 训练(sigmoid 输出层) |
| ⚙️ 工程/系统 | `VectorRecall` | 召回(阶段三) | 用 BPR 的用户向量 × 物品向量做近似检索,小顶堆取 Top-N |
| ⚙️ 工程/系统 | `TwoStageRecommender` | 两级主干(阶段三) | 召回几百个候选 → 用 `Ranker` 精排 → Top-K |

> 象限图例见 [`docs/学习地图.md`](docs/学习地图.md) §1.1。一句话:**阶段 1–3 你已经用上了 🔵 机器学习;🟠 深度学习从阶段四开始。**

**切分**:time-split 0.2 —— 每用户按时间排序,前 80% 训练,后 20% 测试(交互数 < 5 剔除)。
**评估**:去掉训练集已见物品后取 Top-K,和留出的真实物品比对;测试集每用户多条,指标取平均。

## 阶段三实验结果(Stage3Main, K=10)

**表1 召回率天花板**(每路召回取 200 个候选,多路合并取 600):

| 召回通道 | 召回率 | 平均候选数 |
|---|---|---|
| PopularRecall | 25.8% | 200 |
| ItemCF召回 | 27.3% | 198 |
| VectorRecall | 36.9% | 200 |
| **MultiRecall** | **45.2%** | 394 |

**表4 单级 vs 两级**:

```
方案                                   HR@10   NDCG@10      覆盖率       评估耗时
Popular                             0.0295    0.0149     1.0%        2ms
ItemCF                              0.0396    0.0186    10.8%      156ms
BPR-MF(k=100)(单级:扫全库)           0.0438    0.0205    15.3%      587ms
两级(MultiRecall→600→精排)            0.0438    0.0205    15.3%      445ms
两级(VectorRecall→200→精排)           0.0438    0.0205    15.3%      225ms
```

## 阶段三最重要的三个结论

### 1. 两级架构首先解决的是"算力/延迟",不是"效果"

真实系统不会拿模型扫全库。召先把 1 亿物品缩到几百个候选,精排只给这几百个打分。本数据集物品才 6千+,耗时差距不明显;上亿时这就是能不能上线的分水岭。

### 2. 召回率是精排的天花板

召回没捞到的物品,再强的精排也排不出来。表1 里多路召回(45.2%)> 任何单路(最高 36.9%),这正是"多路并行 + 合并去重"存在的理由——不同的召回互相补盲。

### 3. "两级一定比单级好"是错觉 —— 精排必须比召回强

`Stage3Main` 里做了两组对照:

- **精排 = 召回**(都用 BPR 向量):候选集从 50 开到 1000,召回率从 14.9% 涨到 69.2%,**HR@10 却纹丝不动**(恒为 0.0438)。因为精排只是把召回的顺序重算一遍,两级 == 单级,只是白花时间。
- **精排 ≠ 召回**(换成"热门分"这个弱精排):候选集越大,HR 反而越低,`n=1000` 时跌回 Popular 基线的 0.0295——弱精排配大候选池会退化成"纯推热门"。

**教训**:召回与精排是接力棒,得匹配。精排比召回强,两级才有意义;否则不如直接用单级。这也是阶段四要"把精排换成真正不同的模型(带交叉特征的 CTR)"的原因。

## 阶段标签

本项目每个阶段都有 git tag,方便你切换回看:

```bash
git checkout stage-1   # 阶段一:ItemCF + 评分 MF
git checkout stage-2   # 阶段二:BPR-MF 隐式反馈排序
git checkout stage-3   # 阶段三:召回 + 精排两级架构
# 当前工作区即 stage-3 之后的"成书"整理版
```

> 📖 `stage-3` 之后新增标签 `stage-3-book`:把分散的文档重组成一本书([`docs/book/`](docs/book/index.md))。原 `stage-1/2/3` 标签保持不变,旧文档内容可通过它们随时找回。

## 下一步(阶段四)

1. **精排上真正的排序模型**:GBDT / DeepFM 等 CTR 模型,输入用户/物品/上下文的交叉特征,不再只是两个向量点积。
2. **特征体系**:用户画像、物品属性、交叉特征(用户×类别)、上下文(时段、设备)。
3. **语言换到 Python**:PyTorch / LightGBM / Faiss 是这一阶段的标配。前三个阶段用 Java 是为了看清齿轮怎么转,阶段四起用 Python 是为了站上巨人的肩膀。

## 已埋的扩展点

- `Recommender` 接口是统一出口,新算法只要实现 `recommend(userId, topN, exclude)` 就能直接进评估表。
- `Recall` / `Ranker` 两个接口把"召回"和"精排"解耦,任何一路召回、任何精排模型都能自由组合。
- `RecommenderRecall` 适配器让阶段一/二的单级模型零改动地接入多路召回。
- `EmbeddingStore` 演示了"离线训练 → 导出 → 在线 serving"的最小闭环。
- `TrainTestSplitter` 同时提供 `leaveOneOut` 与 `timeSplit`;`Evaluation` 与模型解耦。
