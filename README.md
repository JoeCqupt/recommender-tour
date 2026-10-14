# recommender-tour · 阶段三

从零手写推荐算法,不含任何第三方依赖(mvn + JDK 17)。

**阶段三目标**:把阶段二那个"扫全库才打得动分"的 BPR-MF,改造成能上线的 **召回 + 精排两级架构**;顺便搞清两件容易被忽略的事——**召回率是精排的天花板**、**两级不一定比单级好**。

> 📘 **算法详解文档**:[`docs/算法实现详解.md`](docs/算法实现详解.md) —— ItemCF / 评分 MF / BPR-MF / 两级架构的设计思想、逐段代码走查、评估指标、踩坑复盘与参考资料。  
> 🎨 **MF 可视化教程**:[`docs/MF完全入门教程.md`](docs/MF完全入门教程.md) —— 从"填表格"到"品味空间",含 k=2 散点图和一次真实预测的五步拆解。  
> 🗺️ **学习地图**:[`docs/学习地图.md`](docs/学习地图.md) —— 从入门到精通的六个阶段、三条主线,以及为什么前三个阶段用 Java、后面换 Python。

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

| 组件 | 做什么 | 怎么打分 |
|---|---|---|
| `PopularRecommender` | 兜底 baseline | 按物品被评分次数排序,个性化算法必须超过它 |
| `ItemCF` | Top-N 排序 | 两个物品被同一批用户喜欢就相似;用户历史物品的相似物品相似度累加 |
| `MatrixFactorization` | 评分预测(阶段一) | `r_hat = mu + b_u + b_i + p_u·q_i`,SGD 训练 |
| `BprMF` | Top-N 排序(阶段二) | 对正样本(i)和负样本(j)学 `x_ui > x_uj`,SGD 训练 |
| `VectorRecall` | 召回(阶段三) | 用 BPR 的用户向量 × 物品向量做近似检索,小顶堆取 Top-N |
| `TwoStageRecommender` | 两级主干(阶段三) | 召回几百个候选 → 用 `Ranker` 精排 → Top-K |

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
# 当前工作区即 stage-3
```

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
