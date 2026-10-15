# recommender-tour · 阶段四

从零手写推荐算法,不含任何第三方依赖(mvn + JDK 17)。

**阶段四目标**:给第 6 部的两级架构换上"真正不一样、且更强"的精排——引入**特征交叉**做 CTR 估计。我们**手写一个 FM(因子分解机)**,用它给召回候选重新排序,第一次让两级架构在**效果和性能上同时赢过单级**;再往上一层,读懂工业界主流的深度模型(Wide&Deep / DeepFM / DIN)。

> 📖 **《推荐系统入门:从一行代码到一个能上线的系统》** —— 全书目录:[`docs/book/index.md`](docs/book/index.md)
>
> 一本按"数学 → 初-中-高级算法 → 系统搭建"组织的路线图式教程,共 9 部 + 5 附录。建议从 [序言](docs/book/part-00-preface.md) 开始按顺序读。
>
> | 部 | 章节 | 技术象限 |
> |---|---|---|
> | 1 | [数学够用就好](docs/book/part-01-math.md) | 🔵 |
> | 2 | [推荐第一课:ItemCF](docs/book/part-02-itemcf.md) | 🟢 |
> | 3 | [从统计到学习:矩阵分解 MF](docs/book/part-03-mf.md) | 🔵🟣 |
> | 4 | [隐式反馈与排序学习:BPR-MF](docs/book/part-04-bpr.md) | 🔵🟣 |
> | 5 | [评估:怎么判断推荐好坏](docs/book/part-05-evaluation.md) | ⚙️ |
> | 6 | [召回 + 精排:两级架构](docs/book/part-06-two-stage.md) | ⚙️ |
> | 7 | [CTR 估计:手写 FM 与深度模型](docs/book/part-07-ctr.md) | 🔵🟣🟠 |
> | 8 | [进阶路线图](docs/book/part-08-roadmap.md) | 🟠 |
> | 9 | [工程与服务](docs/book/part-09-engineering.md) | ⚙️ |
> | 附录 | [A 可视化](docs/book/appendix/A-mf-visualization.md) · [B 踩坑](docs/book/appendix/B-pitfalls.md) · [C 自测](docs/book/appendix/C-self-test.md) · [D 参考](docs/book/appendix/D-references.md) · [E 冷启动](docs/book/appendix/E-coldstart.md) |  |
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
java -cp target/classes tour.recsys.Stage4Main data/ml-latest-small 10  # 阶段四:FM 精排 vs BPR 精排
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
           Ranker(精排接口) / FmRanker(阶段四:手写 FM)
  recall/  Recall(接口) / PopularRecall / RecommenderRecall / VectorRecall / MultiRecall
           TwoStageRecommender(两级主干)
  serve/   EmbeddingStore(训练产物 → 线上形态)
  eval/    Metrics(HR@K, NDCG@K, RMSE) / Evaluation
  Main.java       串起单级全流程(阶段二)
  Stage3Main.java 两级架构实验(阶段三)
  Stage4Main.java FM 精排实验(阶段四)
```

## 模型 / 组件

| 象限 | 组件 | 做什么 | 怎么打分 |
|---|---|---|---|
| 🟢 统计/规则 | `PopularRecommender` | 兜底 baseline | 按物品被评分次数排序,个性化算法必须超过它 |
| 🟢 统计/规则 | `ItemCF` | Top-N 排序 | 两个物品被同一批用户喜欢就相似;用户历史物品的相似物品相似度累加 |
| 🔵 机器学习 · 🟣 神经网络 | `MatrixFactorization` | 评分预测(阶段一) | `r_hat = mu + b_u + b_i + p_u·q_i`,SGD 训练(无隐藏层的最小神经网络) |
| 🔵 机器学习 · 🟣 神经网络 | `BprMF` | Top-N 排序(阶段二) | 对正样本(i)和负样本(j)学 `x_ui > x_uj`,SGD 训练(sigmoid 输出层) |
| 🟣 神经网络 | `FmRanker` | CTR 精排(阶段四) | FM:`ŷ = w0 + Σwᵢxᵢ + Σ⟨vᵢ,vⱼ⟩xᵢxⱼ`,把 用户×物品×品类 铺成稀疏特征,log loss + 负采样 SGD |
| ⚙️ 工程/系统 | `VectorRecall` | 召回(阶段三) | 用 BPR 的用户向量 × 物品向量做近似检索,小顶堆取 Top-N |
| ⚙️ 工程/系统 | `TwoStageRecommender` | 两级主干 | 召回几百个候选 → 用 `Ranker` 精排 → Top-K |

> 象限图例见 [第 1 部](docs/book/part-01-math.md) §"技术象限"。一句话:**阶段 1–3 你已经用上了 🔵 机器学习;🟣 在 FM 达到顶点;🟠 深度学习从第 7 部 §6 的深度模型开始。**

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

## 阶段四实验结果(Stage4Main, K=10)

同一路向量召回(BPR 向量,候选 200),换不同精排;FM 用 `k=32`、负采样 10、60 轮。

| 精排器 | HR@10 | NDCG@10 | 覆盖率 |
|---|---|---|---|
| 热门分(弱精排) | 0.0315 | 0.0155 | 2.6% |
| BPR 向量分(=召回) | 0.0438 | 0.0205 | 15.3% |
| **FM(品类交叉)** | **0.0442** | **0.0218** | 7.2% |

**特征消融**(同 FM,只开关品类特征):

| FM 变体 | HR@10 | NDCG@10 |
|---|---|---|
| 仅 `用户 × 物品` | 0.0418 | 0.0201 |
| 加 `品类` 交叉 | **0.0442** | **0.0218** |

**单级 vs 两级总账**:

```
方案                                   HR@10   NDCG@10      覆盖率
BPR 单级(扫全库打分)                   0.0438    0.0205    15.3%
向量召回 → BPR 精排                     0.0438    0.0205    15.3%
多路召回 → FM 精排                      0.0446    0.0219     7.0%
```

## 阶段四最重要的三个结论

### 1. 精排必须"和召回不一样,且更强",两级才有收益

FM 引入了召回没有的「用户 × 物品 × 品类」交叉信号,HR@10 和 NDCG@10 都超过了 BPR 精排,**并且更快**(只给 200 个候选打分)。两级架构第一次同时赢下效果和性能。

### 2. 特征不是越多越好,每个特征都要消融验证

关掉品类特征,FM 退化成"带线性偏置的 MF",效果和 BPR 差不多;打开后效果提升。但——**品类交叉在有些配置下反而拉低效果**(粗粒度先验,数据量小时易过拟合)。特征工程靠的是消融实验,不是直觉。

### 3. 效果与覆盖率是要权衡的一对指标

FM 的 HR/NDCG 比 BPR 高,但**覆盖率更低**(7.2% vs 15.3%)——它更偏好大众口味,推的东西更集中。只看单一指标会骗人。

## 阶段标签

本项目每个阶段都有 git tag,方便你切换回看:

```bash
git checkout stage-1   # 阶段一:ItemCF + 评分 MF
git checkout stage-2   # 阶段二:BPR-MF 隐式反馈排序
git checkout stage-3   # 阶段三:召回 + 精排两级架构
git checkout stage-4   # 阶段四:手写 FM,CTR 精排
# 当前工作区即 stage-4 之后的最新状态
```

> 📖 `stage-3` 之后新增标签 `stage-3-book`:把分散的文档重组成一本书([`docs/book/`](docs/book/index.md))。原 `stage-1/2/3` 标签保持不变,旧文档内容可通过它们随时找回。

## 阶段五已完成(工程与服务)

- 新增 `EmbeddingStore.save/load`、`FmRanker.saveModel/loadModel`,把训练产物导出为 `models/` 下的二进制文件。
- 新增 `RecService` 封装召回、FM 精排、BPR 精排、热门兜底和 A/B 分流。
- 新增 `RecServer`:用 JDK 内置 HttpServer 提供 `/recommend` 和 `/ab` HTTP 接口,零第三方依赖。
- 新增 `Stage5Main`:训练 → 导出 → 加载 → 验证闭环。
- 新增 [第 9 部 工程与服务](docs/book/part-09-engineering.md) 文档。

启动服务:

```bash
mvn -q compile
java -cp target/classes tour.recsys.Stage5Main data/ml-latest-small models
java -cp target/classes tour.recsys.serve.RecServer models 8080
curl 'http://localhost:8080/recommend?userId=1&n=10'
```

## 下一步(阶段六)

1. **论文与前沿**:带着前五个阶段的手感精读经典论文(MF、ItemCF、BPR、NCF、YouTube DNN、DIN、双塔、SASRec)。
2. **深度模型实践**:DeepFM / DIN 这类带隐藏层与注意力的模型,换到 Python(PyTorch)实现。

## 已埋的扩展点

- `Recommender` 接口是统一出口,新算法只要实现 `recommend(userId, topN, exclude)` 就能直接进评估表。
- `Recall` / `Ranker` 两个接口把"召回"和"精排"解耦,任何一路召回、任何精排模型(如 `FmRanker`)都能自由组合。
- `FmRanker` 的特征布局(用户段 / 物品段 / 品类段)是加新特征域的模板——再加"时段""设备"只需扩一段。
- `RecommenderRecall` 适配器让阶段一/二的单级模型零改动地接入多路召回。
- `EmbeddingStore` 演示了"离线训练 → 导出 → 在线 serving"的最小闭环。
- `TrainTestSplitter` 同时提供 `leaveOneOut` 与 `timeSplit`;`Evaluation` 与模型解耦。
