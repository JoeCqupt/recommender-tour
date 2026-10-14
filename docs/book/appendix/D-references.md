# 附录 D:参考资料

## 论文(按阅读优先级)

1. **Koren, Bell, Volinsky. "Matrix Factorization Techniques for Recommender Systems", IEEE Computer, 2009.** —— 第 3 部 MF 的直接出处,偏置项、SGD、正则都在这。必读。
   - IEEE:https://ieeexplore.ieee.org/document/5197422
   - DOI:https://dl.acm.org/doi/10.1109/MC.2009.263
2. **Linden, Smith, York. "Amazon.com Recommendations: Item-to-Item Collaborative Filtering", IEEE Internet Computing, 2003.** —— 第 2 部 ItemCF 的工业界开山之作,解释了 Amazon 为什么选 item-item 而非 user-user。
3. **Sarwar et al. "Item-Based Collaborative Filtering Recommendation Algorithms", WWW 2001.** —— ItemCF 的学术定义与相似度度量对比。
4. **Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009.** —— 第 4 部的入口。成对损失,解决"评分准 ≠ 排得前"的目标错配。
   - https://arxiv.org/abs/1205.2618

### 进阶(第 7 部路线图里的)

5. He et al. "Neural Collaborative Filtering", WWW 2017. —— 用神经网络做协同过滤。
6. Covington et al. "Deep Neural Networks for YouTube Recommendations", RecSys 2016. —— 工业级召回的两塔结构。
7. Zhou et al. "Deep Interest Network for Click-Through Rate Prediction", KDD 2018. —— DIN,用户行为序列建模。
8. Kang & McAuley. "Self-Attentive Sequential Recommendation", ICDM 2018. —— SASRec,序列推荐。
9. Cheng et al. "Wide & Deep Learning for Recommender Systems", 2016; Guo et al. "DeepFM", 2017. —— 特征交叉与深度模型。

---

## 书(中文,建立体系)

10. **项亮.《推荐系统实践》** —— 薄,入门最快,先建立词汇和直觉。
11. **王喆.《深度学习推荐系统》** —— 精排圣经,读完第 4 部之后反复翻。

---

## 数据与代码

12. **MovieLens / GroupLens**:https://grouplens.org/datasets/movielens/ —— 本项目用的 `ml-latest-small` 出处。
13. **微软 recommenders 库**:https://github.com/recommenders-team/recommenders —— 想对照工业实现时看(注意是 Python)。

---

**返回**:[全书目录](index.md)。
