# Med-PRISM v2.3：Task-Aware MS-CRC 方法设计

**推荐版本：保留 MS-CRC 名称，替换其 witness / selector；使用概念原子级历史支持、metric-anchored 权重与可补偿的正迁移保护。**

**状态：待实现、待模型验证的方法设计。** 本文不把已揭盲的 `off_07` / `pair_4_5` 视为新算法成绩，也不声称已经获得 v2.3 的系数或 retention 提升。已有结果来自用户提供的《Med-PRISM v2.3 方法设计任务》；公式、选择规则、门槛与理论命题是本次设计。本文没有项目 evaluator 源码，涉及官方解析与评估语义的部分以现有 evaluator 为准，不自行补写同义词规则或数据集约定。

---

## A. Final diagnosis of v2.2

### A.1 失败的是 exact-set selector，不是已经测试过的干预空间

v2.2 在 Concept Recognition 上要求整个预测概念集合完全正确，才能建立 historical-memory witness。但附件报告：Task3 dev exact-set accuracy 为 0%，official concept micro-F1 约 33%；fit exact accuracy 约 1.56%，holdout 为 0%。最终 P1/P2 witnesses 均为零，返回 `NO_SIGNAL` 与全一系数，没有进行非平凡搜索。

这支持的诊断是：**保护单位与任务的有效行为单位不匹配，使 selector 在搜索前丢弃了几乎全部部分正确行为。** 不能据此认定 current-only 历史支持必然不存在，也不能据此认定 rank-group calibration 无效。

来源：输入设计任务，第 330–380 行。

### A.2 已经成立的正证据与它的范围

| 已测试 composition | Task1 accuracy | Task2 accuracy | Task3 concept micro-F1 |
|---|---:|---:|---:|
| Full | 91.41% | 40.63% | 33.06% |
| `off_07` | 91.41% | 46.48% | 34.25% |
| `pair_4_5` | 91.02% | 49.61% | 33.95% |

附件还报告，`off_07` 相对 full 在 Task2 上没有新增 correct→wrong，并修复了 15 个 wrong→correct；full 已有的旧任务正迁移集合没有额外损失。`pair_4_5` 的 Task3 answer-token CE 超过此前约束，不能直接当作满足同一可行性条件的主结果。

来源：输入设计任务，第 398–502 行。

因此，目前可以说：**在已经检查的 checkpoint、干预面板和 development 数据上，存在兼顾 stability 与 plasticity 的静态选择性干预。** 还不能说这个配置能由 current-only 算法找到、能跨 seed 复现，或能泛化到 untouched test。

### A.3 v2.3 还必须避免两个新的错误

**只把 exact accuracy 换成 sample-F1，不足以完成修复。** 一条样本可能因历史 bank 找回一个 gold concept、同时多生成一个 FP，导致总分净变化很小；样本级净分会掩盖具体受支持的正确概念。micro-F1 与 sample-F1 的平均值也不是同一个指标。外部定义依据：scikit-learn 官方 `f1_score` 文档中 `micro` 与 `samples` 的区分。

**将“一个新增正确概念也不能丢”设成硬约束，也可能再次排除全部编辑。** 这不是附件中已经观察到的结果，而是本次设计识别的风险。因此，新方案保护可解释的正迁移总收益，并额外记录具体丢失项，而不是要求每个 full 输出原子永远不变。

---

## B. Design goal for v2.3

> **用当前任务中由历史参数有限贡献支持的正确行为原子，替代整样本 exact-set witness；在保留当前官方任务指标和正迁移收益的条件下，选择破坏这些原子更少的静态 current-private composition。**

核心可证伪命题保持为：

\[
\boxed{\text{当前任务上的 task-aware historical-memory support，能否选出对真实历史任务更安全的 rank-group 配置？}}
\]

v2.3 修复的是 current-only observability / selection。它不重新证明 A-only 的价值，不恢复 TPM/RCWP，也不重新发明持续学习主干。

---

## C. Three candidate selector designs

### C.1 候选 A：Sample-Soft Memory Support

行为单位仍是一条样本，采用任务分数 \(s_g(x)\in[0,1]\)。Concept 使用 sample-F1；VQA/Diagnosis 使用现有 correctness。

\[
a_i(x)=[s_H(x)-s_{H\setminus i}(x)]_+,
\qquad
L_{\mathrm{mem}}(g)=
\frac{\sum_x \max_i a_i(x)[s_H(x)-s_g(x)]_+}
{\sum_x\max_i a_i(x)}.
\]

正迁移保护采用 \(\sum_x[s_g-s_H]_+\ge\sum_x[s_{\mathrm{full}}-s_H]_+\)，同时约束真实官方任务指标与 CE。选择先最小化 memory loss，再减少正迁移丢失与参数编辑。

该候选只用当前数据和历史参数；样本分数及支持权重仅在 boundary 临时保留，最终只部署折叠系数。已有完整生成可直接重评分，额外工程成本最低。

**主要风险：** sample-F1 不等于 micro-F1；一条样本内不同概念的得失会相互抵消；对于 gold-empty 样本，sample-F1 往往无法区分不同 FP 数量。适合作为低成本对照，不作为主方案。

### C.2 候选 B：Concept-Atomic Memory Support——推荐

行为单位是“某个 gold concept 被识别”或“某个反事实暴露的 FP 被抑制”，而不是整个集合正确。

\[
a_i(x,c)=[u_H(x,c)-u_{H\setminus i}(x,c)]_+.
\]

权重由当前 full 的 micro-F1 锚定，memory loss 只统计历史参数支持原子的退化。正迁移通过“相对 H 的已纠正错误收益”计账：可以丢失少量既有修正，但必须由至少等价的新修正补偿；同时单独报告丢失数量。

选择采用 memory loss、已有正迁移丢失、编辑量的字典序。数据权限、参数折叠与推理结构不变。新增计算主要是对同一批生成结果做集合操作，不需要逐概念重新运行模型。

**主要风险：** H 在当前 Concept 数据上的真实概念能力可能仍然很弱；leave-one-bank 差异可能稀疏或只反映输出格式；当前原子的保护仍未必迁移到历史任务。它消除了整样本准入门槛，但不保证产生足够信号。

### C.3 候选 C：Metric-Anchored Sample Support

对当前 fit 上 full 的 micro-F1 \(q\)，定义逐样本加性计数分数：

\[
z_g(x)=(2-q)TP_g(x)-qFP_g(x).
\]

用 \([z_H-z_{H\setminus i}]_+\) 建立样本支持，再惩罚 \([z_H-z_g]_+\)。正迁移可以用正向计数增益保护，最终仍以实际 micro-F1 与 CE 检查可行性。

它和候选 A 一样只需 boundary 临时计数，无跨任务样本摘要；与现有 evaluator 的对接也较轻。与 sample-F1 相比，它的总计数差和相对 full 的 micro-F1 变化具有精确关系，见 F 节。

**主要风险：** 样本内的概念替换仍会净抵消；较长概念列表可能占主导；一个不受历史支持的新 TP 可以掩盖另一个历史支持 TP 的丢失。因此它适合作为“只修 metric aggregation 是否足够”的对照。

**选择：候选 B 作为主 selector，统一任务接口作为它的实现抽象。`BehaviorMetricAdapter` 不是第四个方法，也不应单独包装成创新模块。**

---

## D. Recommended v2.3

### D.1 名称与模块边界

继续叫 **Memory-Supported Counterfactual Rank Calibration（MS-CRC）**，本版本称 **Task-Aware MS-CRC**。内部可将 witness 实现命名为 `Concept-Atomic Memory Support`，不重命名整个 Med-PRISM。

训练保持：

\[
L=L_{\mathrm{task}}+0.1L_{\mathrm{A\text{-}orth}}+0.01L_{\mathrm{shared\text{-}drift}}.
\]

边界仅编辑当前 private：

\[
\widetilde P_t^w=B_t^w\operatorname{diag}(g^w)A_t^w.
\]

推理保持：

\[
W_{\mathrm{eff}}=W_0+S_t+\sum_{i<t}P_i+\widetilde P_t.
\]

所有历史 banks、全部 A、shared 在校准期间冻结。A-only、private rank=16、36 layers / 72 wrappers、16 个现有 rank groups 全部保留。来源：输入设计任务，第 88–140、282–326 行。

### D.2 第一版只改三项

将 whole-sample exact witness 换成 task-aware 原子支持；将 exact wrong→correct 保护换成原子级正迁移收益约束；将任务效用检查接到现有 official metric，而不是统一使用 exact accuracy。

**不增加 teacher 网络、不增加梯度统计、不加 history sidecar、不细分到 32/1152 个可控系数。** 初始 gate 复用完整的既有 finite panel，并对它重新评分；不给 `off_07` 或 `pair_4_5` 任何特殊优先权。

---

## E. Mathematical formulation

### E.1 固定 composition 参考

第一任务没有历史 banks 时直接提交 identity；不构造空的历史支持目标。

令：

\[
H=W_0+S_t+\sum_{i<t}P_i,
\qquad
H\setminus i=W_0+S_t+\sum_{k<t,k\ne i}P_k.
\]

\[
f_g=H+P_t(g),\qquad f_{\mathbf1}=\mathrm{full}.
\]

全部参考与候选使用同一个 \(S_t\)、同一当前输入、同一 generation 配置和同一官方解析器。历史参数的支持必须来自实际完整生成或官方所需的完整任务前向，不能用 gold-token CE 的差异替代概念生成差异。

### E.2 统一行为表示

对当前数据划分 \(D\)，适配器返回：

\[
s_t(x;g),\quad\mathcal A_x,\quad u_g(x,a)\in[0,1],\quad\omega(x,a),\quad R_D(g).
\]

其中，\(s_t\) 是样本级展示分数；\(u_g\) 是行为原子效用；\(R_D\) 才是官方数据集级指标。Concept 的 \(R_D\) 是汇总 TP/FP/FN 后的 micro-F1，不能写成 \(\frac1{|D|}\sum_x s_t(x;g)\)。

原子集合、参考支持和权重在评估候选之前固定。当前候选不能通过改变 witness 定义来降低自己的损伤分数。

### E.3 Historical support 与去重

\[
a_i(x,a)=[u_H(x,a)-u_{H\setminus i}(x,a)]_+.
\]

\[
\bar a(x,a)=\max_{i<t}a_i(x,a),
\qquad
w(x,a)=\omega(x,a)\bar a(x,a).
\]

Concept 原子中 \(a_i\in\{0,1\}\)。多个历史 banks 同时支持同一原子时，池化损失只计一次；保留每个 bank 的 \(a_i\) 用于归因诊断，不将重复出现解释成多份独立证据。

某个历史 bank 的支持量为零时，不能声称该任务得到了直接保护；池化支持非零也不意味着所有历史任务都可观察。

这里的零点有明确语义：“移除历史 bank 是否降低了这个正确行为”，不是待调超参数。不引入 top-k、分位数阈值或把无支持原子强行赋予正权重。

对连续原子，\(a_i\) 自然保留支持强度。**Concept 采用的是离散原子上的细粒度加权损失，不应虚称它对模型参数连续可微。** finite search 本来也不要求可微。

### E.4 Memory-preservation objective

令：

\[
Z_D=\sum_{x,a}w(x,a).
\]

当 \(Z_D>0\) 时：

\[
\boxed{
L_{\mathrm{mem},D}(g)=
\frac{1}{Z_D}\sum_{x,a}w(x,a)[u_H(x,a)-u_g(x,a)]_+.
}
\]

它只惩罚由历史参数支持的正确行为退化，不保护 H 的错误，也不强制候选复制 H 的整段答案。

若 \(Z_D=0\)，返回 `NO_SUPPORT`，不通过分母加一个 epsilon 制造“有效损失”。若有支持但 \(L_{\mathrm{mem},D}(\mathbf1)=0\)，记为 `NO_EXPOSED_HARM`：当前可见的历史支持行为并未被 full 破坏，保护目标没有推动非平凡编辑的依据。

### E.5 Positive-transfer ledger：保护收益，不冻结整个 full 答案

定义相对 H 的正向原子增益：

\[
v_g(x,a)=[u_g(x,a)-u_H(x,a)]_+,
\qquad
G_D(g)=\sum_{x,a}\omega(x,a)v_g(x,a).
\]

再定义 full 已有增益的丢失与新增修正：

\[
D_{\mathrm{PT},D}(g)
=\sum_{x,a}\omega(x,a)[v_{\mathbf1}(x,a)-v_g(x,a)]_+,
\]

\[
U_{\mathrm{PT},D}(g)
=\sum_{x,a}\omega(x,a)[v_g(x,a)-v_{\mathbf1}(x,a)]_+.
\]

存在精确恒等式：

\[
\boxed{G_D(g)-G_D(\mathbf1)=U_{\mathrm{PT},D}(g)-D_{\mathrm{PT},D}(g).}
\]

要求：

\[
\boxed{G_D(g)\ge G_D(\mathbf1).}
\]

即既有修正的损失必须由至少等价的新修正补偿。对 Concept，只有 H 漏掉的 gold concept 被补回、或 H 的 FP 被纠正，才能贡献该收益。**恢复一个 H 本来正确但被 full 弄错的概念，不算新增正迁移补偿；那属于 memory repair。** 因而两类目标没有混为普通当前任务 F1 最大化。

定义归一化丢失项：

\[
L_{\mathrm{PT},D}(g)=D_{\mathrm{PT},D}(g)/G_D(\mathbf1),
\]

仅在 \(G_D(\mathbf1)>0\) 时使用。若基线没有可见正向增益，该项置为选择上的零值，但诊断标为 `NOT_EVALUABLE`，不能声称验证了正迁移保留。

**本约束保护的是当前可观测、metric-anchored 的正迁移收益总量，不保证 full 每一个正确原子都保留，也不直接保证旧分布上的 positive transfer。** 必须同时报告 lost-new-TP、reintroduced-corrected-FP，以及补偿它们的具体计数。

### E.6 可行集合与选择顺序

\[
\mathcal F_D=\left\{
 g:
 R_D(g)\ge R_D(\mathbf1),\quad
 G_D(g)\ge G_D(\mathbf1),\quad
 \mathrm{CE}_D(g)\le\mathrm{CE}_D(\mathbf1)+\epsilon_{\mathrm{CE}}
\right\}.
\]

沿用 v2.2 的 CE 检查及 answer-token 归一化。前版设计协议建议 \(\epsilon_{\mathrm{CE}}=0.02\) nats/answer-token；实现应核对已执行配置，不能将容忍度不同的结果混为一个实验，更不能单独为 `pair_4_5` 放宽。

令 \(\widehat{\mathcal G}\) 是预先固定、已实际评估的候选集合，包含 identity：

\[
\boxed{
 g_{\mathrm{fit}}=
 \operatorname*{lexmin}_{g\in\widehat{\mathcal G}\cap\mathcal F_{\mathrm{fit}}}
 \left(L_{\mathrm{mem},\mathrm{fit}}(g),
 L_{\mathrm{PT},\mathrm{fit}}(g),
 \|g-\mathbf1\|_1\right).
}
\]

顺序明确为：**先保护 memory-supported atoms，再少丢已有正迁移，最后少改参数。** 全部相同则按预注册 manifest 顺序处理，不使用旧任务分数破同分。

第一版不将“更高 current F1”用作最后的强制选择依据。没有 memory 改善时，identity 会因零正迁移损失、零编辑量而胜出，避免悄悄退化成 current-task pruning。

### E.7 Holdout 只做一次确认

在 current holdout 上独立运行同样的参考构造、metric anchor 与统计规则，只比较已锁定的 \(g_{\mathrm{fit}}\) 和 identity，不比较整批候选。

部署要求：holdout 有非零支持；候选仍满足三项可行性检查；memory loss 相对 identity 严格下降。否则返回 identity，并记录 `NO_REPLICATED_MEMORY_EFFECT`、`UTILITY_REJECTED` 或 `INCONCLUSIVE_SUPPORT`，而不是在同一个 holdout 上改选第二名。

这些是样本上的接受条件，不是总体无损保证。应报告按独立样本/病例聚类计算的区间；仅由一两个样本支撑的改善不能包装成稳定结论。

---

## F. Concept Recognition specialization

### F.1 解析与集合计数

设官方解析后的 gold 与预测集合为：

\[
Y_x,\qquad\widehat Y_g(x).
\]

\[
TP_g(x)=|Y_x\cap\widehat Y_g(x)|,
\quad FP_g(x)=|\widehat Y_g(x)\setminus Y_x|,
\quad FN_g(x)=|Y_x\setminus\widehat Y_g(x)|.
\]

样本展示分数可以是：

\[
s_t(x;g)=\frac{2TP_g(x)}{2TP_g(x)+FP_g(x)+FN_g(x)},
\]

但选择的官方效用是：

\[
\boxed{
F_{\mu,D}(g)=
\frac{2\sum_xTP_g(x)}{2\sum_xTP_g(x)+\sum_xFP_g(x)+\sum_xFN_g(x)}.
}
\]

这与标准 micro-F1 定义一致；项目是否存在 label filtering、样本权重等额外约定，必须对照现有官方 evaluator 核实，不能仅凭本公式推断。外部定义依据：scikit-learn 官方 `f1_score` 文档。

### F.2 固定原子空间：gold 概念加反事实暴露的负概念

正原子包含全部当前 gold concepts：

\[
\mathcal A_x^+=Y_x.
\]

负原子只取参考生成确实暴露过的非 gold 概念：

\[
\mathcal A_x^-=
\left(\widehat Y_H(x)\cup\widehat Y_{\mathbf1}(x)
\cup\bigcup_{i<t}\widehat Y_{H\setminus i}(x)\right)\setminus Y_x.
\]

\[
u_g(x,c)=
\begin{cases}
\mathbf1[c\in\widehat Y_g(x)],&c\in\mathcal A_x^+,\\
\mathbf1[c\notin\widehat Y_g(x)],&c\in\mathcal A_x^-.
\end{cases}
\]

不枚举整个医学概念宇宙中的 true negatives。不因某个概念从来没有被任何参考生成，就给它一份“历史支持的抑制成功”。

候选新产生、未出现在参考原子空间内的 FP，**仍完整进入官方 micro-F1 与 FP 总数**，不能被漏算；但它不能倒过来扩充自己的 witness 集合。

### F.3 两种明确的 historical support

**Gold-concept support：**

\[
a_i^+(x,c)=
\mathbf1[c\in Y_x\cap\widehat Y_H(x)]
\mathbf1[c\notin\widehat Y_{H\setminus i}(x)].
\]

含义：H 正确生成了该 gold concept；移除历史 bank 后，这个概念消失。H 是否预测对其余概念不影响该证据成立。

**False-positive suppression support：**

\[
a_i^-(x,c)=
\mathbf1[c\notin Y_x]
\mathbf1[c\notin\widehat Y_H(x)]
\mathbf1[c\in\widehat Y_{H\setminus i}(x)].
\]

含义：移除历史 bank 后，出现一个 FP；历史 bank 的存在支持了这个错误的抑制。

两者都来自真实 finite-forward difference，不需要 concept probability、局部梯度或额外概念分类器。这是给定当前 composition 下的支持证据，不是无上下文的唯一因果归因，更不是旧任务知识已被完整识别。

### F.4 TP/FP 权重：来自完整 micro-F1 的有限恒等式

令当前划分的 full 分数为：

\[
q=F_{\mu,D}(\mathbf1),
\qquad
Y=\sum_x|Y_x|,
\qquad
N_g=\sum_x|\widehat Y_g(x)|.
\]

当分母非零时：

\[
\boxed{
F_{\mu,D}(g)-q=
\frac{(2-q)(TP_g-TP_{\mathbf1})-q(FP_g-FP_{\mathbf1})}
{Y+N_g}.
}
\]

因此采用：

\[
\boxed{\omega(x,c)=2-q\ \text{for gold atoms};\qquad
\omega(x,c)=q\ \text{for FP-suppression atoms}.}
\]

这里 \(q\) 由当前 full 直接确定并在该划分内固定，不是待调的 precision/recall 超参数。它不是模型输出的 Taylor 展开；候选 TP/FP 均来自真实生成，公式对任意有限候选成立。

**边界必须写清：**该恒等式针对完整、未加历史支持权重的计数。加入 \(\bar a\) 后的 \(L_{\mathrm{mem}}\) 是有意关注历史支持子集的损伤量，**不等于 micro-F1，也不保证降低它必然提高 micro-F1**。所以真实 micro-F1 仍是独立硬检查。

使用 full 而不是 H 的 F1 作锚，是为了不让很弱的 H 自动把 FP 抑制权重压到几乎零。fit 与 holdout 各自按同一公式计算其 full 锚值，不使用候选分数改权重。

若 Concept full 在整个当前划分上的 F1 为零，第一版将其标为 `METRIC_DEGENERATE`：此时该锚下 FP 权重为零，不能伪称已有完整的 precision–recall 保护信号。按预注册规则扩展当前数据，或返回 identity，不人工补一个可调 epsilon。

### F.5 Concept 的正迁移到底是什么

相对 H，正迁移收益可展开为：

\[
\boxed{
G_D(g)=
(2-q)\sum_x|(Y_x\setminus\widehat Y_H(x))\cap\widehat Y_g(x)|
+q\sum_x| (\widehat Y_H(x)\setminus Y_x)\setminus\widehat Y_g(x)|.
}
\]

第一项是补回 H 漏掉的 gold concepts；第二项是纠正 H 已经产生的 FP。

full 原本补回了某个概念，而 g 又丢掉它，就计入 \(D_{\mathrm{PT}}\)。只有 g 额外纠正了 full 尚未纠正的 H 错误，才计入 \(U_{\mathrm{PT}}\) 并补偿该损失。它不是“总 F1 不差就假装没有丢失正迁移”。

### F.6 两个说明性例子

**例一：整样本不正确，但历史支持明确存在。** 设 gold 为 \(\{a,b,c\}\)，H 输出 \(\{a,b\}\)，\(H\setminus i\) 输出 \(\{b,d\}\)，full 输出 \(\{b,c,d\}\)。H 并非 exact-set correct，但历史 bank 支持 gold \(a\) 与 FP \(d\) 的抑制；full 破坏了这两项，同时新增了对 \(c\) 的正确识别。若一个候选输出 \(\{a,b,c\}\)，它修复历史支持、保留新增识别，并提高真实 F1。这里不需要让 H 整条答案正确。

**例二：样本级净分相同，也可能存在概念级支持。** gold 为 \(\{a,b,c\}\)，H 输出 \(\{a,b,d\}\)，\(H\setminus i\) 输出 \(\{b,c,d\}\)。两者 sample-F1 相同，但历史 bank 在这个 composition 中支持了 \(a\)。候选 A/C 可能不给这条样本任何净支持；候选 B 仍可保留该局部正确行为，同时不保护 H 漏掉 \(c\) 的错误。

这些只是集合运算示例，不是 Med-PRISM 实验结果。

### F.7 空集合、格式和 open-set 边界

| 情形 | 处理方式 |
|---|---|
| Gold 非空，预测为空 | 全部 gold 计 FN；保留这些正原子，效用为零 |
| Gold 为空，预测非空 | 全部预测概念计 FP；可评估反事实暴露的抑制行为 |
| Gold 与预测都为空 | 对 micro 的 TP/FP/FN 计数均为零，不额外奖励大量 TN |
| 整个划分的 metric 分母为零 | 遵循官方展示约定，但校准标为退化，不自定“完美正确” |
| 候选产生新 FP | 即使不在冻结 witness 空间，也必须计入官方 metric |
| 解析错误 | 与合法空集合分开记录，不允许借解析失败获取 FP-suppression 奖励 |

同义词、大小写、标点、概念 ID、重复概念及分隔符处理，全部复用已有 official parser / normalizer。若它按概念集合评估，原子也按 canonical concept ID 去重；若它不是集合语义，应先核实而非强行套用。

不引入额外 LLM judge，不通过新 embedding matching 扩展同义词，不依据 gold 对预测做候选特有的“修正”。解析异常的 H / \(H\setminus i\) 配对不产生概念支持；解析异常的候选不能记作正确的负原子抑制。官方指标仍按原 evaluator 的规则单独计算。

附件没有提供完整解析实现，因此本文不能保证某个具体 token/别名应如何匹配。**工程验收必须先复算已有 full 与 fixed-panel 的官方 micro-F1，才能把后续差异归因于 selector。**

### F.8 对 noisy/trivial units 的控制

不枚举未暴露的负类，避免海量 TN；同一概念在一个样本内不重复计数；多个 banks 支持同一原子不重复加权；以样本/病例而不是概念原子作为不确定性重采样单位。

第一版不加 inverse-frequency weighting 或 per-sample normalization，因为那会改变 micro 指标的计数偏好。应报告支持是否集中在少数通用概念、少量图像或输出格式变化；若确实如此，作为失败诊断，而不是未经验证就叠加新权重。

---

## G. Unified task interface

| Task | 样本分数 / 原子 | 官方聚合与限制 |
|---|---|---|
| VQA | 现有 normalized correctness；若官方提供 soft agreement，直接使用它 | 按该任务原 evaluator 聚合，不擅自把 soft score 二值化 |
| Diagnosis | 单标签 correctness 原子 | accuracy；若实际任务为 multilabel，转相应多标签适配器 |
| Concept | gold-recognition / exposed-FP-suppression 原子 | 汇总 TP/FP/FN 后算 micro-F1，不平均 sample-F1 |
| Grounding | 单目标框可用已匹配目标的 IoU 效用 | 保留真实官方指标；多目标 AP 不能直接用平均 IoU 替代 |
| Reasoning | 最终答案正确性原子 | 不将推理链文字相似度视作正确性 |

`BehaviorMetricAdapter` 负责规范化、原子生成、reference support、官方计数聚合和解析有效性；MS-CRC 控制器负责 composition、可行性、选择与折叠。

**可以统一接口，不能宣称所有任务的统计语义完全相同。** 特别是需要全局排序/匹配的 AP、多目标 grounding 或复杂 reasoning 评分，需要对应 evaluator 的显式适配。第一版实际实现 Concept 与已有单答案任务即可；未接入的任务不得静默使用错误的默认指标。

Task-aware 仅指训练边界已知当前任务所使用的评估适配器；不在推理时依据输入选择参数路径，不引入 test-time task ID。

---

## H. Why this is still Med-PRISM

**History-free：**选择只读取当时合法的当前任务图文与标签、当前参数和冻结历史 banks。当前预测/概念原子只在 boundary 临时存在；下一任务开始前删除，不演变成历史逐样本 memory。

**Parameters-as-memory：**历史支持由 H 与 \(H\setminus i\) 的实际行为差产生。移除这一步就改变了 selector 的信息来源，而不只是少了一个命名。

**A-only preserved：**A 不编辑，\(A_tA_i^\top\) 保持不变；不重新启用 BA geometry。**Cumulative memory：**全部历史 bank 仍参与累积推理，历史 bank 不被关闭或动态选择。

**No routing：**所有输入共享同一组静态系数，且这些系数被折叠进当前 B。部署图不增加适配器、前向次数或路由算子；不能据此宣称模型响应完全不变，校准的目的正是改变部分行为。

16 个 FP32 系数原始负载为 64 bytes；折叠后甚至不需要作为推理 sidecar。可以保存配置 hash、系数与聚合诊断，但不能保留当前样本的逐条输出并在以后调用它们。

---

## I. Why rank-1 / rank-group matters

\[
BA=\sum_{e=1}^{16}b_ea_e^\top
\]

仍意味着普通 rank-16 LoRA 能实现相同操作。**rank-1 不提供额外表达能力；它提供可执行、可折叠的干预坐标。**

已有 `off_07` 与 `pair_4_5` 说明：至少在被检查的分组与 checkpoint 上，不同 rank groups 对 stability/plasticity 的作用并不均匀。全局缩放在附件面板上呈明显 trade-off，而选择性编辑存在更好的已观察点。来源：输入设计任务，第 420–552 行。

但这些结果尚未证明：单个 rank-1 都具有稳定语义、rank-group 优于 layer-only、group 07 在不同 seeds 上始终有害，或任意低秩基变换后同一编号仍对应同种功能。

因此第一版保留 16 groups，不拆 q/v，不上 32 groups，更不独立搜索 1152 个原子参数。分解基保持冻结；跨 seed 的同编号比较只是布局层面的稳定性指标，不是语义专家对齐。后续只有通过 granularity 消融，才能主张 rank 粒度优于普通 layer scaling。

---

## J. Minimal implementation plan

### J.1 优先复用已有 v2.2

首先加载已执行实验的完整 panel manifest、分组映射、checkpoint 和 evaluator 配置，不能凭文档中的 group 编号重新猜索引。保留 trainer 与 LoRA wrapper；只替换 metric/witness scoring 和 positive-transfer 约束。

有兼容的当前逐样本生成缓存时，直接解析并重评分；只有汇总 F1 时，不能恢复概念支持，必须重新运行当前数据上的必要 composition。**无需旧数据，也无需旧数据摘要。**

### J.2 第一轮采用固定面板选择，不扩大搜索

\(\widehat{\mathcal G}\) 直接采用原预注册 finite panel 中的完整固定系数列表，包括全部单组干预和原预注册配对，不只保留后来表现好的组合。identity 必须存在。由旧任务结果选择出来的额外 mask、人工偏好的 `off_07` 优先规则不得加入。

对所有候选计算同一组 \(L_{\mathrm{mem}},L_{\mathrm{PT}},G,F_\mu,\mathrm{CE}\)，按 E 节选择。若复用原 coordinate-search 代码，评分接口相同，但**本轮 gate 的主成绩使用固定候选集合**，以便把变化归因于 selector。

第一版不因 identity 获胜而不断增加组合。若后续确需搜索 \(0.5\) 或多组交互，应对所有相应 groups 使用一致预注册规则，并在 fresh validation 前冻结为新协议。

### J.3 输出与验收

工程应产出：current-only 系数与 checkpoint hash；各 bank 的 gold/FP 支持量；去重后支持量；每候选可行性与淘汰原因；正迁移丢失/补偿计数；current holdout 的一次性接受结果；折叠前后同一系数配置的输出一致性。

当前临时的逐样本审计表与跨任务持久摘要分开管理。进入下一任务前只保留最终参数、系数和非逐样本聚合报告。

### J.4 成本与默认设置

保持 rank=16、72 wrappers、16 groups、现有 key/shared 权重、现有 generation 与 CE 配置。初始 current fit/holdout 可以沿用 128/64；支持不足时按事先固定规则扩展，而不是按旧 dev 偏好挑当前样本。

若有 K 个未缓存候选，核心成本是 K 次 composition-dataset generation，加 H、各 \(H\setminus i\)、full 参考；原子数量增加不会产生逐原子模型调用。真正耗时仍是 generation，不应将一次完整数据集评估写成“一次廉价 forward”。

折叠从不可变原始 B 构造，不能累计缩放；名义 LoRA rank 与 scaling 不因列被置零而变化；系数改变后不得复用依赖旧模型状态的 KV cache。推理图是否相同可由实现检查，实际 latency 仍应测量。

---

## K. Fast v2.3 gate

### Gate 0：Metric consistency

先用官方解析器复算现有 full/current panel 的任务指标。核对 canonical concept ID、重复项、空集、异常输出及 batch 聚合方式。复算不一致时，先解决 evaluator 问题，停止解释方法收益。

### Gate 1：Current observability

在 fit/holdout 分别报告：正支持、FP 抑制支持、支持样本数、支持概念数、各 bank 覆盖、重复支持和 full 已破坏的支持量。

以每个独立样本的支持质量 \(m_x=\sum_a w(x,a)\) 报告：

\[
n_{\mathrm{eff}}=\frac{(\sum_xm_x)^2}{\sum_xm_x^2}.
\]

ESS 是集中度诊断，不是把概念原子当独立样本得到的置信度。不要继续机械使用 v2.2 的“32 条 exact-correct witness”门槛；它的计数单位已经失效。

区分 `NO_SUPPORT`、`SUPPORT_BUT_NO_EXPOSED_HARM` 和 `SUPPORT_WITH_REPAIR_OPPORTUNITY`。只有最后一种才有理由期待当前保护目标产生非平凡编辑。少量高度集中的支持记为 `INCONCLUSIVE_SUPPORT`；可以按预注册方式增加当前数据，不能从旧数据补 witness。

### Gate 2：Current-only nonidentity selection

完整候选集合只用 current fit 排名，锁定 \(g\)，再做 current holdout 的一次性检验。要求 memory loss 在 fit/holdout 都下降，official task utility 不下降，正迁移净收益不下降，CE 合格。

如果 g 非 identity 但所有系数都相同，记录为 `GLOBAL_ONLY`：可能有校准收益，但不能主张已识别出选择性 rank-group 作用。如果所有候选与 identity 的 memory loss 相同，不能用更高 Task3 F1 强行宣布历史保护成功。

区间应按独立图像/病例做 paired resampling，并重新汇总 TP/FP/FN；不能对 sample-F1 求均值后计算一个冒充 micro-F1 的区间。小样本只支持方向检查，不支持 1 pp 级非劣性结论。

### Gate 3：已揭盲 Task2 development 上的机制方向

锁定 g 与选择日志后，再评估已有 Task1/Task2 development 集。要求至少看到 Task2 真实准确率改善，报告 correct→wrong / wrong→correct；检查 Task1 没有明显额外损伤，并报告 Task3 官方 micro-F1。

可以沿用前版约 +3 pp 的 Task2 实用提升目标，但应把“方向为正”和“达到实用幅度”分开报告。该目标是工程决策门槛，不是预期收益。任何结果都不得反过来改选这个 boundary 的 mask。

**本步骤只能称 development/mechanism validation。** Task2 dev 已经参与研发；即使本次运行前锁定了 g，也不能恢复这个集合对整个方法设计的无偏性。来源：输入设计任务，第 795–833、1221–1248 行。

### Gate 4：Matched global scaling

沿用预注册 \(\alpha\) 网格，使用相同 current utility、正迁移与 CE 可行性规则，得到 current-only 选择的 scalar baseline；另报告 development 上该有限网格的诊断性上包络。

比较应使用同一个 plasticity floor，例如“不低于未校准 full 的 current micro-F1”，而不是把严重损失 Task3 的 \(\alpha=0\) 与保留 Task3 的 selective mask 当作等预算对照。若没有完全匹配的测量点，报告离散 Pareto 关系，不对曲线线性插值制造一个未测配置。

只有非均匀 current-only g 在相同可行性条件下显示真实旧任务收益、且不能由 global scaling / current-only utility control 同样解释，才进入 fresh pilot。

### 失败后如何定位

| 结果 | 解释与处理 |
|---|---|
| 原子支持仍为零 | 当前输入缺少可见历史支持；不加任意正权重伪造信号 |
| 有支持，但 full 没破坏它 | 当前保护目标没有可修复对象；identity 合理 |
| Fit 可选、holdout 不复现 | 选择噪声或覆盖不足；不在同一个 holdout 上换 mask |
| 当前支持改善、Task2 development 不改善 | 当前 witness 与历史安全性不迁移 |
| 仅 micro-F1/CE 改善，memory 指标无选择力 | 更接近普通 current-task calibration |
| 与 global scaling 或去历史支持版本相当 | 不足以支持 rank-selective / parameter-memory 增量主张 |

leave-one-bank 还可能漏掉冗余记忆：两个历史 banks 都足以支撑同一行为，逐个移除均无变化。只有诊断确实指向这种情况，才可另立开发实验增加一次 \(W_0+S_t\) 的 whole-history-off 前向，检查集体历史支持；不把它默认堆入主版本。若所有合法当前干预都缺乏区分信息，task-aware weighting 本身无法创造缺失的观测。

---

## L. Fresh formal validation plan

### L.1 数据与训练隔离

在 development gate 后冻结：parser/evaluator、原子定义、TP/FP 权重公式、正迁移规则、候选 manifest、分组、CE 容忍度、current fit/holdout 选择与扩展规则、identity fallback。

重新执行 Task1→Task2→Task3，至少三个 fresh training seeds；使用 untouched test，matched no-geo 与完全相同的基础训练超参数，不做 per-seed selector 调参。

**Fresh seed 不能单独消除测试集泄漏。** 真正必要的是 untouched evaluation 与冻结的方法选择。原已揭盲 Task2 dev 继续保留 development 标记。

各 boundary 的校准发生在该任务数据仍合法可访问时；下个任务开始后不回访其校准缓存。no-geo 与新方法采用匹配的数据划分；若为校准预留数据，应在对照中同样预留，避免改变 trainer 训练预算。

### L.2 主指标与行为转移

设 \(R_{t,i}\) 是任务 t 提交后在任务 i 的官方分数，统一到 0–100：

\[
\mathrm{Avg}_T=\frac1T\sum_{i\le T}R_{T,i},
\qquad
\mathrm{BWT}_T=\frac1{T-1}\sum_{i<T}(R_{T,i}-R_{i,i}).
\]

不同任务可以使用其既定官方指标，但“accuracy 与 micro-F1 的平均”必须明确为 benchmark score average，不写成统一准确率。BWT 使用每个方法自己 boundary 提交后的 \(R_{i,i}\)，同时记录 pre/post-calibration 分数，避免把早期校准降低起点隐藏在更好的 BWT 中。

主报告包含 Task1/Task2 retention、Task3 plasticity、Concept micro-F1、Avg/BWT、边界前后变化以及 boundary 成本。

对单答案任务报告 sample correct→wrong / wrong→correct。对 Concept 改为报告：保留/丢失 TP、新恢复 TP、新增/消除 FP，以及新增正确原子的丢失和补偿；exact-set transition 仅作辅助，不再成为主要行为分组。

既要相对历史任务 teacher 计算总 forgetting，也要相对同一 shared 下的 no-current-private 参考计算新 bank induced change，两者分开。

### L.3 系数稳定性

每个 seed 报告非一系数个数、编辑总量、选中 depth/rank groups、支持覆盖与 holdout 接受状态。可以报告同布局 Jaccard，但应说明各 seed 的 rank 列并没有天然语义对齐。

更有解释力的是各 depth band 的编辑质量、不同 groups 的 finite behavioral effect 分布，以及算法改善方向是否稳定。不能把“再次选中 07”当作成功的必要条件。

未通过 selector 的 seed 也纳入总体统计，按 identity 部署；不只展示能够产出非平凡 mask 的 seeds。

---

## M. Necessary ablations

| 对照 | 回答的问题 |
|---|---|
| No-geo | 是否有真实增量 |
| Global alpha，使用相同 current feasibility | 是否只是统一缩小 P_t |
| 严格 current-task-only rank calibration | 仅优化当前官方指标与 CE、不运行历史 ablation，能否同样有效 |
| 原 v2.2 binary exact witness | 是否确实修复了 structured-task 的准入塌缩 |
| Sample-soft witness（候选 A） | 简单把 exact 换成 sample-F1 是否已经足够 |
| 完整 task-aware atomic witness | 主方案 |
| 去 historical support | 保留 H 的正确原子和同一正迁移约束，但取消 H vs H\i 筛选，检验 bank support 的作用 |
| 随机系数置换 | 固定系数 multiset 与编辑量，检验“改哪组”是否重要 |
| 原子零损失正迁移约束 vs 本文 gain ledger | 是否因逐原子禁止任何丢失而再次产生可行性塌缩 |

所有 mask-selection 对照复用同一候选预算、current 数据与 evaluator。严格 current-only 对照不能偷偷调用 H 构建正迁移集合；去 historical-support 对照则有意保留 H，用来单独隔离 leave-one-bank 支持的贡献，两者不能混称。

候选 C、等权 TP/FP、layer-only 或更细 grouping 作为第二阶段消融。若现有 granularity 没有明确收益，不上更大的网格。TPM/RCWP 不属于本轮必要组合。

---

## N. Theory / proposition candidates

### N.1 原子支持可以严格超出 exact-set witness 的覆盖

存在数据使得对所有 x，\(\mathbf1[\widehat Y_H(x)=Y_x]=0\)，但存在 \((x,c,i)\) 满足 \(a_i^+(x,c)=1\) 或 \(a_i^-(x,c)=1\)。F.6 已给出构造。

因此整样本 exact witness 为空，并不能推出原子 witness 为空。该命题证明的是行为粒度修复能够保留被丢弃的信号，**不证明在真实 Task3 上支持一定充足**。

同时，\(w(x,a)>0\) 必须对应至少一个历史参数移除导致的效用下降；所以该目标在定义上不等价于任意 current-task utility pruning。是否带来独立收益仍由去历史支持消融回答。

### N.2 Micro-F1 的有限计数恒等式与正迁移记账

令 \(q=2TP_1/(Y+N_1)\)，且 \(N=TP+FP\)，则：

\[
2TP_g-q(Y+N_g)
=(2-q)(TP_g-TP_1)-q(FP_g-FP_1).
\]

除以 \(Y+N_g\) 得到 F.4 的精确式。它不要求小扰动，也不要求网络可微。

此外，由 \(z=[z]_+-[-z]_+\)，逐原子求和可得：

\[
G(g)\ge G(\mathbf1)\iff D_{\mathrm{PT}}(g)\le U_{\mathrm{PT}}(g).
\]

这准确界定了“允许等价修正替换”的保护范围，不将总收益保护偷换成逐个正确概念不丢失。

### N.3 可折叠性与选择性编辑的 Pareto 关系

\[
B_t\operatorname{diag}(g)A_t=\widetilde B_tA_t.
\]

所以全部 A 及 A-only 交叠项不变，推理可使用原结构。

在理论系数族中：

\[
\mathcal G_{\mathrm{global}}=\{\alpha\mathbf1:\alpha\in[0,1]\}
\subseteq
\mathcal G_{\mathrm{selective}}=[0,1]^{16}.
\]

若 S 是历史表现、P 是当前表现，则在相同 plasticity floor b 下，完整 selective family 的最优 S 不低于 global family：

\[
\sup_{g:P(g)\ge b}S(g)
\ge
\sup_{\alpha:P(\alpha\mathbf1)\ge b}S(\alpha\mathbf1).
\]

严格大于需要额外函数结构或实验支持。附件 `off_07` 是已测有限网格上的正证据，**不能证明优于所有未测的连续 alpha，更不能证明当前 selector 能达到理论最优**。来源：输入设计任务，第 508–556 行。

### N.4 经验安全不等于历史分布保证

identity 在 fit 可行，按指定字典序选择不会提高 fit memory loss，并满足经验 current utility / gain / CE 约束；holdout 再检验这些性质。

但旧输入上不存在无条件保证：可以构造两个具有相同当前数据、参数及当前所有有限前向观测的环境，使两个候选仅在未观察的旧输入上产生不同错误，而两个环境的旧分布恰好偏向不同输入。current-only selector 在两者中只能返回同一个选择，无法仅凭这些观测保证同时最优。

这里缺少的是从“当前历史支持原子”到“旧任务风险”的覆盖或排序迁移假设。该假设只能通过预先锁定的 selector 在 fresh sequences / untouched test 上得到支持，不能通过一个更复杂的权重公式自动获得。

### 创新性定位

低秩正交持续学习已有 O-LoRA；参数隔离不是 v2.3 的新贡献。参考：Wang 等，*Orthogonal Subspace Learning for Language Model Continual Learning*，Findings of EMNLP 2023。

micro-F1 的计数公式与低秩列缩放也不应各自包装为新原理。这里可检验的增量是：**将历史参数的有限行为支持、任务合适的原子粒度与保留正向修正的静态提交选择结合起来，解决已经观察到的 current-only selector 失效。** 若效果由 ordinary current-only calibration 完全解释，就应缩小这一主张，而不是继续扩大命名。

---

## O. Paper-ready method narrative

Med-PRISM 使用训练后的 private LoRA banks 作为累积参数记忆，并通过 A-only key isolation 降低新旧 private key 的参数交叠。然而，参数隔离并不意味着新 private 在历史输入上功能隔离：给定边界的 composition audit 表明，新增 private bank 可以成为主要行为干扰来源。实际 rank-group 干预进一步显示，新 bank 内部存在对稳定性与可塑性作用不同的可折叠分量，因此关键问题从“是否存在有用干预”转向“如何在不访问旧数据的条件下选择干预”。

我们在保持训练目标、shared/private decomposition 与累积推理不变的前提下，引入 Task-Aware Memory-Supported Counterfactual Rank Calibration。该方法在任务边界冻结模型，以当前任务数据执行历史 bank 移除和当前 private 秩组干预，并将保护单位适配到真实任务行为。对于 Concept Recognition，方法不要求整条概念集合完全正确，而是分别识别由历史 banks 支持的 gold-concept recognition 与 false-positive suppression。其权重由当前完整模型的 micro-F1 锚定，并使用真实有限生成结果评估候选对这些原子的损伤。

为避免保守地复制历史参考或冻结当前完整模型的所有输出，校准同时约束当前官方任务指标、原有 answer-token loss，以及相对历史参考的正向修正收益；已有修正可以被等价的新修正替换，但所有丢失与补偿均显式记录。方法在满足这些条件的静态候选中优先减少历史支持原子的退化，并通过独立 current holdout 决定提交或回退 identity。最终系数折叠到当前 LoRA 的 value 因子，历史 private 与全部 key 因子保持不变，不保存历史逐样本数据，也不增加推理路径或路由。其对真实历史任务的有效性由固定协议下的 fresh sequential training 与 untouched test 验证，而不由开发集上的 oracle mask 或局部代理指标替代。

**最终研究判据：不仅要让 witness 从零变成非零，还要让它在相同 current utility 下提供超出 current-task-only calibration 的选择信息。只有 current-only 选出的 mask 在新的历史评估上稳定更安全，v2.3 才完成了这次修复。**

---

### 文档状态与推导检查

本次仅进行了合成数值检查：2,000 组随机计数/原子效用验证了有限 micro-F1 恒等式与正迁移 ledger 恒等式。没有执行 Med-PRISM generation、系数搜索或模型训练；没有从附件反推出 v2.3 应选哪一组。文中的具体接受结果、系数和模型收益均待工程实现后验证。
