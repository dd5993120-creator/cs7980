# 聊天记录（第一部分）：10/1 – 10/15

由对话系统保存的原始记录自动转换而来，内容逐字保留。
只保留双方的文字内容；Claude 的内部思考和工具调用过程已省略，本轮生成的文件列在每轮回复末尾。

---

## 第 1 轮 · 用户（2026-10-01 10:54 UTC）

附件：13320_Dreaming_Up_Authors_Fact (1).pdf

请帮我分析一下这篇论文的主要观点和论据。

## 第 1 轮 · Claude

这篇匿名 ACL 投稿研究一个很具体的问题：大模型回答“某篇论文的作者是谁”时，出错的概率是否取决于这篇论文在训练数据里出现得多不多？商用模型的训练数据不公开，所以作者用引用数作为“事实流行度”的代理，并假设引用越少、幻觉越多。下面先梳理论文的四个主要观点和论据，再评估论证的强弱。

## 主要观点与论据

### 观点一：引用数可以代表论文在训练数据中的流行度

这是全文的前提。作者分层抽取了 6,048 篇论文，做了两项验证。

第一项是 Google 命中数。作者用“标题 + 作者”作查询词，剔除每个引用区间内高于 80 分位的离群值后，平均命中数随引用数上升。

第二项是 Infini-gram 计数。作者统计“标题 AND 第一作者”在 OLMo2（6.1T tokens）和 RedPajama（1.4T tokens）两个开放语料中的出现次数，结果也随引用区间上升。

作者承认这两个信号本身很不稳定：Google 容易被短标题误匹配，n-gram 计数对大小写等格式极其敏感。因此他们主张改用更稳定、易得、定义一致的引用数。

### 观点二：幻觉率随引用数增加而显著下降（核心结论）

数据构建过程如下：

- 从 Open Academic Graph 随机抽取约 1,000 万篇论文，用 Semantic Scholar 补全引用数和规范化的作者名。
- 经过文献类型、语言、作者名格式、跨库一致性等五道过滤，保留 192 万篇。
- 按 8 个学科 × 13 个指数间隔的引用区间分层，抽出 9,108 篇用于评测。

评测方法如下：

- 被测模型是 GPT-4o、DeepSeek-R1 和 Claude Sonnet 4.5，提示词要求只输出作者名单。
- 按“姓氏 + 名字首字母”做模糊匹配，统计匹配（M）、编造（H）、遗漏（U）的人数。
- 主指标 HR₂ 就是 Jaccard 距离，即错误人数占全部人数的比例。

主要结果：

- 0–2 次引用的论文，三个模型的 HR₂ 都超过 0.98。
- 最高区间（4097 次以上）降到 0.364（GPT-4o）、0.362（DeepSeek-R1）和 0.169（Claude）。
- 24 个“学科 × 模型”组合的 Spearman ρ 全部为负（−0.23 到 −0.55，p < 0.001）。
- 模型中 Claude 整体最好，GPT-4o 最差。学科中材料科学和化学最难，计算机和物理相对最好。

稳健性检验相当充分：

- 五种 HR 指标结论一致，编造率 HR₄ 和遗漏率 HR₅ 在每个区间几乎相等。
- 规则评估与 LLM 评估的差异很小（RMSE 0.067）。
- 控制作者数、标题长度、年份和学科后，回归显示引用每翻一倍，幻觉几率下降 34–36%。
- 学科随机截距模型的 ICC 约 2%，说明学科本身解释的差异很少。
- 换三种提示词措辞，相关系数没有显著差异。

### 观点三：低引论文上的失败主要是“想不起来”，而不是“不知道”

证据来自 GPT-4o 上的四选一识别实验。正确选项是真实作者名单，三个干扰项是从同学科作者池里随机抽出的等长名单。

- 0–2 次引用的论文，开放式回答几乎全错，但经机会校正后的选择准确率有 79%。
- 在开放式回答基本失败（HR₂ > 0.5）的 8,342 篇中，93.9% 仍能选对。

作者据此认为，模型“知道”的远多于它在自由生成时能说出来的。

### 观点四：模型在开始生成之前，内部就已经编码了论文的流行度

闭源模型拿不到内部激活，作者改用 Qwen3-32B 和 Mistral-Small-3.2-24B。

- 用提示最后一个 token（P1）的隐藏状态做岭回归，预测 log(引用数+1)，交叉验证 R² 分别为 0.615 和 0.585。只用学科、年份、作者数、标题长度等元数据时，R² 只有 0.039。
- 打乱标签后信号消失；限定在同一学科内，信号依然存在。
- 按解码出的流行度把论文分成十组，HR₂ 从最低组的约 0.99 降到最高组的 0.84–0.88。
- 中介分析估计，这个内部信号承载了引用与幻觉之间关联的 67.3%（Qwen）和 51.9%（Mistral）。

还有几项辅助发现：

- 排位越靠后的作者，越容易被遗漏或编造。
- 作者越多，HR₂ 反而越低。原因是模型不管真实人数都倾向多报名字：单作者论文平均被报出 2.5 个名字，其中 2.3 个是编的。
- 编造名字的族裔分布和数据集基线大体一致，只有轻微偏差（Cramér's V 约 0.03–0.10）。

## 论证评估

### 强项

任务选得很巧。作者归属有确定答案，答案是多人名单，引用数天然提供了流行度梯度，又直接对应“AI 编造参考文献”这个现实问题。数据从真正的随机样本出发，过滤流程透明。指标考虑了部分答对的情况，还拆出了编造和遗漏两部分。各项稳健性检验都比较规范，局限性部分也算坦诚。观点二是全文证据最扎实的部分。

### 薄弱之处

**第一，核心效应集中在极少数高引论文上。** 按表 1，引用 ≤128 次的论文占全部数据的 95.7%。在这些区间里，GPT-4o 的答对程度（1−HR₂）都不到 0.06（表 A34），曲线基本是平的。真正的下降只发生在前 4% 左右的论文。

分层抽样有意放大了高引区间，方便画出曲线，但这也意味着 ρ ≈ −0.4 不代表真实论文分布下的相关强度。更贴近实际的说法是：对绝大多数论文，这些模型几乎完全答不对。

附录 A8.1 说“最著名的前 1% 论文仍有 36.4% 的错误率”，这也不准确。36.4% 对应的是 4097+ 区间，全库只有 178 篇，约占万分之一。按表 1 和表 A34 粗略加权，真正的前 1%（大约三四百次引用以上）里，GPT-4o 的 HR₂ 在 0.8 以上。

**第二，代理指标的验证比正文呈现的弱。**

- Google 命中数：剔除各区间前 20% 后（这个阈值是作者“经过几次尝试”、为了压低变异系数而选的），最低 7 个区间（0–128 次引用）的均值都在 2.7–3.4 之间，看不出趋势，129 次以上才开始上升（表 A11）。
- Infini-gram：除 4097+ 外，每个区间的中位数都是 0；直到 257–512 区间，连 75 分位数都是 0。均值主要被个别论文拉动，比如 257–512 区间的均值 33.17，约七成来自一篇出现了 11,763 次的论文。重新抽样后，各区间排序的一致性只有 0.39（Spearman）。

论文一边用这两个信号证明引用数可靠，一边又说这两个信号本身不可靠，逻辑上有张力。

更可惜的是，这个 6.1T tokens 的 OLMo2 索引，从规模看应该是 OLMo 2 32B 的完整训练数据（文中称之为“指令微调语料”并不准确）。作者本可以直接评测 OLMo 2，在单篇论文层面把幻觉率和真实训练频次对应起来。这恰好能正面解决论文开头说的“训练数据不可得”的障碍。

**第三，选择题实验的干扰项设计撑不起“想不起来”的结论。** 干扰项是从整个学科作者池里随机拼成的名单，并不是真实存在的团队。模型可能不需要认识这篇论文，就能靠别的线索选对：

- 真实团队通常内部一致：同一机构、同一语言背景，姓名格式也接近。论文自己的少样本示例里就有全中文名、全韩文名和全英文名的团队。随机拼凑的名单则往往族裔和格式混杂。
- 干扰池取自分层后的评测集，高引论文的知名作者占比偏高。模型可以凭“这位名人不做这个方向”排除选项。

一个佐证是：0–2 次引用的论文在 OLMo2 语料中平均只出现 0.02 次，识别准确率却有 79%。这和第 3 节“低引论文几乎不出现在语料中”的结论明显矛盾。

要支持“知道但想不起来”，至少需要几组对照：

- 给虚构的标题配同样的选项；
- 去掉标题，只给选项；
- 用同学科、同年代其他论文的真实团队做干扰项。

退一步说，即使识别优势是真的，它反映的也可能只是“这个题目大概属于哪个研究群体”的粗略印象，而不是对具体作者的记忆。“想不起来还是不知道”这个二分本身就太粗了。另外，这个实验只在 GPT-4o 上做过。

**第四，“机制证据”的说法过强。**

- **缺少纯文本基线。** P1 的隐藏状态包含整个提示，自然也包含标题的含义，而标题的主题和措辞本身就能预测引用数（热门方向引用更多）。作者的对照只有学科、年份、作者数、标题长度这几项粗糙的元数据。合适的基线是用一个不太可能记住这些论文的小模型给标题做嵌入，或直接用 TF-IDF 做同样的回归。大量低引论文几乎不在训练语料中，所以 R² ≈ 0.6 里，有多少来自“记得这篇论文”、多少来自“这个标题看起来像高引论文”，目前分不清。作者在附录里也承认可能存在更细的主题或用词混杂。
- **中介分析算不上机制。** 所谓“中介变量”本身就是为拟合引用数而训练出来的预测值，和引用数高度相关。论文也没有做任何干预实验，比如沿这个方向调整激活、看幻觉率会不会变。所以 67% 和 52% 只是统计上的拆分。
- **样本和模型选择有限。** 这两个开源模型几乎总是答错，最高十分位组的 HR₂ 仍有 0.84–0.88，可比较的空间很小。而且行为实验和探针实验用的不是同一批模型。DeepSeek-R1 其实是开放权重的，文中把它归为“闭源模型”不准确，只是规模大、跑起来成本高。
- **遗漏了相关工作。** 例如 Gottesman & Geva（EMNLP 2024）的 KEEN 用实体的内部表示预测模型对该实体的掌握程度；Ferrando 等人（ICLR 2025）找到了区分“已知/未知实体”的表示方向，并通过干预证明它会影响模型是拒答还是编造。

**第五，实验测的是“被迫作答时的错误率”，不是模型自发编造的倾向。** 提示词要求只输出作者名、不许附加任何说明，模型没有回答“不知道”的余地。遗漏的作者也被算进“幻觉率”。文中只在 100 篇的试点里报告了格式合规率，没有报告各模型在正式实验中的拒答情况。

面对确实不认识的论文，这种设计几乎是在逼模型编造。既然论文自己发现模型内部编码了流行度信号，更有现实意义的问题恰恰是：如果允许弃答，模型能不能利用这个信号知道自己不知道？

**第六，部分推论超出了数据。** 附录 A8.3 从“较新模型的负相关更强”推出“能力提升会放大对训练频率的敏感性，单靠扩大规模解决不了长尾幻觉”。但所有模型在低引区间的错误率都已接近 0.98 的上限，任何在高引论文上表现更好的模型，相关系数都必然更负。这是天花板效应造成的，说明不了规模的作用。况且这三个模型在数据、架构和推理方式上处处不同，也没有受控的规模对比。结论里“普遍的”“根本属性”这类说法，同样只建立在三个模型、一个任务之上。

**此外还有一些细节问题：**

- 第 4.1 节说各分层 HR₂ 的 95% 置信区间平均为 ±0.17。这恰好等于 1.96 × 标准差 0.087，描述的是单篇论文的离散程度。每层 100 篇时，均值的置信区间应约为 ±0.017（这个错误反而低估了他们自己的精度）。
- 附录 A7.4 正文写 Mistral 顶部十分位的 AUROC 为 0.930，表 A39 则是 0.922。
- 主实验的解码设置没交代清楚，包括温度、Claude 是否开启扩展思考、R1 的推理预算。
- 匹配规则只要求姓氏和首字母一致，Wang、Li、Zhang 这类常见姓氏可能碰巧匹配上，但文中没有给出随机配对的基线，例如拿模型对 A 论文的回答去匹配同学科 B 论文的作者。

## 总体评价

简单说，这篇论文证据最扎实的结论恰恰最不新颖，最新颖的结论证据最薄弱。

“越冷门的事实越容易答错”已经被 PopQA（Mallen et al., 2023）、Kandpal et al.（2023）、Head-to-Tail（Sun et al., 2024）等工作确立。本文的主要贡献是把它推广到多人名单的作者归属任务，并提供了一个规模可观、设计周到的数据集。而“想不起来而非不知道”和“机制性编码”这两个最有新意的论点，都还缺少关键对照。

无论是审稿还是引用，比较稳妥的做法是把行为结果当作可靠的经验证据，把选择题和探针部分当作有待检验的假说。从实用角度看，这篇论文最直接的启示反而最朴素：对绝大多数论文，即使最强的模型也基本无法凭记忆说出正确作者，LLM 给出的任何文献信息都必须查证。

---

## 第 2 轮 · 用户（2026-10-02 07:43 UTC）

附件：Enhancing Memory Of LLMs - HackMD.pdf

这篇论文其实是一个引子，希望我从这里开始研究增强LLM记忆的问题。你也能看出，我们这次希望研究的是什么呢？llm里面其实有很多知识包含在参数里面，但是这些知识在正常生成的过程中却完全提取不出来。下面这个介绍文档其实也很有参考价值，你先来帮我分析分析。

## 第 2 轮 · Claude

能看出来。上一篇论文回答的是"模型知不知道"，这个项目要回答"怎么把它知道的东西拿出来"。借用记忆心理学的说法，这是可存储性（availability）与可提取性（accessibility）的区别：知识存在参数里，不等于正常生成时能调出来。

文档把问题分成了三层。第一层是把"生成–识别差距"测准，弄清到底有多少可提取的空间。第二层是找缩小差距的方法：白盒的有激活引导，黑盒的有"先生成候选、再让模型选"和首字母提示。第三层是画出这块空间怎样随流行度变化，找到"提取不出来"和"根本没存下"的分界线。下面先看参考文献把问题放在什么位置，再逐一分析三个方向，最后说我认为该先做什么。

## 参考文献划出的位置

参考文献其实代表两种竞争的视角。

[3] 和 [4] 是"容量"视角。IKP 用分成 7 个冷门等级的 1,400 道事实题，专门挑无法靠推理推出的知识，发现事实容量随参数量呈对数线性增长。[4]（Morris 等，2025）估计 GPT 类模型每个参数大约只能存 3.6 bit。按这派的逻辑，长尾事实很可能确实没存下。

[1] 和 [2] 是"提取"视角，认为很多错误是知识存了却调不出来。你们的项目正好站在两者的交界处。

[2] 是和本项目最接近的工作，需要仔细对照。它是 Google Research 和 Technion 今年 2 月公开的论文，构建了 2,150 条维基百科事实，每条配十道题，分别考编码、回忆和识别。

值得注意的是它对"编码"的定义：在类似预训练的语境里，让模型补全命题或回答语境化问题，不泄露答案，能复现就算已编码。这个定义基于生成，不受选择题干扰项质量的影响。

它的主要结论有四条：
- GPT-5 和 Gemini-3-Pro 编码了 95–98% 的事实，却有 26–34% 无法直接回忆，开启思考后仍有 11–12% 答不出。
- 冷门事实的编码率和热门事实差不多，回忆率差距却大得多。
- 反向问题在选择题里不更难，甚至更容易，在开放式生成里却明显更难。
- 在思考优化的模型上，思考能找回 40–65% 已编码但答不出的事实，对未编码的事实只能找回 5–15%。

文档里要超越的 CoT 基线就是这一组结果。

另一篇建议补进参考文献的是 [2] 部分作者的前作 Inside-Out（Gekhman 等，2025）。它把"知识"定义为：在一对正确、错误候选答案中，正确答案被排在前面的比例。用输出概率打分算外部知识，用中间层计算打分算内部知识。结果是内部知识平均比外部高约 40%；而且有些答案反复采样也几乎不出现，可一旦放进候选就会排第一。文档中"用探针确认正确答案可以线性解码"这一步，可以直接借用这个框架；后一个发现则决定了方向二的上限。

所以，本项目的新意不能是"回忆是瓶颈"本身，[2] 已经占了这个论点。可以占的空间主要有三块：

- **更长的尾。** [2] 比较的冷门事实是流行度最低的 20%，但那仍是维基百科上的事实。你们的数据一直延伸到 0–2 次引用、在开放语料里几乎没出现过的论文。[2] 的"编码接近饱和"迟早会在某处失效，找到这条边界就是方向三。
- **白盒方法。** [2] 和 Inside-Out 都只诊断、不干预，激活引导是新的。
- **任务结构。** 作者名单是多实体答案，而且"由标题问作者"很可能在相当程度上是个反向问题，下文细说。

## 对三个方向的分析

### 方向一：激活引导

文档末尾的流程基本是 ITI 的做法：逐个注意力头训练线性探针，找"真/假"最好分的位置，取两类激活的均值差作为方向，推理时按比例加回去，扫描强度并在留出集上评估。

流程本身没问题，但有个概念问题要先想清楚。引导向量对所有输入加的是同一个方向，它本身不带"这篇论文作者是谁"的信息，只能改变模型的某种"模式"。所以方向一真正的假设是：具体信息已经在模型里，只是某种全局的"检索模式"没被激活。

这个假设成不成立并不显然。TruthfulQA 主要考模型能否抵制常见误解，正确答案模型往往本来就知道，ITI 的成功更像是在两个已知选项之间换了一个。强度调大时，真实性的提升还有一部分是靠回答变含糊换来的。从零生成一个冷门作者名，难度不在一个量级上。

更具体的风险在于对比样本怎么构造。文档把"真/假"和"回忆成功/失败"写成可以互换的两种标签，但两者得到的方向含义完全不同。

"回忆成功/失败"是按问题分组的，而模型答对的问题集中在高引论文上。所以两类的均值差，很大程度上就是上一篇论文发现的那个"流行度/熟悉度"方向。Ferrando 等人（ICLR 2025）发现，沿"已知实体"方向引导，会让模型对陌生实体自信地编造，而不是变得更准。也就是说，沿这个方向推，最可能的结果是幻觉增加。

更好的做法是按 CAA 的方式构造"真/假"：同一个问题，后面分别接正确和错误的作者名单。两类样本问题相同，流行度自然被控制住，得到的方向才更接近"这个答案是对的"。这样做还有个实际好处：正样本是直接把正确答案写进去，不需要模型自己答对，避开了开源模型正确样本太少的问题（上一篇论文里，Qwen3-32B 生成的名字只有不到 3% 是对的）。评估时要把编造率和遗漏率分开报，因为引导可能只是在两者之间做交换。

至于 CAA、加权均值、中位数等稳健估计和 PCA 之间的比较，我认为是次要问题。Marks 和 Tegmark 的 Geometry of Truth 发现，均值差方向往往比逻辑回归探针方向更有因果作用。稳健估计在低引样本噪声大时可能有帮助，适合做消融，但不宜当主要贡献。

文档里的另一个问题更有意思："引导方向是否依赖引用数"。可以在各引用区间分别求方向，看它们之间的余弦相似度和跨区间迁移效果。

另外要有心理准备，引导方法的可靠性争议不少。Tan 等人（NeurIPS 2024）发现引导效果在不同输入之间方差很大，有时还起反作用；在 AxBench（2025）上，简单的提示词普遍胜过表示层引导。所以基线里除了 [2] 的 CoT，还应该有强提示词，以及 DoLa 这类对比不同层输出、用来挖掘事实知识的解码方法。

### 方向二：把回忆转化为选择

我认为这是三个方向里最有希望、也最容易分析干净的一个。成功率可以拆成两个因子：
- **覆盖率**：正确答案是否出现在候选池里；
- **选择准确率**：出现之后能否被选中。

两者要分开报，因为瓶颈在哪一侧，决定了下一步改进什么。和 [2] 的 CoT 比较时要对齐计算预算（采样次数、生成的 token 数），因为先生成再选择本身也是一种测试时计算。两者还可以叠加使用。

**选择环节**有个附带的好处：模型自己采样出的候选天然就是"难干扰项"，同领域、名字风格合理、格式一致，比随机拼凑的团队有说服力得多。用它测出来的才是更可信的识别能力。

但要避免循环论证：方法本身依赖识别，就不能再用同一种识别测试当"上限"。上限要用独立的方式测，比如 [2] 式的编码测试，或者用真实团队做干扰项的选择题。

选择最好在作者层面做，而不是在名单层面：把所有样本里出现过的名字汇成候选池，逐个打分，再按阈值拼出最终名单。这样能处理部分正确的情况，也和 HR 指标对得上。打分方式可以比较模型的口头判断、P(True)、序列似然，以及 Inside-Out 式的隐藏状态探针。

**真正的天花板在覆盖率。** Inside-Out 发现有些答案几乎永远采样不出来，所以光调高温度很可能不够。扩充候选池的办法有：
- 换不同的提示语境；
- 按作者逐个生成后汇总；
- 给定第一作者再问其余作者；
- 文档里提出的首字母提示。

首字母提示本质上就是心理学里的"线索回忆"，很适合逼出采样不到的答案。但 26 个字母各跑一遍成本高，可以改为只取模型在姓氏第一个 token 上概率最高的几项作为线索。

还要注意，线索本身泄露了信息。给了首字母，猜一个常见姓氏就可能碰巧对上（"姓氏 + 名字首字母"的匹配规则本来就宽松）。所以这类结果要单独报，并估计随机匹配的基线。

**还有一个值得专门验证的假设：作者归属可能在很大程度上是个反向问题。** 在参考文献列表里，作者几乎总写在标题前面。而 [2] 发现反向问题在识别时不难、在生成时却明显更难，这正好能解释上一篇论文里"生成很差、识别很好"的现象。

如果成立，一个自然的打分方式是顺着训练时的方向来：把每组候选作者放进参考文献格式，计算模型接着生成该标题的似然。

不过方向的构成可能本身也随流行度变化。高引论文主要以"作者在前"的形式出现在大量参考文献里；低引论文则主要以"标题在前"的形式出现在自己的首页或数据库条目里。所以最好同时测"由标题问作者"和"由作者问标题"两个方向，看这种不对称怎样随引用数变化。这个实验成本很低，诊断价值很高。

### 方向三：可提取空间是否随流行度变化

倒 U 形假设很合理，但要注意它和已有结果之间有冲突。它假设低流行度事实没被编码，可上一篇论文的选择题结果是：0–2 次引用的论文识别率仍有 79%，差距并不小；[2] 也认为冷门事实大多已被编码。

这两种说法在低引端不可能同时成立。上一轮提到的干扰项问题，正是在这里直接影响这个项目：
- 如果换成难干扰项后，低引端识别率降到接近随机，倒 U 形成立，[1] 的低引端数字就需要修正；
- 如果识别率依然很高，低引端反而是收益空间最大的地方。

两种结果都有价值，前提是测准。

"提取不出来"和"根本没有"的边界还有个方法论难点：任何方法失败，都只能说明"这种方法提不出来"，不能证明知识不存在。要让"真正缺失"站得住，需要几方面证据同时成立：
- 训练语料中出现次数为零；
- 难干扰项选择题上是随机水平；
- 最强的内部打分方法也失败。

第一条只有训练数据公开的模型才做得到，所以我强烈建议把 OLMo 2 纳入研究对象。上一篇论文已经用过它的 Infini-gram 索引，可以直接按标题与作者在训练语料里的真实共现次数给论文分层，用真实频次替代引用数这个代理变量。

还可以把 [2] 的知识状态细分为四类："从未见过""见过但没存下""存下了但提不出""能直接回忆"。前两类只有在数据可查时才能区分。OLMo 公开的中间检查点，还能用来观察一个事实怎样随训练从"缺失"变成"潜在"，再变成"可回忆"。

"恢复率"也要事先定义清楚。比如定义为（方法准确率 − 基线准确率）/（上限 − 基线准确率），上限取独立测得的识别率。但低引端基线接近零，上限也可能很低，这个比值会很不稳定，所以要同时报绝对增益和置信区间。

## 我认为应该先做的几件事

**第一，重建基线差距。** 文档已把这一步列在最前，我非常同意，整个项目都建立在它上面。具体做法：
- 选择题干扰项改用同学科、同年代、真实存在的合作团队，最好连名字的族裔构成也相近；
- 再加"只替换一位作者"的细粒度干扰项；
- 设两组对照：一组用虚构的论文标题配同样的选项，一组去掉标题只给选项。这两组的准确率都应接近随机，否则说明模型在靠启发式作答。

文档列出的混杂因素（论文年代、姓名常见度、干扰项质量）之外，我会再加上作者本人的知名度。模型可能是因为知道某位知名学者做这个方向才答对，而不是记得这篇论文。用同主题的虚构标题测一下模型会不会"回忆"出同一位学者，就能把对论文的记忆和对领域的联想分开，后者对应 [2] 说的"无编码的推断"。

这一步还会给每篇论文打上"已编码但答不出"的标签，这正是引导实验最该检验的那批样本。

**第二，先把目标缩小到第一作者。** 单个实体会让探针、引导、logit lens 分析和覆盖率计算都干净得多，第一作者也是上一篇论文里回忆最好的位置。方法成熟后再扩展到完整名单。

**第三，选好模型。** 引导和探针需要开放权重，但开源模型的基线回忆可能低到无从引导，所以动手前先测一遍候选模型的基线回忆率。比较理想的组合是 OLMo 2（数据公开，用来画边界）加一个更强的开源模型（用来比较方法）。

base 模型和 instruct 模型也值得一起测。[3] 指出，对重度安全对齐的模型，拒答策略可能压掉几十个百分点本来能答对的知识。base 和 instruct 之间的差距本身就是后训练造成的"丢钥匙"，测起来几乎没有额外成本。

**第四，把流行度信号用作路由器，而不是引导方向。** [2] 自己也承认，思考有计算成本，而什么时候该启用目前并不清楚。如果倒 U 形成立，就可以用 P1 位置的流行度探针预测哪些问题落在"救得回来"的区间，只对这些问题启用多次采样或思考，把算力花在最可能有回报的地方。这样能把上一篇论文的发现、方向三的结论和方法部分串成一条完整的故事线。

按成本和诊断价值排序，我会这样推进：
1. 用难干扰项和对照重建基线，同时做正反方向对比；
2. 做先生成再选择，分开报覆盖率和选择准确率；
3. 最后做激活引导，用 CAA 式成对样本并控制流行度。

引导风险最高、最不确定，放在后面，可以先看前面的结果再判断值不值得投入。需要的话，我可以接着把第一步细化成具体的实验方案，包括干扰项怎么构造、对照怎么设、每个区间需要多少样本。

---

## 第 3 轮 · 用户（2026-10-02 09:06 UTC）

steering的操作非常玄学，这个是我同组另外一个人在搞，这方面我也懒得研究，因为本身就非常魔幻。我主要在分析其中第二个思路，这方面对于我来说比较重要。目前主要是需要制定一个研究的具体思想计划或者纲领。

## 第 3 轮 · Claude

（无文字内容）

---

## 第 4 轮 · 用户（2026-10-02 09:06 UTC）

对于multichoice这方面我该怎么开展研究和分析呢？

## 第 4 轮 · Claude

明白，那就专注方向二。这个方向比引导好把控得多：它是黑盒的，API 模型上也能跑，而且每个环节都能单独测量。下面是一份可以直接照着开工的纲领。

## 一、核心思路

认知心理学里有个经典的"生成–识别"两阶段回忆理论（Kintsch, 1970；Anderson & Bower, 1972）。它认为回忆分两步：先从记忆里联想出若干候选，再逐个判断"认不认识"，任何一步失败，回忆就失败。识别测验更容易，是因为候选已经给出，跳过了第一步。

方向二正是这个理论的工程版本：用采样补"生成"，用选择补"识别"。相应地，模型直接回答答错，有两种可能：
- **生成失败**：正确答案根本没被生成出来；
- **选择失败**：正确答案生成过，但没被选为最终答案。

这个方向的价值，就在于把这两种失败分开测量、分别对症下药。

在这个框架下，选择题有两个必须分开的角色：
- **测量工具**：把正确答案放进选项，看模型能否认出来，用来估计"存了多少"，也就是这类方法的上限。这时选项必须独立于模型和方法来构造。
- **提取方法**：选项来自模型自己的采样，正确答案未必在里面，目的是真正提高准确率。这时上限受"覆盖率"约束，即正确答案进没进候选池。

两者混为一谈就会循环论证：拿识别能力当上限，再用依赖识别的方法去逼近它，结果自然好看，却什么也没证明。

这条路并非没人走过。Kadavath 等（2022）让模型用 P(True) 给自己的样本打分；Orgad 等（2025）有证据表明，用隐藏状态训练的验证器有望提升表现；Inside-Out 揭示了反复采样的天花板。但据我所知，还没有工作在跨度这么大的流行度轴上，把失败拆成生成和选择两部分；也很少有人认真处理选择题测量本身是否可靠。

所以论文主线可以是：在作者归属这个长尾任务上，把 LLM 的回忆失败分解为生成失败和选择失败，测量两者如何随流行度变化，并据此设计在同等计算量下优于 CoT 的"先生成再选择"方法。

## 二、研究问题

- **RQ1（测量）**：严格排除启发式线索之后，识别与回忆的差距有多大？它如何随引用数变化？
- **RQ2（分解）**：直接回答的失败里，多少是"没生成出来"，多少是"生成了但没选中"？两者如何随引用数变化？
- **RQ3（方法）**：哪些候选生成和选择策略，能在同等计算下超过直接回答、自洽投票和 CoT？
- **RQ4（机理，可选）**："由标题问作者"是否相当于一个反向问题？差距中有多少来自方向性？

## 三、阶段一：把选择题做成可信的测量工具

这是整个方向的地基，也直接回应上一篇论文里那个 79%。

### 干扰项分层

干扰项按难度分层，每升一层，就多排除一类"不靠记忆也能答对"的线索：

| 层级 | 干扰项怎么构造 | 排除了哪类线索 |
|---|---|---|
| L0 | 同学科随机名字拼成的团队（上一篇论文的做法） | 不排除，作为参照 |
| L1 | 同学科、作者数相同的另一篇论文的真实团队 | 团队内部一致性（同机构、同语言背景） |
| L2 | L1 + 发表年份相近（如 ±2 年） | 年代和姓名格式 |
| L3 | L2 + 标题语义最接近的论文的真实团队，并匹配团队知名度 | 主题和研究群体的关联、"名人效应" |
| L4 | 真实团队只换掉一位作者，换成这几位作者在其他论文里的真实合作者 | 只知道"这群人常合作"，不知道"这篇是谁写的" |
| L5 | 模型自己对这篇论文生成的错误答案 | 最有诱惑性的干扰项，直接对应方法场景 |

两点注意：
- L1–L3 要排除和真实团队有共同作者的候选。
- L5 的答案和 Semantic Scholar 真值格式不同。所有选项都要先统一成同一种格式（比如"名字首字母 + 姓"），否则格式差异本身就是线索。

### 对照组

至少设三组对照：
- **只给选项、不给标题**：Balepur 等（ACL 2024）发现，不少选择题基准只看选项就能明显高于随机。这组的准确率应接近随机。
- **替换标题**：把选项配给同领域、同年代另一篇论文的标题。如果模型仍偏向原来的"正确团队"，说明它是凭团队本身的特征在答。
- **陷阱题**：去掉正确答案，只留干扰项和"以上都不是"，测模型的误认率，也就是候选里根本没有正确答案时它能否拒绝。这在方法场景里是常态。

另外，LLM 对选项位置和字母有明显偏好（Zheng 等，ICLR 2024）。每道题应在四种循环排列下各问一次，多数排列都选对才记为"认出"。这样还能得到更稳的逐题标签。

### 题型与主指标

建议三种题型同时做、相互对照：
- 四个选项一起给的联合选择题；
- 逐个判断"这是不是这篇论文的作者"的是非题；
- 不给选项，直接比较模型对各候选的条件概率（似然打分）。

是非题可以借用信号检测论，把"真的认识"（d′）和"倾向于说是"（判断偏向）分开。

三种题型可以用同一个指标统一起来。Inside-Out 把知识定义为：在"正确–错误"答案对中，正确答案排在前面的比例。这其实就是 AUC，也等于两选一强迫选择的期望准确率。它不受阈值和偏向影响，适用于任何打分方式，建议作为主指标。

### 信度检查

把论文按"能否直接回忆"和"能否认出"排成 2×2 表，"能回忆却认不出"那一格应接近零。心理学里这一格确实存在，Tulving 和 Thomson 在 1973 年正是用它挑战了两阶段理论。但在 LLM 上，这一格偏大更可能说明测量有噪声，比如位置偏好或干扰项出了问题，需要先排查。

### 阶段一的产出

这一阶段产出两样东西：
- 一张"识别准确率 × 引用区间"的图，每个干扰项层级一条线，再叠上对照组。L0 和 L3/L4 在低引端的差距，就是启发式造成的虚高。
- 每篇论文一个"难干扰项下是否认出"的稳健标签，后面阶段要用。

建议先在上一篇论文用过的 GPT-4o 上（如果 API 还能用）复现 L0，和原结果对上，再看换成高层级后降了多少，这样新旧结果能直接接上。全名单和只看第一作者两个版本都值得做，后者线索更少，结论更干净。

## 四、阶段二：拆解回忆失败

对每篇论文，除了贪心解码，再在几个温度下各采样 k 次（k 取到 50 或 100），统计正确的第一作者是否出现在至少一个样本里，记为 coverage@k（覆盖率）。用 Codex 论文的无偏估计量，从 n 个样本估计各个 k 下的值，按引用区间画出曲线。

这里有个容易被忽略的陷阱：k 一大，常见姓氏几乎必然会出现在候选池里。按"姓 + 名字首字母"的匹配规则，如果真实第一作者是 J. Wang，五十个样本里冒出一个 J. Wang，并不能说明模型记得这篇论文。

所以必须做置换基线：把 A 论文的候选池拿去匹配同学科、同年代 B 论文的真实作者，得到"随机覆盖率"。真实覆盖率扣掉它才有意义，k 越大越关键。

结合阶段一的标签，每篇论文可以归入四类：
- **直接回忆**：贪心解码就答对。
- **可以采到**：贪心答错，但正确答案出现在 k 个样本里。这部分靠选择能救回来。
- **认得但采不到**：样本里都没有，难干扰项识别却答对。这就是 Inside-Out 说的那类答案：几乎不会被采样出来，可一旦出现就会排第一。要靠线索等新的生成手段。
- **认不出**：难干扰项下也失败，很可能真的没存。

按引用区间画出这四类的堆叠比例，我认为是整个项目的核心图。它一次回答三个问题：
- 能救回来的部分集中在哪个流行度区间，也就是倒 U 形是否成立；
- 这部分靠选择能救，还是要靠更好的生成；
- "空书架"从哪里开始。

后两类的划分依赖 k，所以这张图要给出几个不同 k 的版本。k 增大时从"采不到"变成"采得到"的部分，属于"概率低但够得着"；k 很大时仍然剩下的，才是"实际上采不到"。

## 五、阶段三：方法——先生成，再选择

### 候选生成：提高覆盖率

可以比较的策略有：
- 调温度和 top-p；
- 换提示语境，比如问答格式、"标题：……作者："这类数据库条目格式、参考文献格式；
- 排除式提示：要求给出和已有答案不同的作者，避免反复采到同一个错误；
- 分解生成：单独生成第一作者，或给定一位作者再问其余作者。

首字母线索要分两种用法：
- **把真实首字母告诉模型**：这泄露了信息，只能作诊断，用来衡量"缺一个线索"能解释多少差距，而且要扣掉只凭首字母猜最常见姓氏的基线。
- **把 26 个字母逐一试过再选**，或只试模型在姓氏首个 token 上概率最高的几个：这没有泄露任何真实信息，才是合法的方法。

所有生成策略都要在相同计算预算下比较，横轴用生成的 token 数或前向次数。

### 选择：在正确答案已在池中时把它挑出来

可以比较的打分方式有：
- **联合选择题**：对去重后的候选出题，并加上"以上都不是"，允许模型拒绝。
- **P(True)**：逐个判断。Kadavath 等发现，先让模型看到自己的多个样本再判断，效果更好。
- **带 PMI 校正的似然**：用候选在该标题下的对数概率，减去它在空白或无关标题下的对数概率，去掉"常见名字本来概率就高"的偏差（参考 Holtzman 等 2021 年关于 surface form competition 的工作）。这同时处理了文档里列的"姓名常见度"混杂。
- **顺着训练方向打分**：把候选作者放进参考文献格式，计算模型生成该标题的对数概率，同样减去没有作者时的基线。理论上 PMI 是对称的，两个方向算出来应该一样；但语言模型是分别学习两个方向的，所以两者之差本身就测出了方向不对称，正好回答 RQ4。
- **自洽投票**：选出现次数最多的候选，作为基线。

最关键的消融是：选择有没有带来出现频率之外的信息。如果模型的选择基本等于多数投票，说明识别环节没贡献新信息，论文的叙事就要调整。

### 评估

- 单独报告"给定覆盖下的选择准确率"，即只在正确答案已在池中的论文上统计。
- 定义选择效率 =（选择后准确率 − 贪心准确率）/（覆盖率 − 贪心准确率），表示能救的部分实际救回了多少。
- 端到端比较的基线包括：贪心直接回答、少样本提示、CoT、思考模型、k 次自洽投票。和 [2] 的 CoT 基线在同等 token 预算下比，画准确率–计算量的帕累托图。
- 再画一张准确率–作答率曲线，看只在有把握时作答，准确率怎样随作答比例变化。从幻觉的角度看，能主动拒答本身就是成果。

扩展到完整名单时，在作者层面做：汇总所有样本里的名字，逐个打分，按阈值拼出名单，用 HR₂、HR₄、HR₅ 评估。上一篇论文发现模型普遍多报名字，阈值这一步正好能纠正。

## 六、分析与统计规范

论文大致需要这些核心图：
- 分干扰项层级的识别曲线；
- 回忆 × 识别的 2×2 表；
- 分区间的 coverage@k 曲线；
- 四类堆叠图；
- 各选择方法在给定覆盖下的准确率；
- 含 CoT 基线的帕累托图和准确率–作答率曲线。

统计方面：
- 数据按引用区间和学科分层，划出约两成作开发集，用来调提示词、温度和阈值；其余作测试集。分析方案在碰测试集之前就定下来，避免"试几次再挑阈值"。
- 不同方法在同一批论文上比较，用配对检验（McNemar 或配对 bootstrap）。每篇论文有多个样本，回归用含论文和学科随机效应的混合效应逻辑回归。
- 协变量至少包括：作者本人的知名度、姓氏常见度、论文年代、标题特异性、作者人数。
- 每个引用区间约七百篇论文，估计 5% 左右的比例时标准误不到一个百分点，够用。最高的两三个区间样本少，必要时合并。

工程方面：
- 以第一作者为目标、每篇采样 50 次，用同一提示并行采样（如 vLLM 的 n=50），提示只需处理一次，总生成量在数百万 token 量级，开源模型跑起来很轻松。
- 需要对数概率的方法（似然、PMI、方向性打分）只能在开源模型上做；联合选择题在 API 模型上也能做。
- base 模型的似然可能更适合打分，instruct 模型更适合做选择题，值得对照。

## 七、预期结果与决策点

- **难干扰项下，低引端识别率降到接近随机**：倒 U 形成立，上一篇论文低引端的数字需要修正，"空书架"的起点被定位，方法集中在中间区间。
- **对照组都接近随机，低引端识别率却依然很高**：深长尾里确有大量潜在知识，瓶颈主要在覆盖率，线索式生成会成为核心贡献。
- **中间区间以"可以采到"为主**：选择方法便宜又有效，在同等计算下超过 CoT 的希望很大。
- **"认得但采不到"占比很大**：纯采样会撞到天花板，而这批论文正好是做引导的同事最该瞄准的样本。
- **选择约等于多数投票**：识别没带来新信息，重心转向生成端和测量本身。

## 八、推进顺序

1. **先导实验（一两周）**：取 500 篇分层子集，在两三个候选模型上测贪心第一作者准确率和 coverage@50，选出不至于全军覆没的模型；同时搭好名字规范化、匹配、采样和打分的流水线。
2. **阶段一**：重点是干扰项分层和三组对照。
3. **阶段二**：产出四类堆叠图。到这里，已经有一篇诊断性论文的骨架。
4. **阶段三**：方法比较。

阶段一、二的逐篇标签，尤其是"可以采到"和"认得但采不到"两类，建议尽早共享给做引导的同事。两条线用同一套样本和指标，最后合成一篇完整的论文也更自然。

如果需要，我可以接着把阶段一的干扰项构造和三组对照写成具体的实验脚本。

---

## 第 5 轮 · 用户（2026-10-02 09:27 UTC）

你能不能帮我把这个实验计划写进一个文档，当然这个时候最好是英文的，是否可以写进一个google文档里面？我可以给你提供一个空白的google文档：[无标题文档 - Google 文档](https://docs.google.com/document/d/1-Y_PfBaadW4ttq5au2CGhIOXrHBMdOPf6tqB395bI9U/edit?tab=t.0)

## 第 5 轮 · Claude

我没法直接写进你的 Google 文档：这里没有能编辑 Google Docs 的连接，我在连接器目录里也没找到。所以我把计划做成了一份英文 Word 文件，Google 文档可以直接打开。

放进你那份空白文档的方法：把文件上传到 Google Drive，右键选"用 Google 文档打开"，然后全选复制、粘贴进你的文档。标题层级、表格和列表都会保留，表格也按 Google 文档的兼容方式设置过。

文档约 10 页，包括摘要和核心论点、背景、两阶段框架、四个研究问题、三个实验阶段、统计规范、结果与决策对照表、时间表、和做激活引导的同事的协作方式，以及 18 篇参考文献。两点需要你确认：
- **时间表**：先导实验之后各阶段的时长是我估的，请按你们的进度调整。
- **参考文献**：部分条目只写了"et al."，正式引用前最好再核对一遍。

*本轮交付的文件：Generate_then_Select_Research_Plan.docx*

---

## 第 6 轮 · 用户（2026-10-02 18:38 UTC）

不用，你这个稿子写的太完善了，背景什么的可以全不要，简单一点交代清楚计划本身就可以，这不是正式文档

## 第 6 轮 · Claude

改成了 3 页的精简版，背景、文献和论证都删了，只留计划本身：核心想法、实验设置，以及从先导到方法的四个步骤，每步写明做什么、测什么、产出什么，最后附几条分析注意事项。导入 Google 文档的方式和之前一样。

*本轮交付的文件：Generate_then_Select_Plan.docx*

---

## 第 7 轮 · 用户（2026-10-05 21:20 UTC）

请帮我上面那个复杂版的dox文件做几张ppt，三五张就行。

## 第 7 轮 · Claude

（无文字内容）

---

## 第 8 轮 · 用户（2026-10-05 21:26 UTC）

附件：屏幕截图 2026-10-05 142520.png

按照这个格式来，做三五张就行，不需要耗多少时间了，请尽快出结果

## 第 8 轮 · Claude

按截图的格式做了 5 张，文件在下面：

1. 核心思路：回忆 = 生成 + 识别，两种失败，选择题的两种用法
2. Step 1：干扰项梯度 L0–L5、三组对照、AUC 主指标
3. Step 2：覆盖率、随机匹配基线、四类画像
4. Step 3：生成策略、选择打分、关键消融和评估
5. 时间线和决策点

版式照搬截图：左上编号、右上 Presenter 框、大标题、页脚 CS 7980、右下页码。我按截图里 Generate-then-select 的负责人，把演讲人填成了 Minyan Xu，主色也用了你那条线的珊瑚红；如果演讲人不对，改一下右上角就行。

第 5 页的排期是照截图时间线上你那四条色块估出来的，对齐了 Update 1、Update 2 和 Draft paper 三个节点。合并进团队的 deck 时，左上角的 01–05 记得按总顺序重新编号。

*本轮交付的文件：Generate_then_Select_Slides.pptx*

---

## 第 9 轮 · 用户（2026-10-05 21:31 UTC）

给我个稿子

## 第 9 轮 · Claude

下面是英文讲稿，按每分钟约 150 词算，全程大约 5 分钟。开头按 Minyan 的身份写的，不是的话改一下名字就行。

**Slide 1 · Core idea (~1 min)**

Hi, I'm Minyan, and I work on the generate-then-select track. My starting point is a simple idea from memory research: recall is really two steps. First you generate candidates, then you recognize the right one. So when a model names the wrong authors for a paper, it can fail in two different ways. Either the correct answer was never generated at all, which is a generation failure. Or it was generated at some point but not chosen as the final answer, which is a selection failure. My goal is to measure these two failures separately across citation levels. Then I want to build a method that generates candidates and lets the model pick, and that beats direct answering and chain-of-thought at the same compute.

One thing I'm careful about is that multiple choice plays two different roles here. In Step 1, it's a measurement. We hide the true answer among distractors and ask how much the model recognizes, which gives us the ceiling. In Steps 2 and 3, it's a method. The options are the model's own samples, so the true answer may not even be there. If we mix these up, the argument becomes circular, so I keep them separate.

**Slide 2 · Step 1 (~1 min)**

Step 1 is building a multiple-choice test we can actually trust. In our earlier paper, GPT-4o got 79% chance-corrected accuracy even on papers with almost no citations. But the distractors were random names from the same field, and a real author team is easy to spot: same lab, same language background. So the model may be using shortcuts instead of memory.

To fix this, I use a distractor ladder. L0 is the old design, and each level removes one more shortcut. L1 uses real teams from other papers. L2 adds the same publication era. L3 adds the most similar title with similar fame. L4 is the true team with just one author swapped. L5 is the model's own wrong answers, which are the hardest.

On the right are three controls that should all be near chance: no title, a swapped title, and catch trials where the truth is removed and "none of the above" is correct. I also rotate the option order to remove position bias. The main metric is pairwise AUC, which works for every question format. The first thing I'll do is reproduce the 79% at L0, then see how much it drops at the harder levels.

**Slide 3 · Step 2 (~1 min)**

Step 2 asks where recall actually fails. For each paper, I take the greedy answer plus up to 50 to 100 samples. Then I check coverage: does the correct first author show up at least once? There's one trap here. With many samples, common surnames like "J. Wang" will match by pure luck. So I compare against a chance baseline, where one paper's samples are matched against another paper's true authors.

Then every paper falls into one of four profiles:
- **Direct recall:** the greedy answer is already right.
- **Reachable:** greedy is wrong, but the answer appears in the samples, so better selection can fix it.
- **Recognized but unreachable:** the model never generates it but does recognize it in Step 1, so we need better generation.
- **Not recognized:** the fact probably isn't stored at all.

The key figure shows how these four profiles change across citation levels. It tells us where the recoverable knowledge sits, what kind of fix it needs, and where the "empty shelves" begin.

**Slide 4 · Step 3 (~1 min)**

Step 3 is the method itself, and it has two halves. On the left is generation: how do we get the right answer into the pool? I'll try:
- different temperatures and prompt formats,
- asking for an author not already listed,
- generating authors one at a time,
- first-letter cues.

One note on cues: telling the model the true first letter leaks information, so that's only a diagnostic. Trying all 26 letters and then selecting is a fair method.

On the right is selection: how do we pick the right candidate? Options include multiple choice with "none of the above", P(True), and a likelihood score that corrects for how common a name is. Another option is scoring in the authors-to-title direction, which also tests whether this is a reversal-curse problem. Majority vote is the baseline. The must-check result is whether selection beats majority vote. If it doesn't, recognition adds nothing. Everything is compared at the same token budget against greedy, few-shot, chain-of-thought, and self-consistency.

**Slide 5 · Plan (~1 min)**

Here's my schedule, which matches the team timeline. The pilot runs for the first three weeks. The harder multiple-choice test is ready by Update 1, the failure split by Update 2, and the selection methods by the draft paper at the end of November.

At the bottom are my decision points:
- If recognition falls to chance for low-citation papers under hard distractors, the inverted-U pattern holds, and I'll focus on the middle citation levels.
- If recognition stays high, there's real hidden knowledge in the long tail, and generation with cues becomes the main contribution.
- If "reachable" dominates, selection should beat chain-of-thought.
- If "recognized but unreachable" is large, I'll pass those papers to Hana's steering track and Zhiqi's J-Lens track.
- If selection is no better than majority vote, I'll shift the focus to generation and measurement.

Thank you.

需要的话，我可以把这份稿子放进 PPT 每页的演讲者备注里。

---

## 第 10 轮 · 用户（2026-10-08 02:00 UTC）

附件：Proposal Presentation (2).pptx

现在请结合我给你的这个PPT，写一个proposal，这个是我们上课的时候展示的结果，现在有这么几点需要注意：
一.首先阅读一下目前给的proposal的要求：This deliverable has two parts.
PART 1:
You are to write and submit a project proposal following the guidelines discussed in class and in the readings shared in last week's class. Your proposal must contain the following parts/sections:

* Title
* Authors
* Introduction (including the research questions)
* Related Work
* Methods  (including the proposed approach, datasets, evaluation metrics)
* Execution timeline
   * Please use a Gantt Chart, and if working in pairs, add who's responsible for what.
* References

The Introduction must clearly state the research questions the project aims to investigate. The "Methods" section is NOT expected to discuss a concrete solution, but rather the approach(es) that will be investigated; those may be (and typically are) related to previous works and improvement of them. Similarly, the evaluation plan should discuss what kind of experiments, and the necessary resources will be used in order to support the project's thesis.
PART 2:
You are to prepare slides in order to present your proposal in class. The presentation should be <= 10 minutes+5 mins questions. Note that this is not a lot of time, thus you need to be concise; more specifically, you need to be able to focus on a high-level overview of your research questions, the methods you propose to answer these questions, and a detailed timeline. Note however that the presentation is to be done during the class, while the proposal itself is due a couple of days later (to allow post-presentation adjustments to it).
DELIVERABLES:
1. Presentation in class. Everyone must speak and will be graded.
2. By Wednesday after the class, submit two files, the proposal and the slides, both in the PDF format. Even though the presentation is due in class before the proposal's deadline, both files can be submitted at the same time. The proposal's document must be typeset using LaTeX using [ACM's Conference templateLinks to an external site.](https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.overleaf.com%2Flatex%2Ftemplates%2Fassociation-for-computing-machinery-acm-sig-proceedings-template%2Fbmvfhcdnxfty&data=05%7C02%7Cyif.hu%40northeastern.edu%7C13056e644b3244db82cc08ddf3a990d9%7Ca8eec281aaa34daeac9b9a398b9215e7%7C0%7C0%7C638934631783228884%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=BU%2FDC8pGoWum%2Flw250%2FDY%2BjGsVQ5gMEYU%2BfCd%2FMtAwM%3D&reserved=0). Each student needs to submit, if you are in the same team, the files are the same.
注意，ppt是不用准备了，因为已经做完了。还有，只用写我的部分就可以，不需要写他们的。
二，目前存在的一个问题是，我自己方向——multi-choice这个方向，到底是独立于其他两个方向的一种研究思路，还是一个其他两种研究方向的预先准备工作？这件事要是没搞明白整件事的逻辑就是完全混乱的。

## 第 10 轮 · Claude

你的方向既不是完全独立的一条线，也不是给另外两条线打下手。问题出在"multiple choice"这个词在项目里同时指了两件事，拆开以后逻辑就清楚了。

- **作为测量，它是全队共享的上游工作。** 这就是识别测试：把真答案混进干扰项里，看模型认不认得出来。它对应团队的 RQ1：给每篇论文打上"知道且说得出 / 知道但说不出 / 不知道"的标签，同时给出每种方法能达到的上限。steering、J-Lens 和你的方法都依赖它。你在这部分的贡献是把测试做难、加上对照（proposal 里叫 Stage A）。
- **作为方法，它是和另外两条线并列的第三种方法。** 这就是先生成再选择，对应团队的 RQ2：选项来自模型自己的采样，正确答案不一定在里面。它不需要标签就能跑，也不需要看模型内部，所以它同时是两种白盒方法在同等算力下必须打败的基线。这部分是独立的研究线。
- **两者之间还有一条反馈。** 你的失败拆分（Stage B）会回到共享标签里，告诉另外两条线哪些失败在输出层面就能修好（采样里出现过），哪些只能靠干预模型内部（认得但采不到）。后者正是 steering 和 J-Lens 真正有价值的地方。

一句话概括：测量在上游、大家共用；方法和其他两条线并列；拆分结果再反馈给全队。proposal 的引言里专门写了一段讲这个，还画了一张关系图（Figure 1）。

团队 PPT 里有两处容易让人混淆，proposal 已经处理了：
- **名称撞车。** 你的幻灯片把识别测试叫"Step 1"，而团队的 Step 1 是 Hana 做的第一版测试。proposal 里我把你的三部分改名为 Stage A/B/C，并写明了它们和团队 RQ 的对应关系。
- **工作可能重复。** Hana 的 Step 1.5 也在收紧筛选，她 PPT 里提到的换入真实合作者，就是你的 L4。建议和她说定分工：她做真实作者概率，你做干扰项梯度和对照。

另外有一个评估上的坑：那 3,341 篇论文本来就是用选择题筛出来的。只在它们上面评估一个靠"选择"的方法，结果会偏好看。所以 proposal 里写了同时报告全部 9,108 篇的结果。

**文件怎么用：** 在 Overleaf 新建一个空白项目，上传 `main.tex` 和 `references.bib` 两个文件，编译时选 pdfLaTeX。Overleaf 自带 ACM 模板的文档类，我用的是 `[sigconf,nonacm]`，会去掉 ACM 的版权框。我这边没有装这个文档类，没法直接用 ACM 模板编译。我换了一个替代文档类把正文、表格、关系图和 Gantt 图都编译过一遍，没有报错，但如果 Overleaf 上报错，把错误信息发我。

**提交前请核对几处：**
- 引言里的数字都来自团队 PPT 的 Step 1：1.41%、69%、36%、3,341 篇，以及 50/10/40 的数据划分。
- Gantt 图的时间照搬了团队时间线。我额外加了一条 11/16–11/30 的"结果并入团队对比"，对应 Update 3，不需要的话删掉那一行即可。
- 作者处只写了 Northeastern University，没有写邮箱，需要的话自己补上。
- 参考文献里 BiPO 和 Qwen3 技术报告这两条，建议再核对一下作者和出处。

*本轮交付的文件：main.tex、references.bib*

---

## 第 11 轮 · 用户（2026-10-08 02:23 UTC）

你这个有点问题，两件事，一我的部分只是整体proposal的一部分，最后要合并到整体里面去，你觉得需要这么齐整的格式吗？
第二个问题更加明显——导入进去以后也显示的是这样的效果：\begin{tikzpicture}[
  box/.style={draw=black!40, rounded corners=2pt, align=center, font=\scriptsize, inner sep=3pt},
  shared/.style={box, fill=black!5, text width=7.6cm, minimum height=0.85cm},
  method/.style={box, fill=black!5, text width=2.2cm, minimum height=1.6cm},
  mine/.style={method, fill=coral!12, draw=coral},
  arr/.style={-{Stealth[length=1.6mm]}, black!55, semithick}]
\node[shared] (S) at (0,0) {\textbf{Shared measurement (team RQ1):} recognition test $\rightarrow$\\per-paper labels, ceiling, metrics, data split\\[1pt]\textcolor{coral}{\textbf{Stage A (this proposal): harder distractors and controls}}};
\node[method] (St) at (-2.7,-2.3) {\textbf{Activation}\\\textbf{steering}\\CAA, ITI, BiPO\\white-box: write};
\node[method] (J)  at (0,-2.3)    {\textbf{J-Lens}\\read-out and\\directions\\white-box: read};
\node[mine]   (G)  at (2.7,-2.3)  {\textbf{Generate-}\\\textbf{then-select}\\sample + select\\black-box\\(this proposal)};
\node[shared] (C) at (0,-4.45) {\textbf{Head-to-head comparison (team RQ2--RQ3):}\\same held-out papers, reported by citation bin};
\draw[arr] (St.north |- S.south) -- (St.north);
\draw[arr] (J.north |- S.south) -- (J.north);
\draw[arr] ([xshift=-0.4cm]G.north |- S.south) -- ([xshift=-0.4cm]G.north);
\draw[arr, dashed, coral] ([xshift=0.4cm]G.north) -- ([xshift=0.4cm]G.north |- S.south)
  node[midway, right, font=\tiny, text=coral, inner sep=1.5pt, align=left] {Stage B\\profiles};
\draw[arr] (St.south) -- (St.south |- C.north);
\draw[arr] (J.south) -- (J.south |- C.north);
\draw[arr] (G.south) -- (G.south |- C.north);
\end{tikzpicture}
\caption{How this track fits the team project. The recognition test (top) is shared measurement that labels papers and sets the ceiling for every method; Stage~A hardens it. The three method families run in parallel, and generate-then-select is the only black-box one. Stage~B's failure profiles flow back into the shared labels (dashed). All methods are compared on the same held-out papers (bottom).}
\label{fig:logic
这个格式似乎有bug

## 第 11 轮 · Claude

两个问题你都说得对，我重做了一版只含你那部分的片段，文件在下面。

**一、格式。** 既然要合并进团队的整体 proposal，就不该自带标题、作者、导言区和全队背景。新版本去掉了这些，只保留你那条线的内容，约一页多，按要放进的章节分成四块：
- **引言块：** 先用三句话讲清选择题的两个角色——作为测量，是全队共享的识别测试；作为方法，是三种方法里唯一的黑盒方法，也是白盒方法要在同等算力下打败的基线。后面接你这条线的三个子问题，并标注各自对应团队的哪个 RQ。
- **相关工作块：** 一段，只写和你方向相关的文献。
- **方法块：** 一个 subsection，按 Stage A、B、C 加评估来组织。数据集和模型只说"沿用团队的设置"，不重复描述。
- **时间线块：** 一小段文字，另附你那几行 Gantt 条目，供做团队甘特图的人直接用。

文件顶部写了每块该放在哪里，复制到团队文档对应位置即可。参考文献单独放在一个 .bib 里，只含这部分引用到的 12 条，合并进团队的 .bib 就行。如果团队那边已经有 Dreaming Up Authors 这类条目但用的键名不同，保留一个，把 `\cite` 改成团队的键名。

**二、代码原样显示。** 最常见的原因是 Overleaf 的可视化编辑器（Visual Editor）不渲染 TikZ，会把这类代码照原样显示。切到 Code Editor，看右侧编译出来的 PDF，图其实是正常的。如果是 PDF 里本身出现了代码，通常是合并后的文档没有加载 tikz 宏包。

不过那张图画的是全队三条线的关系，本来就不该放在你的部分里，新版本已经删掉。现在整个片段只用基础 LaTeX，不依赖任何额外宏包，可视化编辑器里也能正常显示。

唯一的例外是那几行 Gantt 条目。我默认把它们注释掉了，直接粘贴不会报错。等团队用 pgfgantt 做甘特图时，取消注释、放进图里即可。到时候它们在可视化编辑器里同样会显示成代码，这是正常的，以 PDF 为准。

另外，片段我用替代文档类编译过一遍，没有报错。

*本轮交付的文件：generate_then_select_part.tex、generate_then_select_refs.bib*

---

## 第 12 轮 · 用户（2026-10-08 02:30 UTC）

我试验了一下，你第一次那个确实是能够编译成功出pdf的，但为什么引用的况里面都是 [? ? ? ]

## 第 12 轮 · Claude

`[?]` 说明 BibTeX 没有生成参考文献列表，问题不在引用本身。所有引用键在 bib 文件里都有，我这边编译出来是正常的编号 [11, 15, 19]。在 Overleaf 上最常见的原因有三个，按顺序排查：

1. **bib 文件没在项目里，或名字对不上。** `main.tex` 里写的是 `\bibliography{references}`，项目里必须有一个和 `main.tex` 放在同一层、名字正好叫 `references.bib` 的文件。浏览器重复下载时常会把它改名成 `references (1).bib`，这样就找不到了。
2. **Overleaf 用了旧缓存。** 点 Recompile 旁边的小箭头，选 "Recompile from scratch"，清掉缓存重新编译。
3. **BibTeX 报错或没跑。** 打开 "Logs and output files" 看日志。如果出现 "I couldn't open database file references.bib"，就是第 1 条的问题。文档里其他地方的报错也可能让 BibTeX 没运行，需要先把那些错修掉。

为了省掉排查，我做了一个单文件版本 `main_single.tex`，文件在下面。它把参考文献直接写在文件开头，第一次编译时会自动生成 `references.bib`。新建一个空白项目，只上传这一个文件就能编译。如果第一次编译后还显示 `[?]`，再点一次 Recompile，或者用 "Recompile from scratch"。

合并进团队文档时也是同一个道理：我那份 bib 里的条目要粘到团队 `\bibliography{...}` 指向的那个 .bib 文件里，否则你那部分的引用在合并后的 PDF 里同样会变成 `[?]`。

---

## 第 13 轮 · 用户（2026-10-08 06:10 UTC）

他们现在的版本是这个，我希望把我的部分插进去：
文件1,sigconf.tex
\documentclass[sigconf,nonacm]{acmart}

\begin{document}

\title{Enhancing Memory of LLMs}

\author{Hana Feng}
\affiliation{\institution{Northeastern University}\country{USA}}
\email{feng.yi3@northeastern.edu}


\author{Zhiqi Zhang}
\authornotemark[1]
\affiliation{%
  \institution{Northeastern University}
  \country{USA}
}

\email{zhang.zhiqi2@northeastern.edu}

\author{Minyan Xu}
\authornotemark[1]
\affiliation{%
  \institution{Northeastern University}
  \country{USA}
}
\email{xu.minyan@northeastern.edu}

\maketitle

\section{Introduction}
Large language models (LLMs) often produce fluent but false facts, and these errors are most frequent for facts that are rare in their training data~\cite{kandpal2023longtail,mallen2023popqa}. Zhong et al.~\cite{zhong2026dreaming} study this problem through academic authorship attribution: given a paper's title and publication year, the model must name the paper's authors. Using citation count as a proxy for factual prevalence, that is, how often a fact appears in training data, they find hallucination rates above 98\% for rarely cited papers and below 36.4\% for highly cited ones. Such errors are costly in science, where invented authors and references mislead readers.

A wrong answer, however, does not always mean that the knowledge is missing. In a multiple-choice test, GPT-4o often picks the correct author list for papers whose authors it cannot name, which indicates that these failures mainly reflect limited retrieval rather than absent knowledge~\cite{zhong2026dreaming}. Studies of other factual questions reach the same conclusion: LLMs can encode the correct answer internally yet generate a wrong one~\cite{orgad2025knowmore,gekhman2025insideout}, and frontier models encode most of the facts they are tested on, while recall remains the main bottleneck, especially for long-tail facts~\cite{calderon2026emptyshelves}.

This suggests that much of this knowledge could be brought out at answer time, without updating the model's weights. We study three families of methods that work this way. Activation steering adds a direction to the model's hidden states while it generates~\cite{rimsky2024caa,li2023iti,cao2024bipo}. Lens-based methods read out which concepts the intermediate layers represent, which can show whether the correct answer is present and where to intervene~\cite{gurnee2026verbalizable}. Generate-then-select samples several answers and selects among them~\cite{wang2023selfconsistency,kadavath2022know}. These methods were developed for other goals, such as making answers more truthful or improving reasoning accuracy. Whether they can make a model state a specific long-tail fact that it holds but fails to recall, and which of them works best, remains open.

We therefore ask three research questions:
\begin{enumerate}
  \item[\textbf{RQ1.}] When an LLM fails to name a paper's authors, how often does it still hold this knowledge, and how can such retrieval failures be distinguished from missing knowledge for individual papers?
  \item[\textbf{RQ2.}] Can methods that leave the model's weights unchanged make the model generate authors that it holds but fails to recall, on papers that were not used to build the method?
  \item[\textbf{RQ3.}] How do activation steering, lens-based methods, and generate-then-select compare, and how does their benefit vary with factual prevalence?
\end{enumerate}
We hypothesize that the gains are largest for papers of medium prevalence, whose authors the model is likely to hold but does not reliably retrieve.

Answering these questions will yield three outcomes: (1) per-paper labels on the benchmark of Zhong et al.\ that separate retrieval failures from missing knowledge, which any elicitation method can reuse; (2) a head-to-head comparison of the three families of methods on the same held-out papers; and (3) a map of where along the prevalence range each method recovers facts, and where none of them does.

\section{Related Work}

\subsection{Intermediate Representation Readout}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% why we use J-Lens and how prior work on R-Lens, token-position sensitivity, and Concept Lens reveals important limitations of J-Lens as a measurement tool.
Several recent methods attempt to recover semantically meaningful
information from intermediate transformer activations. J-Lens constructs
a layer-specific Jacobian-based readout and shows that intermediate
concepts can become verbalizable before the model produces the final
answer \cite{gurnee2026verbalizable}. This is particularly useful for
our setting because the factual target of interest may be internally
available even when the final generated author list is incorrect.

However, later work shows that the reliability of such readouts depends
strongly on where and how they are measured. R-Lens modifies the backward
attribution rules used to construct the readout and often recovers known
concepts substantially earlier than standard J-Lens. For example,
``Japan'' is recovered at Layer 2 rather than Layer 14 in the sushi
example, and ``basketball'' at Layer 4 rather than Layer 20 in the
Michael Jordan example \cite{blank2026rlens}. This suggests that an
apparently absent early-layer representation may reflect readout error
rather than true absence of the concept.

Token position is another source of variation. Short Horizons and Sparse
Concepts shows that Jacobian energy is highly concentrated in a small
subset of source-target position pairs. Retaining the top 20\% of
high-energy pairs increases average intermediate-concept recall from
18.71 to 22.14, while retaining only the top 1\% reduces it to 16.45
\cite{shorthorizons2026}. These results motivate treating token position
as a tunable part of the readout protocol rather than always using the
last prompt token or uniformly averaging all positions.

Finally, author identities are inherently multi-token concepts. Concept
Lens separates candidate generation from concept scoring and reports a
large gap between normal Rank@10 and oracle-candidate Rank@10
(44.3\% versus 91.9\% on Llama-3.1-8B), indicating that failure to
surface a concept is not equivalent to failure to represent it
\cite{conceptlens2026}. We therefore distinguish token-level author
accessibility from full multi-token author-concept recovery.

\subsection{Steering and Internal Decodability}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% successful behavioral steering and improved J-Lens decodability are not equivalent.
% we need comparing baseline and steered states for the same paper

Activation steering methods such as CAA, ITI, and BiPO modify internal
activations without updating model weights. In our project, these methods
provide the intervention mechanism, while J-Lens provides a separate
measurement layer for analyzing what changes internally.

Importantly, successful steering does not necessarily imply improved
decodability. Steerable but Not Decodable evaluates function-vector
steering using progressively stronger readout mechanisms, including
Logit Lens, Tuned Lens, and nonlinear MLP probes, and shows that some
behaviorally effective interventions remain difficult to decode
\cite{nadaf2026steerable}. We therefore treat steering success and
J-Lens accessibility as separate outcomes rather than assuming that one
must imply the other.

This distinction also affects how we compare internal states before and
after intervention. A recent study using SAE feature steering together
with a Jacobian lens finds that isolated post-steering states are nearly
uninformative for identifying steering provenance, whereas paired
clean-versus-steered comparisons for the same prompt are substantially
more informative \cite{jones2026saesteering}. This motivates using
within-paper changes, rather than absolute post-steering scores, as the
primary unit of mechanistic analysis.


\subsection{Steering Specificity and Control Design}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we need baseline、negative、random、unrelated controls. control experiment can help us distinguish effects specific to the learned steering direction from changes caused by generic activation perturbation.

A further concern is that large behavioral effects do not by themselves
demonstrate that a steering direction captures the intended semantic
property. In a replication of evaluation-awareness steering, a
semantically unrelated control contrast involving ``top shelf'' versus
``bottom shelf'' produced one of the largest behavioral changes observed
under some intervention settings \cite{read2026evaluationsteering}.
The same study also reports strong dependence on intervention magnitude
and non-monotonic effects across steering strengths.

These findings motivate a control structure that separates semantic
specificity from generic activation perturbation. Our test protocol
therefore includes a no-steering baseline, the learned positive
direction, the corresponding negative direction, and a norm-matched
random direction. When computationally feasible, we additionally include
a semantically unrelated direction. Only if the learned direction
outperforms these controls on held-out papers do we interpret it as
evidence for a recall-associated steering effect.

\section{Methods}
%%%%%%%%%%%%%%%%%%%%%%%%%
%Minyan:
% 1. Improve shared recognition measurement
%    harder distractors + shortcut controls

% 2. Decompose baseline failures
%    Direct recall
%    Reachable
%    Recognized-but-unreachable
%    Not recognized

% 3. Build black-box Generate-then-Select
%    sample candidates
%    score/select candidates

% 4. Evaluate black-box recovery
%    coverage@k
%    selection accuracy
%    final recall

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hana ：
% 1. Build steering direction
%    CAA / ITI / BiPO

% 2. Tune intervention
%    layer / alpha / heads

% 3. Run held-out test conditions
%    baseline / +v / -v / random

% 4. Compute behavioral recall outcomes
%    and assign transition labels

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zhiqi：
% 1. Collect paired internal activations
%    reuse the same baseline / steered test runs
%    save hidden states across layers and selected token positions

% 2. Apply J-Lens readout
%    for each layer, map hidden states into vocabulary space
%    extract gold-author score and rank

% 3. Measure layer-wise factual accessibility
%    rank gain / score gain
%    first emergence
%    peak rank
%    Top-K coverage / persistence
%    layer-integrated accessibility

% 4. Handle multi-token author representations
%    aggregate author-name token accessibility
%    use Concept Lens as a secondary robustness analysis

% 5. Compare successful vs failed recall rescue
%    Wrong→Correct vs Wrong→Wrong
%    test whether successful rescue shows larger internal accessibility gains

\subsection{Overall Experimental Design}



\subsection{Dataset and Data Split}


\subsection{Activation Steering}



\subsection{J-Lens Mechanistic Analysis}
% core measurement pipeline. 
% Hidden states are collected from each transformer layer, passed through the corresponding layer-specific J-Lens readout, and converted into gold-author scores and vocabulary ranks.

The J-Lens analysis reuses the same held-out runs used for behavioral
evaluation. For each test paper $i$, intervention condition $c$, layer
$L$, and pre-selected token position $p$, we save the residual-stream
activation

\begin{equation}
    h^{(c)}_{i,L,p}.
\end{equation}

J-Lens provides a distinct readout operator $J_L$ for each transformer
layer. We therefore apply the corresponding layer-specific operator:

\begin{equation}
    \tilde{h}^{(c)}_{i,L,p}
    =
    J_L h^{(c)}_{i,L,p}.
\end{equation}

The transformed representation is mapped into vocabulary space using the
model's output unembedding matrix $W_U$:

\begin{equation}
    z^{(c)}_{i,L,p}
    =
    W_U \tilde{h}^{(c)}_{i,L,p}.
\end{equation}

For each gold-author token $a$, we record its J-Lens score

\begin{equation}
    s^{(c)}_{i,L,p}(a)
\end{equation}

and its rank among all vocabulary tokens

\begin{equation}
    r^{(c)}_{i,L,p}(a).
\end{equation}

Thus, a single paper does not produce one J-Lens number. For each
condition it produces a layer-by-position trajectory. For example, if
the model has $N$ transformer layers and the final protocol retains
$P$ token positions, a paper produces $N \times P$ readout locations
before author-level aggregation.

Our primary mechanistic measurement is paired. For the same paper, layer,
and token position, we compare the steered run against that paper's own
no-steering baseline. We therefore focus on quantities such as

\begin{equation}
    \Delta s_{i,L,p}(a)
    =
    s^{\mathrm{steer}}_{i,L,p}(a)
    -
    s^{\mathrm{base}}_{i,L,p}(a),
\end{equation}

rather than interpreting absolute J-Lens scores from unrelated papers as
directly comparable evidence.



\subsection{Token-Position Protocol}
% Specifies where J-Lens readout is applied within the prompt or generation process
J-Lens measurements can depend substantially on token position. We
therefore treat position selection as a tuning decision rather than
assuming that the final prompt token is always optimal.

On the tuning set, we compare the following candidate locations:

\begin{enumerate}
    \item the final prompt token;
    \item the final token of the paper title;
    \item all tokens in the complete title span;
    \item the publication-year token;
    \item high-Jacobian-energy source-target positions, if computationally
          feasible.
\end{enumerate}

For the title span, candidate aggregation policies include the mean score
across title tokens and the score at the final title token. These rules
are compared only on the tuning set. The final position and aggregation
policy are then frozen before test evaluation.

This design follows the finding from Short Horizons and Sparse Concepts
that J-Lens position-pair energy is sparse and that retaining the top
10--20\% of high-energy pairs can outperform uniform averaging
\cite{shorthorizons2026}.

\paragraph{Interaction with intervention scope.}
The validity of a baseline-versus-steered comparison at a particular
prompt position depends on where the steering intervention is applied.

If steering is applied during prompt prefill, hidden states at title,
year, and final-prompt positions may all differ between the baseline and
steered runs and can therefore be compared directly.

If steering is applied only during autoregressive decoding, previously
cached title and year states are not altered by the intervention. In that
case, the primary paired J-Lens analysis will instead focus on states
actually affected by steering, such as the final prompt state and
generation-time states.



\subsection{Layer-Wise Accessibility Metrics}
% the raw layer-wise scores and ranks are converted into quantitative measures of factual accessibility, including rank gain, score gain, first emergence, peak rank, Top-$K$ coverage, persistence, and JMRR.

For every selected token position, J-Lens provides a trajectory of
gold-author accessibility across model layers.

\paragraph{Layer-specific rank gain.}
For author token $a$,

\begin{equation}
    \mathrm{RankGain}_{i,L,p}(a)
    =
    r^{\mathrm{base}}_{i,L,p}(a)
    -
    r^{\mathrm{steer}}_{i,L,p}(a).
\end{equation}

A positive value indicates that the gold-author token moves upward in the
vocabulary ranking after steering.

\paragraph{Layer-specific score gain.}
We additionally retain the continuous J-Lens score:

\begin{equation}
    \Delta s_{i,L,p}(a)
    =
    s^{\mathrm{steer}}_{i,L,p}(a)
    -
    s^{\mathrm{base}}_{i,L,p}(a).
\end{equation}

Rank and score are reported separately because a large continuous score
change does not necessarily produce an equally large discrete rank
change.


\paragraph{First emergence.}

For a fixed threshold $K$, we define the first layer at which the
gold-author token enters the J-Lens Top-$K$:

\begin{equation}
E_K(a)
=
\begin{cases}
\min \{L : r_L(a) \leq K\}, &
\text{if such a layer exists},\\
L_{\max}+1, &
\text{otherwise}.
\end{cases}
\end{equation}

We then define

\begin{equation}
    \mathrm{EmergenceGain}_K
    =
    E_K^{\mathrm{base}}
    -
    E_K^{\mathrm{steer}}.
\end{equation}

A positive value indicates that the correct author becomes strongly
readable earlier after steering.


\paragraph{Peak rank.}

The strongest vocabulary rank achieved across analyzed layers is

\begin{equation}
    \mathrm{PeakRank}(a)
    =
    \min_{L \in \mathcal{L}} r_L(a).
\end{equation}

This metric asks whether steering increases the maximum layer-specific
accessibility of the correct author.


\paragraph{Top-$K$ layer fraction.}

We measure the fraction of analyzed layers at which the gold-author token
appears in the Top-$K$:

\begin{equation}
    \mathrm{TopKFraction}(a)
    =
    \frac{
        |\{L \in \mathcal{L}: r_L(a) \leq K\}|
    }{
        |\mathcal{L}|
    }.
\end{equation}

This measures how broadly across the network the representation is
readable.


\paragraph{Consecutive persistence.}

Because Top-$K$ layer fraction does not distinguish continuous from
intermittent visibility, we additionally record the longest consecutive
sequence of layers in which the gold-author token remains within the
Top-$K$:

\begin{equation}
    \mathrm{Persistence}_K(a)
    =
    \max
    \left(
    \text{length of consecutive layer runs satisfying }
    r_L(a) \leq K
    \right).
\end{equation}

This distinguishes a concept that briefly appears in isolated layers from
one that remains continuously readable through a larger portion of the
network.


\paragraph{Layer-integrated reciprocal-rank summary.}

To summarize the complete layer trajectory with one continuous value,
we introduce a layer-integrated reciprocal-rank score. This is an
analysis metric defined for our study rather than a standard J-Lens
metric.

For each layer,

\begin{equation}
    RR_L(a)
    =
    \frac{1}{r_L(a)}.
\end{equation}

We define

\begin{equation}
    \mathrm{JMRR}(a)
    =
    \frac{1}{|\mathcal{L}|}
    \sum_{L \in \mathcal{L}}
    RR_L(a).
\end{equation}

The primary paired summary is

\begin{equation}
    \Delta \mathrm{JMRR}(a)
    =
    \mathrm{JMRR}^{\mathrm{steer}}(a)
    -
    \mathrm{JMRR}^{\mathrm{base}}(a).
\end{equation}

A positive value indicates that the gold-author token is, on average,
ranked more highly across the analyzed layers after steering.


\subsection{Multi-Token Author Representation}
% Token-level accessibility is first aggregated within each author and then across authors in the same paper
Author identities usually consist of multiple vocabulary tokens.
Therefore, the rank of a single surname token is not treated as
equivalent to recovery of the complete author identity.

Suppose author $a_i$ is tokenized into $m_i$ retained name tokens
$t_{i1},\ldots,t_{im_i}$. We define

\begin{equation}
    \mathrm{AuthorTokenMRR}(a_i)
    =
    \frac{1}{m_i}
    \sum_{j=1}^{m_i}
    \frac{1}{r(t_{ij})}.
\end{equation}

For a paper containing $n$ gold authors, we define

\begin{equation}
    \mathrm{PaperAuthorMRR}
    =
    \frac{1}{n}
    \sum_{i=1}^{n}
    \mathrm{AuthorTokenMRR}(a_i).
\end{equation}

Averaging first within each author and then across authors prevents names
that tokenize into more subwords from automatically receiving larger
weight.

However, this quantity is interpreted only as
\emph{token-level author-name accessibility}. It is not treated as a
direct measure of a unified multi-token author concept.





\subsection{Mechanistic Comparison by Behavioral Outcome}
% answers whether successful recall rescue is associated with larger internal accessibility changes.
% examining full layer-wise trajectories, statistical stability, and readout robustness.
The main mechanistic question is whether successful and unsuccessful
recall rescue are associated with different changes in the internal
accessibility of the correct author representation.

We divide baseline failures into two groups:

\begin{align}
G_{\mathrm{rescue}}
&=
\left\{
i :
\mathrm{Baseline}_i=\mathrm{Wrong},
\mathrm{Steered}_i=\mathrm{Correct}
\right\},
\\
G_{\mathrm{fail}}
&=
\left\{
i :
\mathrm{Baseline}_i=\mathrm{Wrong},
\mathrm{Steered}_i=\mathrm{Wrong}
\right\}.
\end{align}

We use the accessibility measures defined above to compare rescued and
failed papers. Multi-token authors are evaluated using the aggregation
procedure defined above.

The main comparison tests whether rescued papers show larger accessibility
gains than failed papers. Using $\Delta \mathrm{JMRR}$ as one summary
measure, we test

\begin{equation}
\mathbb{E}
\left[
\Delta \mathrm{JMRR}
\mid
G_{\mathrm{rescue}}
\right]
>
\mathbb{E}
\left[
\Delta \mathrm{JMRR}
\mid
G_{\mathrm{fail}}
\right].
\end{equation}

We also compare the full layer-wise rank and score trajectories of the
two groups to identify where their internal accessibility begins to
diverge across the network.

For the main comparisons, we report mean or median effects together with
bootstrap 95\% confidence intervals. Non-parametric tests or
multiple-comparison corrections are used only when required by the final
analysis.

Beyond the binary rescue/failure grouping, we also examine whether larger
continuous behavioral improvements are associated with larger changes in
J-Lens accessibility using Spearman rank correlation.

As robustness checks, selected cases are re-evaluated with R-Lens for
early-layer accessibility and with Concept-Lens-style scoring for
multi-token author representations. Results are also checked against the
control interventions defined in the steering protocol.




\section{Execution Timeline}

\bibliographystyle{ACM-Reference-Format}
\bibliography{references,references_zhiqizhang}

\end{document}

文件2，reference：
@inproceedings{zhong2026dreaming,
  title     = {Dreaming Up Authors: Factual Prevalence and {LLM} Hallucinations in Academic Authorship Attribution},
  author    = {Zhong, Sijia and Yirga, Tamir and Ma, Yijia and Le, Thai and Hu, Yifan},
  booktitle = {Proceedings of EMNLP 2026},
  year      = {2026}
}

@inproceedings{kandpal2023longtail,
  title     = {Large Language Models Struggle to Learn Long-Tail Knowledge},
  author    = {Kandpal, Nikhil and Deng, Haikang and Roberts, Adam and Wallace, Eric and Raffel, Colin},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year      = {2023}
}

@inproceedings{mallen2023popqa,
  title     = {When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories},
  author    = {Mallen, Alex and Asai, Akari and Zhong, Victor and Das, Rajarshi and Khashabi, Daniel and Hajishirzi, Hannaneh},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2023}
}

@inproceedings{orgad2025knowmore,
  title     = {{LLMs} Know More Than They Show: On the Intrinsic Representation of {LLM} Hallucinations},
  author    = {Orgad, Hadas and Toker, Michael and Gekhman, Zorik and Reichart, Roi and Szpektor, Idan and Kotek, Hadas and Belinkov, Yonatan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

@inproceedings{gekhman2025insideout,
  title     = {Inside-Out: Hidden Factual Knowledge in {LLMs}},
  author    = {Gekhman, Zorik and Ben David, Eyal and Orgad, Hadas and Ofek, Eran and Belinkov, Yonatan and Szpektor, Idan and Herzig, Jonathan and Reichart, Roi},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2025}
}

@article{calderon2026emptyshelves,
  title   = {Empty Shelves or Lost Keys? Recall Is the Bottleneck for Parametric Factuality},
  author  = {Calderon, Nitay and Ben-David, Eyal and Gekhman, Zorik and Ofek, Eran and Yona, Gal},
  journal = {arXiv preprint arXiv:2602.14080},
  year    = {2026}
}

@inproceedings{rimsky2024caa,
  title     = {Steering {Llama} 2 via Contrastive Activation Addition},
  author    = {Rimsky, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024}
}

@inproceedings{li2023iti,
  title     = {Inference-Time Intervention: Eliciting Truthful Answers from a Language Model},
  author    = {Li, Kenneth and Patel, Oam and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{cao2024bipo,
  title     = {Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization},
  author    = {Cao, Yuanpu and Zhang, Tianrong and Cao, Bochuan and Yin, Ziyi and Lin, Lu and Ma, Fenglong and Chen, Jinghui},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}

@inproceedings{wang2023selfconsistency,
  title     = {Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author    = {Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023}
}

@article{kadavath2022know,
  title   = {Language Models (Mostly) Know What They Know},
  author  = {Kadavath, Saurav and Conerly, Tom and Askell, Amanda and Henighan, Tom and Drain, Dawn and Perez, Ethan and others},
  journal = {arXiv preprint arXiv:2207.05221},
  year    = {2022}
}



@article{gurnee2026verbalizable,
  author = {Wes Gurnee and Nicholas Sofroniew and Adam Pearce and Mateusz Piotrowski and Isaac Kauvar and Runjin Chen and Anna Soligo and Paul Bogdan and Euan Ong and Rowan Wang and Ben Thompson and David Abrahams and Subhash Kantamneni and Emmanuel Ameisen and Joshua Batson and Jack Lindsey},
  title = {Verbalizable Representations Form a Global Workspace in Language Models},
  journal = {arXiv preprint arXiv:2607.15495},
  year = {2026},
  eprint = {2607.15495},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2607.15495}
}

@misc{blank2026rlens,
  author = {Camila Blank and Agam Bhatia and Neel Nanda},
  title = {R-Lens: Making J-Lens More Faithful on Early Layers},
  year = {2026},
  howpublished = {AI Alignment Forum},
  note = {Research post, August 5, 2026},
  url = {https://www.alignmentforum.org/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers}
}

@article{shorthorizons2026,
  author = {Shi-Qi Yan and Kai-Xuan Ding and Chao-Hong Tan and Qian Chen and Wen Wang and Xiangang Li and Zhen-Hua Ling},
  title = {Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-Lens},
  journal = {arXiv preprint arXiv:2608.25347},
  year = {2026},
  eprint = {2608.25347},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2608.25347}
}

@article{conceptlens2026,
  author = {Xijie Gong and Zimeng Huang and Tonghan Wang},
  title = {Verbalizing Multi-Token Concepts in LLMs},
  journal = {arXiv preprint arXiv:2608.31084},
  year = {2026},
  eprint = {2608.31084},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2608.31084}
}

@article{nadaf2026steerable,
  author = {Mohammed Suhail B. Nadaf},
  title = {Steerable but Not Decodable: Function Vectors Operate Beyond the Logit Lens},
  journal = {arXiv preprint arXiv:2604.02608},
  year = {2026},
  eprint = {2604.02608},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2604.02608},
  url = {https://arxiv.org/abs/2604.02608}
}

@misc{jones2026saesteering,
  author = {Timothy Jones},
  title = {Can a Jacobian Lens Detect SAE Steering?},
  year = {2026},
  publisher = {Praxagent},
  note = {Research note, version 2026.09.04},
  url = {https://praxagent.ai/blog/posts/2026/07/jacobian-lens-sae-steering/}
}

@misc{read2026evaluationsteering,
  author = {Thomas Read and Bronson Schoen and Santiago Aranguri and Joseph Bloom},
  title = {Reproducing Steering Against Evaluation Awareness in a Large Open-Weight Model},
  year = {2026},
  howpublished = {LessWrong},
  note = {UK AISI Model Transparency Team, April 10, 2026},
  url = {https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1}
}

## 第 13 轮 · Claude

你的部分已经插进团队的 `sigconf.tex`，团队原有内容我只改了最后一行 `\bibliography`（加上你的 bib 文件），其余逐行比对过，没有变动。插入位置如下：

- **Related Work：** 在 "Steering Specificity and Control Design" 后面新增一小节 "Sampling, Selection, and Recognition Tests"。
- **Methods 的 "Recognition Measurement and Failure Profiles"（3.3）：** 对应你注释里的任务 1–2，放在 "Dataset and Data Split" 之后、"Activation Steering" 之前。这样安排是因为它是全队共用的测量，三种方法都要用它产出的标签。这一节开头一段专门讲清选择题的两个角色。
- **Methods 的 "Generate-then-Select"（3.10）：** 对应任务 3–4，放在 Zhiqi 最后一节之后。它和 steering 是并列的方法，放在最后不会打断 steering 到 J-Lens 的连贯叙述。
- **Execution Timeline：** 这一节目前还是空的。你的四行 Gantt 条目先以注释形式放在里面，等有人做团队甘特图时取消注释即可。

**参考文献：** 你的部分用到的新条目单独放在 `references_minyanxu.bib` 里，共 8 条，和 Zhiqi 单独一个文件的做法一致。团队文件里已有的条目（如 `orgad2025knowmore`、`kadavath2022know`）我直接沿用了他们的键名，没有重复。这个文件要上传到 Overleaf 项目里，和 `sigconf.tex` 放在同一层，否则你那部分的引用会显示成 `[?]`。

合并后的文件我用替代文档类编译过，没有报错，所有引用和交叉引用都能正常解析。

团队文件里还有三处问题，我没有改动，建议告诉他们：

1. Zhiqi 和你的名字下都有 `\authornotemark[1]`，但 Hana 那里没有对应的 `\authornote{...}`，编译出来会出现一个没有脚注的上标。如果想表示同等贡献，就在 Hana 的 `\author` 后面补一个 `\authornote{...}`；否则把这两行删掉。
2. 引言说 GPT-4o 的选择题结果"表明失败主要来自检索受限"，而你的 3.3 节指出，隐藏标题后仍有 36% 能选对，说明选择题会泄露线索。两处说法放在一起略有冲突，引言那句话最好改得留有余地一些。
3. "Dataset and Data Split" 还是空的。你的两节都提到了"共享的调参集和测试集"以及 9,108 篇论文，写这一节的人需要把 3,341 篇的筛选方式和 50/10/40 的划分写进去，这些引用才说得通。

*本轮交付的文件：sigconf.tex、references_minyanxu.bib*

---

## 第 14 轮 · 用户（2026-10-08 06:15 UTC）

注意，他们已经写好的地方不要改变，只加入我的部分即可

## 第 14 轮 · Claude

已改好。上一版唯一动了团队原文的地方是最后一行 `\bibliography`，我在里面加了自己的 bib 文件名。现在已经恢复成他们的原样。我和原文件逐行比对过，`sigconf.tex` 和 `references.bib` 原有的每一行都没有改动，只有新增。

**`sigconf.tex`：** 你的内容插在四处，都是新增段落：
- Related Work 末尾新增一小节；
- Methods 里 "Dataset and Data Split" 之后新增 3.3；
- Zhiqi 最后一节之后新增 Generate-then-Select；
- Execution Timeline 标题下加了几行注释掉的 Gantt 条目。

**`references.bib`：** 不再单独建 bib 文件，而是把你用到的 7 条新文献直接追加到他们这个文件的末尾，前面加了一行注释标明"以下由 Minyan 添加"。这样不用改 `\bibliography` 那一行也能引用到。另外我删掉了对 Qwen3 技术报告的引用，模型的出处留给写 "Dataset and Data Split" 的人去引。

如果团队的 `references.bib` 在你发给我之后又被别人改过，不要直接用我这个文件覆盖。只需把末尾注释以下那几条复制过去，粘到他们最新版本的末尾即可。

合并后的文件用替代文档类编译过，没有报错，所有引用都能正常解析。

*本轮交付的文件：sigconf.tex、references.bib*

---

## 第 15 轮 · 用户（2026-10-08 06:21 UTC）

附件：屏幕截图 2026-10-07 232057.png

编译完了之后有一处小小的问题：我该修改哪里？

## 第 15 轮 · Claude

问题出在 Zhiqi 那部分，**Layer-Wise Accessibility Metrics** 下面 **Consecutive persistence** 段落里的那个公式。公式本身不会自动换行，里面那段 `\text{...}` 太长，超出了 ACM 双栏的单栏宽度，就跑到右栏去了。

把这一段：

```latex
\begin{equation}
    \mathrm{Persistence}_K(a)
    =
    \max
    \left(
    \text{length of consecutive layer runs satisfying }
    r_L(a) \leq K
    \right).
\end{equation}
```

整段替换成：

```latex
\begin{equation}
\begin{aligned}
    \mathrm{Persistence}_K(a)
    = \max \bigl(
    &\text{length of consecutive layer}\\
    &\text{runs satisfying } r_L(a) \leq K
    \bigr).
\end{aligned}
\end{equation}
```

文字一个字没变，只是把公式拆成两行，第二行和第一行的文字对齐。括号从 `\left( \right)` 改成了 `\bigl( \bigr)`，因为 `\left`/`\right` 不能跨行。公式编号仍然是 (14)，可能会挪到下一行，这是正常的。

我按 ACM 的单栏宽度测过，拆开后能放下。其余公式我也都查过，没有其他会溢出的。

因为这段是 Zhiqi 写的，改之前最好跟他说一声。

---

## 第 16 轮 · 用户（2026-10-08 06:23 UTC）

约在原文的第多少行

## 第 16 轮 · Claude

要替换的是从 `\begin{equation}` 到 `\end{equation}` 这 9 行，看你用的是哪个文件：

- **我发你的合并版 `sigconf.tex`：** 第 469–477 行。上面第 462 行是 `\paragraph{Consecutive persistence.}`。
- **团队原来的版本（没插入你那部分之前）：** 第 425–433 行。上面第 418 行是同一个 `\paragraph`。

如果你们之后又改过文件，行号会有变化。在 Overleaf 里按 Ctrl+F（Mac 上是 Cmd+F）搜 `Consecutive persistence`，往下几行就是这个公式。

---

## 第 17 轮 · 用户（2026-10-08 06:28 UTC）

等一下，我的这份计划里面，有没有判断模型是否真的知道这个知识的办法

## 第 17 轮 · Claude

有，但目前写得还不够严格，按现在的写法还不能对单篇论文下"模型确实知道"的结论。

**计划里已有的判断办法**

核心是 3.3 节的识别测试：在难干扰项下，模型在 4 种选项顺序里多数都选对真作者，就打上"recognized"标签。两组对照用来排除"不认识论文也能选对"的捷径。此外还有生成方面的证据：正确作者在采样里出现的频率高于随机基线，算作"reachable"。这两类证据合起来，就是那四类画像。

**三个缺口**

1. **没说清用哪一级干扰项来判定。** L1–L3 选对，可能只说明模型知道"这是哪个研究圈子"，不等于知道这篇论文的作者。真正检验"知道这篇论文"的是 L4（只换掉一位作者）。判定标准应该明确定在 L4。
2. **对照只用在了整体层面。** 现在写的是对照组的准确率"应接近随机"。这只能说明测试整体没有泄露线索，不能保证每一篇被判为"知道"的论文都可靠。要对单篇下判断，需要逐篇对照：同一组选项配上它自己的标题能选对，换成别篇的标题就选不对，才算知道。团队 Step 1 对隐藏标题已经是这样逐篇做的。另外，纯靠猜也有约 5% 的论文能在 4 种顺序里猜对 3 次以上，这个误判率应该报告出来。
3. **第一作者和全名单没有对齐。** 失败画像按第一作者算，识别测试却用全名单，两者需要统一口径。

**根本限制**

行为测试只能证明"知道"，不能证明"不知道"。"认不出"只是下限，不代表参数里一定没有。能做的是用两种方式检验这个标签：
- **已知的反例：** 换了标题的题和虚构的论文，模型不可能真的认识，用它们估计标签的误判率。
- **预测效度：** 被判为"知道"的论文，应该比"不知道"的更容易被各种方法救回来。如果两组的救回率一样，说明标签没测到知识。

**可以补进去的一段**

下面这段可以直接加在你 3.3 节 "Harder recognition test" 那段之后，只是新增内容，不改动任何人的原文：

```latex
\paragraph{Per-paper decision.}
We call a paper \emph{known} only if the model picks the true author list at level L4, where it differs from a distractor by a single swapped author, in most of the four option orders, and does not pick the same list when it is paired with another paper's title. For first-author analyses, the same test is run with single-name options. Swapped titles and fabricated titles serve as known negatives from which we estimate the false-positive rate of this label; random guessing alone yields three or more correct choices out of four for about 5\% of papers. Because a behavioral test can show that knowledge is present but cannot prove that it is absent, ``not recognized'' is treated as a lower bound. We validate the label by checking that papers labeled known are rescued more often than unknown ones by every method.
```

如果之后想再加一层内部证据，Zhiqi 的 J-Lens 读出可以作为独立的旁证。不过那属于团队层面的事，这次没有写进你的部分。

---

## 第 18 轮 · 用户（2026-10-08 06:43 UTC）

那我这一部分到底是基于他们的判断来做的，还是我自己从9k论文的数据库那里开始做

## 第 18 轮 · Claude

两者都有，看是哪一块：**测量部分从 9k 篇开始做，方法比较部分用他们的划分。**

**一、测量部分：从全部 9,108 篇开始（3.3 节的难识别测试和失败画像）**

不能只用他们筛出来的 3,341 篇，原因有三个：
- **只能删、不能补。** 你的测试本来就是要检验和改进他们的判断。如果只在已通过旧筛选的论文上做，只能剔除误判的，却永远找不回被旧筛选漏掉的论文，比如因为选项位置偏好而没通过的那些。
- **画像比例会失真。** 四类画像在各引用区间的比例，以及"中等流行度收益最大"这个假设，都要在没筛过的完整样本上算。3,341 篇是筛选后的子集，各区间的分布已经被扭曲了。
- **负对照不在里面。** "认不出"那组负对照，恰恰是没进 3,341 篇的论文。

不过不必从零跑。团队 Step 1 已经在全部 9,108 篇上跑过开放式回答和第一版选择题，这些结果可以直接复用：开放式回答就是你的贪心基线，第一版选择题就是你的 L0。你要新跑的只是 L1–L5、对照组和采样。

**二、方法比较部分：用他们的划分（3.10 节的先生成再选择）**

和 steering 正面比较，必须在同一批测试论文上进行。所以你沿用团队现有的划分：那 3,341 篇按 50/10/40 分成构建、调参、测试三部分，你在 10% 的调参集上调参数，在 40% 的测试集上报告。调参只能用调参集，不能在全部 9k 篇上调，否则会碰到测试集，比较就不公平了。另外，你的方法还会在全部 9,108 篇上再报告一次，这一点 proposal 里已经写了。

**两部分怎么衔接**

你的新标签应该是给现有划分加注释，而不是重新划分。Hana 已经在用这个划分构建 steering 向量，中途换掉测试集会打乱她的工作。正确的做法是测试集保持不变，各方法的结果再按你的标签拆开报告。如果 steering"救回"的论文在难测试下其实是"认不出"的，那多半是蒙对的，这本身就是很有价值的分析。

```
9,108 篇 ──Step 1（团队，已完成）──> 3,341 篇 ──> 50/10/40 划分（固定，三条线共用）
   │                                                    │
   └──你的难测试 + 采样（全部 9,108 篇）──> 逐篇标签和画像 ──┘ 作为注释，用来拆分各方法的结果
```

**建议在 proposal 里写明**

你的 3.3 节开头那段，在最后一句 "...and is evaluated like the others." 后面接上下面两句。这是你自己的段落，不涉及别人的内容：

```latex
The harder test and the failure profiles are computed on all 9,108 papers, reusing the greedy answers and the first-version test results as level L0. Their labels annotate the shared split rather than redefine it, so that all method families are still compared on the same held-out papers.
```

---

## 第 19 轮 · 用户（2026-10-08 06:47 UTC）

那为啥不需要判断模型是否真的知道这个知识 不知道的话岂不是回答无数次也不会回答正确

## 第 19 轮 · Claude

需要判断，而且计划里一直在判断，3.3 节的难识别测试和逐篇判定做的就是这件事。我上次说的是"方法运行时不需要这个标签"，不是"不需要判断"。这两件事要分开看：

- **运行时不能用标签。** 实际使用时，模型面对一个问题，事先并不知道自己知不知道。方法必须对任何问题都能直接跑，所以"知不知道"不能当作方法的输入。
- **评估时必须用标签。** 只有知道哪些论文模型真的知道，才能判断方法救回来的是真知识还是蒙对的，也才知道上限在哪里。

你的直觉是对的，"不知道就怎么问都答不对"正是计划里"认不出"那一组的用途：它是负对照。在这组论文上，任何方法的成功率都应该和随机猜差不多；如果明显更高，说明方法在蒙。

不过这句话要加两个修正，这也是为什么光靠"采样多少次能不能答对"判断不了：

1. **不知道也可能碰巧答对。** 真实第一作者叫 J. Wang 的话，采 100 次很可能撞上一个 J. Wang。所以计划里有随机基线，出现频率要高于碰巧水平才算数。
2. **知道也可能永远采不到。** Inside-Out 发现，有些模型认得的答案几乎从不会被采样出来。所以"采不到"也不能证明"不知道"。

因此要把两种证据交叉起来看，这就是四类画像的来源：

| | 采样中出现（高于随机） | 采样中从不出现 |
|---|---|---|
| 难测试中认得出 | 可以采到：知道，选对了就能救回 | 认得但采不到：知道，但只靠采样救不回 |
| 难测试中认不出 | 少见：可能是测试漏检，也可能是碰巧 | 认不出：多半不知道，作负对照 |

贪心解码直接答对的"直接回忆"，是左上格里的特例。

也不存在"回答无数次"的问题。正式的方法比较本来就是在那 3,341 篇"知道但说不出"论文的测试集上做的。在全部 9k 篇上采样，只是为了画出哪些引用区间有知识可救，每篇最多采 100 次，总共几个 GPU 小时。

这张表还暴露了 proposal 里的一个小漏洞：左下格的论文，按现在的定义既算"可以采到"，又算"认不出"，四类画像有重叠。建议在你 3.3 节四类画像列表的后面补一句，规定归类顺序：

```latex
Profiles are assigned in the order listed, so a paper whose correct author appears among the samples more often than the chance baseline predicts counts as reachable even if recognition fails; we report how often this happens as a consistency check between the two kinds of evidence.
```

---

## 第 20 轮 · 用户（2026-10-15 05:55 UTC）

附件：2. Read Papers_Do Literature Search (1).pdf；1. Capstone Class Intro (1).pdf；3. Write a Proposal.pdf；Enhancing_Memory_of_LLM___First_Draft (2).pdf；unnamed

这是下一次作业的要求：

1. Homework (10%): read chapter 5, and answer these 3 questions (type in the comment box)

1. 

1. Confirm that you read the chapter. Answer: Yes/No?
2. The chapter suggests that the introduction should clearly tell the reader what is new in the paper and what the outcomes are. Why? Why not “kept secret until a surprise ending”, as people do when writing novels?
3. State the role of the related work section (you can quote the chapter if you like)

2. Intro/Related Work sections (due 10/8. 90%): Your task is to submit the first draft of your project report's Introduction and Related Work sections. While based on your project proposal, these sections should be more detailed. Homework (10%): read chapter 5, and answer these 3 questions (type in the comment box)

1. 

1. Confirm that you read the chapter. Answer: Yes/No?
2. The chapter suggests that the introduction should clearly tell the reader what is new in the paper and what the outcomes are. Why? Why not “kept secret until a surprise ending”, as people do when writing novels?
3. State the role of the related work section (you can quote the chapter if you like)

2. Intro/Related Work sections (due 10/8. 90%): Your task is to submit the first draft of your project report's Introduction and Related Work sections. While based on your project proposal, these sections should be more detailed. 

* The Introduction section is the first impression the reader will have of the paper and you want to 1) define the problem and the context, very brief discuss prior work (leave more discussion to the related work section) and the limitations to motivate your work 2) state your research problems/hypothesis 3) typically you will state your contributions at the end of the Introduction section, for now just write “Our contributions are…(TBD)” as a place-holder
* The Related Work draws heavily from the literature review you've done, but it will likely expand to include more related work you have found. It should be focused and concise, discuss what has been done, and convince the reader that other works do not solve your problem, or your work improves the performance further; also include work that you utilize to build your own. We will discuss these in our next class, and you should further polish this section.
* Although there are no strict minimum or maximum number of pages required, in a 2 column format, a safe target is 2-2.5 pages for these two sections
* An example to motivate you: https://arxiv.org/pdf/1810.04805

* The Introduction section is the first impression the reader will have of the paper and you want to 1) define the problem and the context, very brief discuss prior work (leave more discussion to the related work section) and the limitations to motivate your work 2) state your research problems/hypothesis 3) typically you will state your contributions at the end of the Introduction section, for now just write “Our contributions are…(TBD)” as a place-holder
* The Related Work draws heavily from the literature review you've done, but it will likely expand to include more related work you have found. It should be focused and concise, discuss what has been done, and convince the reader that other works do not solve your problem, or your work improves the performance further; also include work that you utilize to build your own. We will discuss these in our next class, and you should further polish this section.
* Although there are no strict minimum or maximum number of pages required, in a 2 column format, a safe target is 2-2.5 pages for these two sections
* An example to motivate you: https://arxiv.org/pdf/1810.04805

这个要求奇葩的地方有两点，第一，我并不知道他这个第五章到底在哪里，其实根本就找不到；你先看看他发布的前三节内容跟这个“第五章“”有没有关系。
第二个地方是，我把第一次计划的东西拿给你看，pdf已经贴好了，原版代码附在后面你自己看。你根据上面那个内容把我们的部分加进他这第一次计划里面（我实在搞不明白这东西和之前那个proposal有什么区别）

## 第 20 轮 · Claude

## 一、"第五章"在哪里

三份课件里都没有"第五章"，这些章节应该来自一本单独发的书：
- 第 2 讲的作业写着 "Read Chapter 3"。
- 第 3 讲要求按"上周分享的阅读材料"写 proposal。
- 第 1 讲里有一个 Syllabus 链接。

从章节编号和题目内容看，最可能是 Justin Zobel 的 *Writing for Computer Science*（第 3 版）。它的第 3 章是 Reading and reviewing，第 5 章是 Writing a paper。第 2 讲问的是"为什么要做文献综述、怎么找相关论文"，正对应第 3 章；这次问的是引言和相关工作，正对应第 5 章。不过这是我推断出来的，最好在 Syllabus 或 Canvas 里核实一下，或者直接问老师。这本书是 Springer 出版的，学校图书馆通常能借到电子版。

和前三讲的关系：第 3 讲里关于引言写"是什么、为什么"，以及相关工作"不要罗列文献，要说清你的工作有什么不同"那部分，和第五章的主题重合。第 1 讲列出的后续课程里有 "How to write a technical report/paper"，作业里也说"下节课讨论"，所以第五章应该是那节课的配套阅读。

三道题里，第 1 题（是否读过）要按实际情况回答。另外两题可以先抓住这几个要点，读完章节后用自己的话写：
- **第 2 题：** 论文不是按顺序读完的小说。审稿人和研究者往往只看摘要和引言，就决定要不要往下读。提前讲清新意和结论，读者才能把后文当作检验这些结论的证据来读；把结果藏到最后，读者很可能根本读不到。
- **第 3 题：** 相关工作要把你的工作放进已有研究里：别人做了什么、还有什么局限、你的工作哪里不同或更好，同时说明你借用了哪些方法、和哪些基线比较。它不是文献清单。

## 二、这次作业和 proposal 有什么区别

- **Proposal 是计划书：** 讲做什么、怎么做、什么时候做，所以有方法、时间表和甘特图。
- **这次作业是最终报告（论文）的头两节：** 只有引言和相关工作，内容要比 proposal 更详细，尤其是相关工作，贡献先写占位符。之后的作业会在后面接上方法、实验和结果。

你们这份 PDF 本身就是这次作业的稿子（只有引言和相关工作），所以只需要把你的部分在里面补充得更详细。

## 三、我在稿子里加了什么

团队写的文字一句没动，我用 diff 逐行比对过，只改了下面两处：

- **引言：** 在研究问题之前插入一段新内容。讲选择题测试会被选项里的线索"破解"：隐藏标题后仍有 36% 能选对，而随机水平约为 6%。所以要换用更难的干扰项并加对照，再把每次回忆失败拆成"没生成出来"和"生成了但没选中"两类。
- **相关工作 2.5 节（你自己那一节）：** 从两段扩展为四段：
  1. 回忆分两步的理论框架，以及"思考"为什么是你的基线；
  2. 重复采样与"选择是独立的瓶颈"，新增两篇文献；
  3. 识别测试的陷阱，包括原论文用的是随机拼凑的干扰项；
  4. 仿照 Hana 2.1 节的写法，用三点说明你的工作和已有研究的区别。

  原来那处显示成 "Section ??" 的引用也修好了。它原本指向方法章节，而这份稿子里没有方法章节。

新增的两篇文献在 `new_references_to_append.bib` 里，要追加到团队 `references.bib` 的末尾，否则会显示成 `[?]`。课上专门提醒过虚假引用的问题，这两篇我都在 arXiv 上逐一核对过标题、作者和编号。

修改后的 `sigconf.tex` 我编译过，没有报错，你那部分的引用全部能解析。

有一点需要注意：这份稿子的引言加相关工作原本就有将近 3 页，超过了作业建议的 2–2.5 页，我的补充大约又加了三分之一页。如果需要压缩，最长的是 Hana 的 2.1 节，但删哪部分要团队一起商量。

*本轮交付的文件：sigconf.tex、new_references_to_append.bib*
