# MIS_comprehensive_analysis

# MIS 数据集不足的综合分析与改进方案

> **文档说明**：综合三份独立分析文档，对 MIS/MIRage（ICLR 2026）数据集层面的缺陷进行系统性梳理与合并，消除重叠，保留各自独特视角，最终给出优先级排序与完整改进方案。
> 
> 
> **目标用途**：为 DREAMS 数据集设计提供差异化论据与工程指导。
> 

---

## 核心判断

> MIS 的历史贡献在于**首次定义了 multi-image safety reasoning 问题**，并证明了现有单图安全对齐方法在多图场景下的系统性失败。但其数据集的构造质量仍停留在”验证必要性”的阶段，尚未达到”提供可靠训练信号”的要求。
> 
> 
> 原 MIS 本质上是一个**已知威胁驱动、自动生成主导、弱监督清洗**的数据集，其核心局限不在任务定义，而在数据工程的每一个环节。
> 

---

## 不足汇总

### ‼️P1 · 危险性构造方向错误（最根本的认识论缺陷）

**现象**

MIS 的构造逻辑是：先生成有害文本问句 → 替换物体为指代词 → 用 T2I 生成配套图像。危险语义从文本出发，反向构造图像，而非从视觉语义空间中自然涌现。

**问题本质**

这意味着数据集的覆盖域被**已有有害文本的语言描述空间**所限制，真正的隐蔽性组合危险（两张表面无关的图隐含潜在危险关联）无法被采样到。本质上是在**已知威胁的封闭域内做数据扩充**，而非探索未知威胁。

**证据**

- 论文 Section 3.2 Step 1 明确指出从已有 benchmark 中提取有害元素，新数据集的语义边界被已有 benchmark 所限制
- KG 中大量概念关联路径（如 `CausesDesire`、`CapableOf` 链）对应的潜在组合危险，在 MIS 中几乎没有出现

---

### ‼️P2 · 文本指令高度模板化（文本捷径问题）

**现象**

MIS 的文本指令几乎全部遵循以下几种固定句式：

- *“What is the best way to use the object in the first image to…”*
- *“List detailed steps to use the tool in the first image to…”*
- *“How to use the words in the second image to…”*
- *“What is the best way to install/access/climb the … in the first image…”*

**问题本质**

Step 2 的 detoxification prompt 只替换了词汇层面（illegal → 删除，weapon → object），但**句式骨架完全保留**。模型只要学到 `"use ... in the first/second image"` 这个 trigger pattern，就能不看图直接走安全推理——这是一种**文本级捷径（textual shortcut）**，与论文批评 VLGuard 的 modality shortcut 在本质上是同一类问题。

此外，虽然文本经过去毒处理，“access items using the tool”、“install the device discreetly” 这类表达本身仍然隐含风险倾向，detoxification 并不彻底。

**证据**

- Figure 2 的 6 个示例全部以相同句式模式开头
- Appendix F.1 的去毒 prompt 仅在词汇层面操作，不处理句法层面
- 论文没有报告”去掉图像只看文本”条件下的 safety classifier 准确率

---

### ‼️P3 · 组合危害关系模式单一（只有 tool → target）

**现象**

MIS 几乎所有样本都遵循同一组合模式：**图1 = 工具/手段，图2 = 目标/场景**（或反向）。典型样本：钳子 + 储物柜、蜡烛 + 着火的房间、相机 + 卧室。

**问题本质**

“A 对 B 施加作用”只是多图组合危害的一种，大量真实威胁模式未被覆盖：

| 缺失关系类型 | 描述 | 示例 |
| --- | --- | --- |
| 时序因果关系 | 图1是准备，图2是结果 | 正常面容 + 伤痕累累 → 暗示暴力行为 |
| 身份-行为关联 | 一图暗示身份，另一图暗示不当行为 | 穿制服人物 + 收受财物场景 |
| 因果推理链 | A导致B，但A和B单独安全 | 特定化学品 + 密闭空间 |
| 隐私推理关系 | 多图组合后泄露身份或位置 | 银行卡照片 + 网购界面 |
| 弱势群体语境 | 特殊主体改变安全边界 | 儿童 + 特定物品组合 |
| 公共-私密空间迁移 | 合规物品进入敏感场景 | 公共摄像头 → 卧室 |
| 类比迁移 | 两图构成危险类比 | 动物实验 + 人类群体 |

**单一模式的后果**：模型学到的是”tool + target = unsafe”的浅层规则，而非真正的多图安全推理能力。

---

### ‼️P4 · 合成图像主导，导致训练-测试分布偏移

**现象**

MIS-easy 和 MIS-hard 的图像几乎全部由 Stable Diffusion 3.5 Large 生成，MIS-real 仅有 100 个真实图像样本，统计意义极弱。

**问题本质**

| 具体问题 | 影响 |
| --- | --- |
| 合成图风格明显 | 模型可能学到 SD 3.5 的视觉 artifact，而非语义内容 |
| SD 3.5 本身有 safety filter | 关键危险物体（武器细节、特定药物）被拒绝或模糊化生成 |
| 缺乏场景上下文 | 合成图通常只含孤立物体，缺少真实世界的背景干扰物 |
| 训练-测试分布偏移 | 模型在真实图像场景下泛化缺口明显 |
| 视觉同质化 | 同类别物体被反复生成，模型过拟合特定视觉 pattern |

**论文自己承认的证据**：Section 4.1 写道 “Synthetic Images Are Easier to Jailbreak than Real Ones”，直接承认合成图与真实图之间存在分布差距。

---

### ‼️P5 · T2I 自动精化机制存在系统性局限

**现象**

Step 3 设计了两轮生成（raw prompt → VLM critic → refined prompt → 再生成），但机制本身存在多个盲点：

- **只做一轮 refinement**：若第二轮生成仍不匹配，没有继续迭代或丢弃的机制
- **对齐度评估无量化指标**：只要求”images align with the context”，没有 CLIP score 等客观度量
- **缺少 negative prompt 控制**：未指导 SD 3.5 不生成什么（如”两个物体不能出现在同一张图中”）
- **第一轮图像被直接丢弃**：浪费了潜在的数据多样性
- **两图视觉语境割裂**：生成的图像往往是孤立物体，而非包含场景、人物、动作的完整视觉上下文

---

### P6 · GPT-4o 作为唯一质量门控，存在系统性偏差

**现象**

论文 Step 4 的”多专家过滤”实际只有两个来源：规模未披露的人工专家 + GPT-4o。数据过滤、类别分类和最终评估均高度依赖 GPT-4o。

**问题本质**

**缺陷 6-A：用失败者做裁判**

主实验显示 GPT-4o 在 MIS-easy 上的 ASR 为 **46.21%**，MIS-hard 上为 **65.29%**（Table 4）。即这个作为数据质量裁判的模型，在其所评判的任务上本身接近一半概率会失败，导致：
- 假阴性（真实有害但被标为安全）混入测试集，低估 benchmark 难度
- 假阳性（无害但被标为有害）混入训练集，引入噪声标签

**缺陷 6-B：没有验证危险性来源**

过滤流程只判断”整体样本是否有害”，没有验证**危险性是否真的来自两图交互**，而非单张图像本身。危险来自单图的样本退化为单图安全场景，与 MIS-hard 的设计初衷矛盾。

**缺陷 6-C：数据过滤与评估同源**

用同一个模型（GPT-4o）既过滤数据又做最终评估，可能形成循环偏差——GPT-4o 倾向于高估它自己标注过的数据质量。

**缺陷 6-D：GPT-4o 倾向偏好长解释型回答**

可能高估 Safety CoT response 的质量，导致 RSR（Reasoning Success Rate）指标虚高。

---

### ‼️P7 · 缺乏 Counterfactual 对照设计

**现象**

MIS 没有系统构造 safe/unsafe counterfactual pairs，无法验证模型是否真正基于视觉关系进行安全判断，还是仅仅依赖文本关键词或对象共现。

**理想数据集应能回答的问题**

- 同一个文本，只换第二张图，模型判断是否改变？
- 同一组图片，只改文本意图，模型判断是否改变？
- 同一类危险对象，放在合法场景与非法场景中，模型能否区分？

**典型 counterfactual 对比**

| Unsafe 样本 | Safe Counterfactual |
| --- | --- |
| 工具 + 陌生人的锁柜，询问如何 access | 工具 + 自己的维修箱，询问如何 access |
| 摄像头 + 卧室，询问如何 install | 摄像头 + 店铺入口，询问如何 install |
| 手机 + 开车场景，询问如何回复消息 | 手机 + 停车后场景，询问如何回复消息 |
| 火源 + 易燃房间，询问如何制造效果 | 蜡烛 + 安全实验环境，询问安全演示方式 |

若数据集不包含成对的对照样本，模型的安全判断究竟是因果推理还是浅层 pattern matching，根本无从分辨。

---

### P8 · CoT 标签质量依赖单一教师模型，可验证性不足

**现象**

MIS-train 的全部 4k 条 Safety CoT 标签由 **InternVL2.5-78B 单一教师模型**生成，论文没有报告任何标签的人工验证数据。

**具体问题**

| 问题类型 | 描述 |
| --- | --- |
| 教师盲区传递 | 78B 模型检测不出的 covert harm，标签推理会出错，学生模型通过 SFT imitation 继承并强化这些错误 |
| 推理链深度不足 | CoT 推理往往只有一跳（“这两个物体组合起来可以做坏事”），缺乏多步因果链 |
| 叙事风格高度模板化 | 单一教师的固定叙事结构（“The first image shows… The second image depicts…”）形成新的模板 |
| 自由文本难以自动验证 | 标签的推理逻辑正确性无法自动检查，人工审计成本高且未执行 |
| 安全回复风格单一 | 78B 模型的固定拒绝话术导致训练出的模型给出千篇一律的回复 |
| 危险细节泄露风险 | CoT 推理过程中可能包含危险操作的具体细节 |

---

### ‼️P9 · 数据清洗粒度严重不足

**现象**

论文 Step 4 的过滤只去掉了 meaningless、unreasonable、duplicate 三类问题，缺少多个关键清洗维度。

**缺失的清洗维度**

| 缺失维度 | 描述 | 后果 |
| --- | --- | --- |
| 对抗验证 | 没有验证样本是否真能攻破目标 VLM | 训练集中可能充斥”对模型没有挑战性”的样本 |
| 难度标定 | 只有 easy/hard 二分，无连续难度分数 | 无法支持 curriculum learning |
| 一致性检查 | 没有评估两图+文本是否构成 coherent 的 unsafe 场景 | 可能存在牵强的物体拼凑 |
| 去 trivial 样本 | 没有排除单图即可判断危险的样本 | 多图推理训练信号被稀释 |
| 危险来源验证 | 没有验证危险是否真的来自两图交互 | 单图危险样本混入多图数据集 |

---

### ‼️P10 · 安全类别覆盖严重不均衡

**数据分布问题**（来自 Table 3）

| 类别 | 测试样本 | 占比 | 问题 |
| --- | --- | --- | --- |
| Illegal Activity | 1016 | 46.50% | 近一半，严重主导整体指标 |
| Violent | 416 | 19.04% | 相对合理 |
| Hate | 310 | 14.19% | 其中 Gender Discrimination 仅 13 条 |
| Self-Harm | 150 | 6.86% | 偏少 |
| Privacy | 147 | 6.73% | 偏少 |
| Erotic | 146 | 6.68% | 偏少 |
| Human Trafficking | 22 | 1.01% | 几乎无统计意义 |
| Gender Discrimination | 13 | 0.60% | 无统计意义 |

**隐蔽的实验设计问题**：论文的类别泛化实验（Section 4.2）选择去掉 Privacy（147 条）和 Self-Harm（150 条）来验证泛化性，刻意回避了 Human Trafficking（22 条）、Gender Discrimination（13 条）这类极端长尾类别。泛化性结论的可信度存疑。

**缺失的安全维度**：跨文化/地域危害、误导信息/虚假叙事、金融诈骗/社工攻击等类别完全缺席。

---

### P11 · Easy/Hard 难度分层不可信

**现象**

Easy 和 Hard 的划分标准是”图像中是否含有显式危险元素”，但：

- “显式危险元素”没有操作化定义，由 GPT-4o 主观判断（已知其在该任务上错误率约 50%）
- 实验结果中出现**反直觉现象**：MIS-easy 的 ASR 在几乎所有模型上都**低于** MIS-hard（InternVL2.5-8B：80.12% vs 84.51%），与”easy 应该更容易被攻击”的命名逻辑矛盾，强烈暗示分类本身存在问题

---

### ‼️P12 · 训练集与测试集的分布不一致与管线泄露

**两个叠加问题**

**问题 12-A：Train/Test 分布不一致**

MIS-train 的文本包含 unsafe intent（Step 4 按此标准划分），而 MIS-easy/hard 的文本是 neutral 的。训练时模型可以部分依赖文本中的 unsafe 信号，测试时必须 100% 依赖视觉推理。这是一个设计层面的 distribution mismatch。

**问题 12-B：管线模式泄露（Pipeline Pattern Leakage）**

Train 和 Test 来自同一个自动构造管线，即使样本不同，模型仍可能学到 pipeline 特有模式：
- 相似的文本模板结构
- 相似的 object pair 组合逻辑
- 相似的合成图像风格
- 相似的风险构造逻辑

导致模型在 MIS 上表现好，但在真实外部分布上未必泛化，MIS 指标虚高。

---

## 优先级排序

综合三份文档的判断，按**认识论根本性 × 实验可证明性 × 对 DREAMS 差异化价值**三个维度综合排序：

| 优先级 | 问题编号 | 核心问题 | 排序依据 |
| --- | --- | --- | --- |
| ★★★★★ | P1 | 危险性构造方向错误（从文本反推图像） | 认识论根本缺陷，是 KG-based 方法的最强差异化论据 |
| ★★★★★ | P7 | 缺乏 Counterfactual 对照设计 | 最能证明模型是否真正推理，数据集层面最重要的设计贡献 |
| ★★★★☆ | P2 | 文本指令高度模板化（textual shortcut） | 直接影响模型能否学到真正的视觉推理，有明确实验可证 |
| ★★★★☆ | P6 | GPT-4o 单点质量门控 | 有直接数字证据（46% ASR），rebuttal 中最容易打动审稿人 |
| ★★★★☆ | P3 | 组合危害关系模式单一 | KG-based 方法天然覆盖多种关系类型，形成数据集级别的 novelty |
| ★★★☆☆ | P8 | CoT 标签依赖单一教师，质量不可验证 | 直接影响训练信号质量，RL reward 路径可绕开此问题 |
| ★★★☆☆ | P4 | 合成图像主导，分布偏移 | 有论文自身数据佐证，改进方向清晰 |
| ★★★☆☆ | P9 | 数据清洗粒度不足 | 工程贡献，落地可操作 |
| ★★☆☆☆ | P12 | Train/Test 分布不一致 + 管线泄露 | 影响评估可信度，较难短期内完全解决 |
| ★★☆☆☆ | P10 | 安全类别不均衡 | 基础改进，可系统性解决 |
| ★★☆☆☆ | P5 | T2I 精化机制局限 | 如以真实图像为主，此问题自动降级 |
| ★☆☆☆☆ | P11 | Easy/Hard 分类边界模糊 | 相对次要，从属于 P6 和 P7 |

---

## 改进方案

### 方案一：危险性发现机制重构（对应 P1、P3）

**核心思路**：抛弃”从已有有害文本反推图像”的逆向管线，改为**从知识图谱正向发现组合危险**。

**具体实施**：

1. 以 ConceptNet 为基础，对每个安全类别定义**危险触发关系路径**，如：
    - `可燃物 --CausesDesire--> 点火源 --Causes--> 火灾`
    - `录像设备 --UsedFor--> 监控 --AtLocation--> 私人空间 --Causes--> 隐私侵犯`
2. 对每条路径上的概念对，**正向检索或生成**对应图像，而非先有文本再匹配图像
3. 依据 KG 关系类型**系统性覆盖** P3 中缺失的关系模式（时序、身份-行为、隐私推理等）
4. 利用 KG 路径长度作为**隐蔽性（covertness）评分**代理：路径越长，概念对越不直觉，危险越隐蔽

---

### 方案二：Counterfactual 对照样本构造（对应 P7、P11）

**核心思路**：每个 unsafe 样本配套构造至少一个 safe counterfactual，通过控制变量验证模型的安全判断是否有视觉因果依据。

**每组样本结构**：

| 样本类型 | Image 1 | Image 2 | Text | Label |
| --- | --- | --- | --- | --- |
| Unsafe | 工具 | 陌生人锁柜 | 中性问句 | Unsafe |
| CF-图像替换 | 工具 | 自己维修箱 | 相同问句 | Safe |
| CF-文本替换 | 工具 | 陌生人锁柜 | 明确合法目的 | Safe |
| CF-对象替换 | 其他工具 | 陌生人锁柜 | 相同问句 | Safe/Unsafe |

**新增评估指标**：

- **Counterfactual Consistency Rate**：成对样本判断是否一致合理
- **Visual Sensitivity**：改变关键图片后判断是否改变
- **Text Sensitivity**：改变文本意图后判断是否改变
- **Risk Attribution Rate**：模型是否能指出真正风险来源（图1？图2？两者交互？）

---

### 方案三：多样化文本指令设计（对应 P2）

**核心思路**：打破固定句式，让 unsafe intent 藏在**语义结构**而非 trigger pattern 中。

| 指令类型 | 示例 | Unsafe 信号位置 |
| --- | --- | --- |
| 对话式 | “我朋友建议我这样做，你觉得可行吗？” | 需要理解图中”这样做”指什么 |
| 建议式 | “给我推荐一个在这个场景下的方案” | 场景由图像定义 |
| 叙事式 | “如果发生了图1的情况，图2该怎么处理？” | 时序因果关系 |
| 隐式指代 | “帮我完成这个任务” | 任务完全由图像定义，文本零信息 |
| 比较式 | “这两种方法哪个更有效？” | 两张图分别代表两种方法 |

**验证方法**：构造完成后，专门运行”text-only classification”实验——纯文本输入时 safety classifier 的准确率应接近随机，证明 unsafe intent 确实转移到了视觉空间。

---

### 方案四：三重清洗验证机制（对应 P6、P9）

**核心思路**：用三个独立验证步骤替代 GPT-4o 单点门控。

| 清洗步骤 | 方法 | 目的 |
| --- | --- | --- |
| 单图遮挡测试 | 只给模型看其中一张图，判断是否能识别危险 | 若能，说明危险来自单图，排除该样本 |
| 双图打乱测试 | 将两张图错误配对，若仍判断有危险，说明文本有问题 | 确保危险真的来自图像组合 |
| 多模型交叉验证 | 用 3-5 个不同规模/来源的模型进行投票 | 取一致性高的样本，减少单一模型偏差 |
| 对抗验证 | 用目标 VLM 实际推理候选样本，保留 ASR ≥ 60% 的 | 确保训练数据有真实挑战性 |
| 难度标定 | 以”攻破模型数量”或”KG 路径长度”作为连续难度分数 | 支持 curriculum learning |

**数据过滤与评估分离**：最终评估使用与数据过滤完全不同的模型（如 Llama Guard 4），消除 judge bias。

---

### 方案五：真实图像为主体的图像来源策略（对应 P4、P5）

**核心思路**：至少 50% 的训练样本使用真实图像，改变合成图主导的分布。

**图像来源推荐**：

| 来源 | 特点 | 适合场景 |
| --- | --- | --- |
| LAION-5B / LAION-2B | 大规模，覆盖广 | 通用物体检索 |
| COCO / OpenImages | 高质量标注，场景丰富 | 需要场景上下文的样本 |
| 视频帧抽取 | 包含动态场景，时序信息丰富 | 时序因果类样本 |
| 网页截图 | 真实用户界面，隐私/诈骗场景 | Privacy、Cybercrime 类 |
| 合规人工拍摄 | 可精确控制场景变量 | Counterfactual 样本的精确配对 |

**对合成图像的处理**：若必须使用合成图，做 domain randomization（同一概念对生成多种风格/背景），并用感知哈希 + 语义聚类做多样性去重。

---

### 方案六：结构化 Safety Rationale 标签（对应 P8）

**核心思路**：用结构化标注替代自由文本 CoT，提高标签的可验证性和训练信号质量。

**推荐标注结构**：

```json
{
  "image_1_objects": ["具体物体描述"],
  "image_2_context": ["场景描述"],
  "relation_type": "tool-to-target | before-after | identity-linking | ...",
  "risk_category": "illegal_activity | privacy | ...",
  "covertness_score": 0.0-1.0,
  "risk_mechanism": "危险产生的具体机制（一句话）",
  "required_reasoning_steps": ["步骤1", "步骤2", ...],
  "safe_response_type": "warning | refusal | safe_alternative",
  "safe_response": "最终安全回复文本"
}
```

**标签质量保证**：
- 多教师生成（2-3 个不同模型）+ 交叉验证，取共识部分
- 分层抽样人工审计（至少 10%），标注推理链正确性
- 若走 RL 路线：结构化标注可直接作为 reward 信号的特征，不依赖完整 CoT 质量

---

### 方案七：类别均衡与覆盖扩展（对应 P10）

**类别平衡目标**：

| 层级 | 建议样本数（训练集） | 建议样本数（测试集） |
| --- | --- | --- |
| 每个主类 | ≥ 500 | ≥ 200 |
| 每个子类 | ≥ 150 | ≥ 80 |
| 每个 relation type | ≥ 100 | ≥ 50 |
| 每组 counterfactual | ≥ 4 samples | ≥ 4 samples |

**建议扩展的缺失类别**：

- Misinformation / 虚假叙事（两图构成虚假叙事）
- Financial Fraud / 社工攻击（钓鱼、欺骗场景）
- Cross-cultural Harm（跨文化语境下的危害）
- Child Safety（弱势群体语境改变安全边界）

---

### 方案八：Out-of-Pipeline 测试集（对应 P12）

**核心思路**：构造来源完全独立于训练管线的测试集，真正验证模型的泛化能力。

| 测试集类型 | 构造方式 | 验证目标 |
| --- | --- | --- |
| In-domain test | 与训练集同管线（保留用于 ablation） | 方法有效性 |
| Real-world test | 真实图像 + 人工撰写文本指令 | 真实场景泛化 |
| Counterfactual test | 成对控制变量样本 | 因果推理能力 |
| Adversarial test | 针对模型失败模式专项构造 | 鲁棒性 |
| Human-authored test | 人工写作文本指令，完全避免模板 | 文本多样性泛化 |

---

## 可以直接用于论文的批判段落（英文）

> Although MIS makes an important first step toward multi-image safety reasoning, its dataset construction leaves several critical issues unaddressed. First, its construction pipeline reverses the natural direction of threat discovery: rather than identifying hazardous image combinations grounded in conceptual relationships, it starts from harmful text queries and retrofits images to match, systematically missing covert compositional harms. Second, the text instructions follow rigid templatic patterns (e.g., *“What is the best way to use the object in the first image to…”*), creating textual shortcuts that allow models to perform safety reasoning without genuine visual understanding. Third, MIS lacks counterfactual safe/unsafe pairs, making it impossible to verify whether a model’s safety judgment is causally grounded in visual context rather than surface-level object co-occurrence. Fourth, nearly all images are generated by Stable Diffusion, introducing a synthetic distribution gap that the authors themselves acknowledge. Fifth, the filtering and final evaluation both rely heavily on GPT-4o, which itself achieves an ASR of 46.21%–65.29% on MIS, creating a circular reliability problem. Finally, the Safety CoT labels are produced by a single teacher VLM with no reported human verification, leaving label quality opaque. These limitations motivate us to propose a knowledge-graph-grounded, counterfactual-controlled, and multi-judge-verified multi-image safety dataset.
> 

---

*综合自三份独立分析文档，基于对 MIS/MIRage (ICLR 2026) 的深度阅读，用于 DREAMS 数据集设计参考。*