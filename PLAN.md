# 修订后的 LLM 训练学习计划

修订日期：2026-09-10。依据本次需求讨论和 2026-09-09 的官方资料核验整理。

本文件定义做什么、为什么做、先后顺序；具体验收见 [ACCEPTANCE.md](ACCEPTANCE.md)，执行状态见 [TASKS.md](TASKS.md)。

## 1. 目标与范围

目标是独立完成一个可复现的单卡小模型训练项目，并能解释数据、损失、参数更新、显存、评测和失败原因。

项目由三个部分构成：

- 训练基础：随机初始化小语言模型，亲自实现训练循环。
- 核心项目：已有 Qwen 模型上的数学 SFT 与 GRPO/RLVR。
- 扩展学习：BabyLM 深入实验、偏好分类、标量 RM、DPO、更复杂数学任务。

核心研究问题：在固定初始模型和评测协议下，额外 SFT、直接 GRPO、SFT 后 GRPO 分别改变了什么，付出了多少计算成本？

“Mini Open-R1”只表示借鉴其研究方向与组件。本项目使用不同规模、数据预算和训练设置，不默认宣称复现 Open-R1 或 DeepSeek-R1 的结果。

## 2. 已知条件与待测条件

| 项目 | 已知条件或暂定方案 |
|---|---|
| 主要训练设备 | Windows 11 + RTX 4070 SUPER；已核验 12282 MiB 显存，原环境采集时占用 2075 MiB |
| 辅助设备 | 16GB M4 MacBook Pro，用于编码、阅读、数据分析和查看实验 |
| 训练系统 | 已验证 WSL2 + Ubuntu 24.04.3；Python 3.12.3、PyTorch 2.11.0+cu128 |
| 已核验资源 | Windows 可见内存约 31.8 GiB，WSL 可见约 15 GiB；虚拟磁盘所在 C 盘实际空闲约 294 GiB |
| 未知资源 | GPU 可用时段；后续各阶段完整训练容量和吞吐待实测 |
| 学习预算 | 每周投入时间与现有数学/PyTorch 基础尚未登记，不承诺固定周数 |

M0 已确认本项目所在 WSL 环境连接目标 GPU，独立虚拟环境重建检查通过。核验结果和范围见 [ENVIRONMENT.md](ENVIRONMENT.md)；最小检查不代表 SFT/RL 配置的容量证明。

不规划 M4 与 NVIDIA GPU 的异构分布式训练。WSL 使用 Windows 侧 NVIDIA 驱动，不在 WSL 内安装 Linux 显示驱动；是否需要额外 CUDA Toolkit 取决于后续是否编译扩展。[S1]

## 3. 执行原则

1. 先完成最小闭环，再扩展规模和实验矩阵。
2. 每阶段先建立适合该任务的评测，再进行正式训练；不必一开始建设通用平台。
3. 实现正确、评测可信且解释充分的负结果，可以通过学习验收。分数提升是研究目标，不是唯一门槛。
4. 每次消融说明比较预算：固定数据、训练 token、epoch 或计算成本中的哪些量。
5. 保存失败实验，不仅保存最佳模型；修改影响训练语义的配置后建立新实验记录。
6. 框架与数据固定版本或 revision。本文中的参数是探测起点，安装后仍须核对对应版本。
7. 小模型自训练与 Qwen 后训练分开记录，后者继续使用其匹配的 tokenizer。

## 4. 阶段路线

### M0：环境与最小实验记录

研究问题：目标训练机能否稳定执行、记录和恢复一个最小训练实验？

- M0-1：登记硬件、主机内存、磁盘、系统、驱动、Python 与 PyTorch 版本；确认目标 GPU。
- M0-2：建立独立环境，完成 CUDA 张量计算、反向传播、参数更新和保存加载检查。
- M0-3：使用本地 JSON/CSV、TensorBoard 或 W&B 中的一种，保存 run ID、配置、随机种子与运行命令。
- M0-4：建立最小代码版本记录和依赖锁定方式；有 Git 时记录 commit 与未提交改动，尚无 commit 时如实记录代码快照。
- M0-5：制定数据/权重的存储位置与检查点保留策略，避免把大文件混入代码版本管理。

先安装本阶段必需依赖。SFT/RL 阶段再增加相应库，按需要维护不同阶段的环境锁定文件。

产物：环境检查记录、可重复运行的最小检查入口、一次实验记录。通过 A0 后进入 M1。

### M1：从零训练基础

研究问题：语言模型如何通过 next-token prediction 更新参数？

- M1-1：用小文本或 TinyStories 子集建立训练/验证划分。可以先用简单 tokenizer，再在训练语料上练习一次 BPE；评测数据不参与 tokenizer 训练。
- M1-2：实现 1M～10M 左右的小型 decoder-only 模型与训练循环。理解 embedding、attention、causal mask、FFN、残差和 LM head；可以使用 PyTorch 模块与 autograd。
- M1-3：明确输入、目标与 logits 对齐，处理 padding、EOS 和有效 token 的损失归一化。
- M1-4：实现 AdamW、学习率调度、梯度累积、梯度裁剪和精度策略，记录参数/梯度/优化器/激活的显存来源。
- M1-5：先过拟合一个小批次，再进行一次包含验证集和固定生成样本的稳定训练。
- M1-6：完成保存恢复，记录模型、优化器、调度器、必要的 scaler、随机数状态、步数及数据读取位置。
- M1-7：完成一次有明确假设的消融，优先比较学习率或上下文长度；注明固定的数据与计算预算。

自定义损失时自行对齐 next-token 目标；调用内部已做位移的模型接口时，不再重复位移。[S2]

产物：训练循环、训练/验证曲线、恢复对比、一次消融报告。通过 A1 即可进入 M2。

约 20M～40M 的 BabyLM 正式基线、多词表、多架构消融属于 X1，不作为进入 SFT 的前置门禁。

### M2：评测协议与短解答 SFT

研究问题：额外监督训练能否改变任务表现，以及变化来自格式还是答案能力？

- M2-1：先用 `Qwen/Qwen2.5-0.5B-Instruct` 验证流程。根据容量实测决定正式实验继续用它，还是迁移到同系列 1.5B；记录模型与 tokenizer revision。
- M2-2：从 GSM8K 官方 train 划分自己的训练集与开发集；测试集保持隔离。按题目划分，保存样本 ID/哈希与来源。[S3]
- M2-3：固定 prompt/chat template、答案抽取规则、生成长度与解码参数，建立初始模型评测。
- M2-4：构建 prompt/answer 或 messages 数据管道，统计长度、截断、EOS 与有效监督 token。先关闭 packing。
- M2-5：使用短且完整的解答，先做少量样本的流程检查，再做约 1K 题的 LoRA SFT；数据量以去重、划分、长度筛选后的实际样本为准。
- M2-6：检查 assistant/completion-only loss。训练模板必须支持所需 mask，不能只依赖配置开关。[S4]
- M2-7：保存 adapter，并在独立加载后复测；对比逐题输出、格式正确率、答案正确率和截断率。
- M2-8：完成一次 LoRA/QLoRA 对照，比较显存、吞吐和效果。若当前资源无法运行某配置，记录容量边界，使用更小规模进行机制对照。

OpenR1-Math-220k 不作为首个随机抽样 SFT 数据源。其轨迹可能很长，且同题候选并非全部正确；后续使用时逐条检查正确性、完整性和目标 tokenizer 下的长度。[S5]

产物：数据划分清单、mask 检查、初始/SFT 对比、adapter 加载说明、LoRA/QLoRA 对照。通过 A2 后进入 M3。

### M3：Countdown 与最小 GRPO

研究问题：可验证奖励如何通过组内相对优势改变生成策略？

- M3-1：明确任务规则，包括允许运算、数字是否必须恰好使用一次、重复数字的计数、分数/负数是否允许。按同一规则生成可解训练题，并建立独立开发题。
- M3-2：实现受限表达式解析与 verifier，检查语法、数字使用及结果。避免直接执行模型输出；限制表达式大小、深度和计算资源。
- M3-3：分别记录格式、合法性和正确性；先使用简单正确性奖励，按需要加入权重较小且显式记录的辅助奖励。
- M3-4：测量初始成功率、组内奖励分布和截断率。若几乎所有组都全错或全对，先调整题目难度、样本或采样设置，形成有区分度的学习信号。
- M3-5：手写采样、reward、advantage、token logprob、ratio、clipping、可选 KL、mask 和参数更新。
- M3-6：用极小的固定样本检查数值性质，打印一组完整中间量，随后进行短训练。
- M3-7：用固定版本 TRL 对照同一任务。明确 loss 类型、归一化、reward scaling、KL、更新次数和生成设置，并定位核心实现。

TinyZero 用作任务和研究思路参考，其单 GPU 说明不等于 12GB 复现保证；不把复现“推理涌现”设为门槛。[S6]

当前 TRL 默认目标与原始 GRPO 可能不同；手写对照必须显式配置，不能凭 Trainer 名称认定公式相同。[S7]

产物：verifier 测试、手算对照、最小实现、框架对照、训练记录。通过 A3 后进入 M4。

### M4：GSM8K 数学 RLVR 闭环

研究问题：在同一评测协议下，SFT 后增加 GRPO 带来什么变化？

- M4-1：沿用 M2 的正式初始 checkpoint 和数据边界。明确 SFT 与 RL 训练题是否重叠，并保存各自清单。
- M4-2：实现或复用最终答案抽取与数值标准化，区分解析失败、格式失败、答案错误和输出截断。
- M4-3：RL 仅使用训练分区的 prompt 与校验目标；不把标准解答泄漏进模型输入。
- M4-4：先完成完整 rollout/update 容量探测，再在小训练子集进行 GRPO，按实际吞吐确定预算。
- M4-5：完成 E0 初始模型、E1 SFT、E3 SFT→GRPO 三组闭环，调参使用开发集。
- M4-6：检查奖励增长是否来自格式投机、解析漏洞、输出长度变化或真实答案改善，保留失败案例。

允许无收益或退化结果通过学习验收，但必须通过正确性检查并给出有证据的分析。三组闭环尚不能完整回答“是否需要 SFT 初始化”；该问题在 M5 补 E2 后讨论。

产物：三组统一开发评测、逐题输出、奖励诊断与成本报告。通过 A4 后进入 M5。

### M5：完整对照与研究报告

研究问题：SFT 与 GRPO 的作用和交互是否有证据支持？

| 实验 | 起点 | 本项目内新增训练 |
|---|---|---|
| E0 | 固定初始 checkpoint | 无 |
| E1 | 与 E0 相同 | SFT |
| E2 | 与 E0 相同 | GRPO |
| E3 | E1 的 SFT checkpoint | GRPO |

若起点为 Instruct，所有结论描述本项目的“额外训练”。更换模型大小、Base/Instruct 类型或 revision 后，需要重建相关基线，不能混在同一因果对照表中。

- M5-1：补齐 E2，使用与 E3 对齐的 RL 数据、奖励、训练设置；明确无法对齐的条件及原因。
- M5-2：完成至少一项 SFT 消融和一项 RL 消融。优先回答已观察到的问题，不要求无目的地跑完大参数网格。
- M5-3：记录训练 token、生成 token、独立题目数、更新步数、耗时与显存；说明 SFT→GRPO 比单阶段多使用的预算。
- M5-4：最终配置和评测协议确定后，评估预先指定的最终测试集；小幅差异结合逐题变化、置信区间或关键配置重复实验解释。
- M5-5：整理成功和失败实验，给出数据、模型、配置、恢复、评测与结果复现说明。

产物：四组对照表、关键消融、成本分析、错误分析和最终报告。通过 A5 才称为“完成四组对照的核心项目”。

## 5. 可选扩展

| ID | 扩展 | 任务与边界 |
|---|---|---|
| X1 | BabyLM 深入预训练 | 约 20M～40M 随机初始化模型，自训练 tokenizer，官方评测与有假设的消融；若遵循 2026 比赛，检查 10M words、最多 10 epochs 等规则。非比赛设置如实标注。[S8] |
| X2 | 偏好分类 | Chatbot Arena 类 A/B/tie 数据，按 prompt 分区、log-loss、A/B 交换、位置/长度偏差与校准；先查数据获取条件，不依赖当前是否能参赛 |
| X3 | 标量 Reward Model | 对单个 `(prompt, response)` 打分，成对 ranking loss；明确 tie 的处理，分析 shortcut，不能把 A/B 联合三分类器直接当成标量 RM |
| X4 | DPO | 推荐在 SFT 后补一个小实验，使用同 prompt 的 chosen/rejected 更新策略；不需要预先训练显式 RM，不阻塞 RLVR。[S9] |
| X5 | 更复杂数学 | 按长度与正确性筛选 OpenR1，或用 MATH 的训练分区；独立 dev 调参，MATH-500 用作配置锁定后的评测，并检查数据重叠。[S5]、[S10] |

X1 自训练 tokenizer 不替换 Qwen 的 tokenizer。不同 tokenizer 的 token-level perplexity 不能直接横向排名，应使用统一下游任务或适当的按字节归一化指标。[S11]

AIMO/ARC 方案阅读与更大模型实验放在核心项目之后，再根据学习方向和资源选择。此计划不以排行榜名次作为完成标准。

## 6. 资源预算与失败回退

以下是容量探测起点，不是保证能运行的配置文件：

| 阶段 | 起点 | 扩大条件 |
|---|---|---|
| 小模型预训练 | 1M～10M，context 64～256，小 micro batch | 完成训练/恢复，测出吞吐后尝试更大模型或 context |
| SFT | 0.5B，LoRA，micro batch 1，完整短样本，长度约 512～1024 | 完成 update/save/reload 后扩大；QLoRA 单独实测 |
| GRPO | 0.5B，短 prompt/completion，G 从 2 开始，减少同时驻留的生成序列 | 完成 rollout、logprob、backward、update，检查有学习信号后再扩大 |

GRPO 的 micro batch、生成 batch、梯度累积与 G 联动配置。当前 TRL 要求生成 batch 能被 G 整除；具体含义与 eval 约束按锁定版本核对。[S12]

先用少量完整 step 记录生成与更新各自耗时、GPU allocated/reserved 峰值及主机内存，再估算总训练预算。初期可以使用常规生成路径，测出瓶颈后再决定是否引入 vLLM/Unsloth。

OOM 时先定位阶段：模型加载、生成/KV cache、logprob/激活、反向传播或优化器更新。优先减少同时驻留量；若改变长度、G 或模型大小，需要新配置和新 run ID。不得静默截掉正确答案或把 G 降为 1。TRL 当前不支持 GRPO 的自动缩 batch 功能。[S12]

无收益时按证据排查数据/模板、mask、verifier、初始成功率、组内方差、采样、更新幅度及预算；不通过反复查看测试集选方案。

## 7. 最终交付与边界

核心交付包括训练代码、数据清单、固定配置、评测入口、adapter/checkpoint 加载说明、手写 GRPO、四组对照、关键消融与技术报告。

各阶段报告均使用 [实验模板](reports/TEMPLATE.md)。模型权重不必进入 Git，但其获取方式、基础模型与 adapter 的对应关系必须清楚。

最终能力表述限定为已验证的单卡小模型实验。多机分布式训练、大规模数据系统、生产部署和广义多轮 RL 不因完成此清单而自动掌握。

## 8. 与原 21 步计划的对应

| 原步骤 | 修订位置 |
|---|---|
| Step 0～1：工程与记录 | M0，保持最小实现 |
| Step 2～5：训练循环、tokenizer、预训练、消融 | 基础放 M1；完整 BabyLM、多轮消融放 X1 |
| Step 6：统一评测 | 每阶段前移建立；数学协议在 M2 固定 |
| Step 7～10：SFT | M2；深入消融按问题在 M5 扩展 |
| Step 11～12：偏好/RM | X2/X3；另补 X4 DPO |
| Step 13～15：Countdown 与 GRPO | M3 |
| Step 16：GSM8K RLVR | M4 |
| Step 17：MATH-500 | X5，明确训练源与测试集分离 |
| Step 18：四组矩阵 | M5 |
| Step 19：稳定性与效率 | 从 M0 持续记录，M3/M4 验证 RL，M5 汇总 |
| Step 20：报告 | 每阶段形成小报告，M5 汇总 |

## 9. 官方参考资料

以下资料已在计划审查中核验；在线默认配置会变化，执行时以锁定版本为准。

- [S1：NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [S2：Transformers GPT-2 接口与 labels 位移](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [S3：GSM8K 官方数据集](https://huggingface.co/datasets/openai/gsm8k)
- [S4：TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [S5：OpenR1-Math-220k 数据卡](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [S6：TinyZero 原作者仓库](https://github.com/Jiayi-Pan/TinyZero)
- [S7：TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [S8：BabyLM 官方规则](https://babylm.github.io/)，[官方评测](https://github.com/babylm-org/babylm-eval)
- [S9：TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [S10：MATH-500 数据卡](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [S11：Transformers perplexity 说明](https://huggingface.co/docs/transformers/perplexity)
- [S12：TRL GRPO 配置与参数校验源码](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py)

[S1]: https://docs.nvidia.com/cuda/wsl-user-guide/
[S2]: https://huggingface.co/docs/transformers/model_doc/gpt2
[S3]: https://huggingface.co/datasets/openai/gsm8k
[S4]: https://huggingface.co/docs/trl/sft_trainer
[S5]: https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
[S6]: https://github.com/Jiayi-Pan/TinyZero
[S7]: https://huggingface.co/docs/trl/grpo_trainer
[S8]: https://babylm.github.io/
[S9]: https://huggingface.co/docs/trl/dpo_trainer
[S10]: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
[S11]: https://huggingface.co/docs/transformers/perplexity
[S12]: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py
