# LLM Training Lab

用单张 RTX 4070 SUPER 开展从零预训练、SFT 和数学 RLVR 实验，建立能够解释、复现、评测和诊断训练的能力。

## 文档入口

| 文档 | 用途 |
|---|---|
| [PLAN.md](PLAN.md) | 修订后的执行路线、阶段任务、资源约束和参考资料 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 各阶段验收标准，以及需要提交的证据 |
| [TASKS.md](TASKS.md) | 当前进度、下一步和验收记录 |
| [ENVIRONMENT.md](ENVIRONMENT.md) | 已验证的环境、重建命令和存储约定 |
| [STYLE.md](STYLE.md) | 练习骨架、题面、注释、自检与文档的统一风格约定 |
| [实验报告模板](reports/TEMPLATE.md) | 每轮实验的问题、配置、结果与结论 |

所有协作者和 AI 助手（Claude Code、Codex 或其它）在写代码或文档前先读 [STYLE.md](STYLE.md)，它是风格的唯一出处，第 0 节是核心约定速览。

当前 M0 / A0 已通过，证据见 [M0 报告](reports/m0-20260910-025345/REPORT.md)。M1-1 的字符编码与批次构造练习已通过自检，当前进入 [M1-2 入门：batch 张量、embedding 与 logits](scripts/inspect_forward.py)，骨架已准备，待学习者填写和运行。M1 尚未运行模型训练，A1 待验收。训练学习由学习者执行，纯环境校验由助手代办；每完成一项，将证据路径记入 TASKS.md，再更新状态。

## 主线

1. M0：环境与最小实验记录。
2. M1：从随机初始化训练小语言模型，理解训练循环。
3. M2：固定评测协议，完成短解答 SFT。
4. M3：Countdown 奖励验证、手写 GRPO、框架对照。
5. M4：GSM8K 数学 SFT → GRPO 闭环。
6. M5：补齐四组对照、关键消融与可复现报告。

完整 BabyLM、偏好分类、Reward Model、DPO 和更复杂数学任务作为扩展。它们不阻塞数学 RLVR 主线。

预训练练习的小模型和后训练使用的 Qwen 是不同的模型谱系。项目完成标准是具备经过实验验证的单卡小模型训练能力；覆盖范围以实际完成的任务为准。
