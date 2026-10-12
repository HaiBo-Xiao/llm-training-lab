export const chapters = [
  { slug: 'tokens', number: '01', title: '字符、词表与预测目标', english: 'TOKENS & TARGETS', description: '把一句话变成数字，再看清每个位置究竟要预测什么。', tags: ['字符编码', 'next-token'], file: '01_inspect_tokens.py', complete: true },
  { slug: 'batches', number: '02', title: '从一段文本到一个批次', english: 'DATA & BATCHES', description: '划分数据、切出样本，把输入与目标放进对应的位置。', tags: ['数据划分', 'batch'], file: '02_split_dataset.py', complete: true },
  { slug: 'forward', number: '03', title: '从 ID 到候选词分数', english: 'EMBEDDING & LOGITS', description: '查出向量，再用一次矩阵乘法为每个候选字符打分。', tags: ['embedding', 'logits'], file: '03_inspect_forward.py', complete: true },
  { slug: 'attention', number: '04', title: '因果自注意力', english: 'CAUSAL SELF-ATTENTION', description: '理解信息如何流动，以及模型为什么不能读取未来。', tags: ['Q / K / V', 'causal mask'], file: '04_inspect_attention.py', complete: true },
  { slug: 'loss', number: '05', title: '衡量预测的误差', english: 'LOSS & VALID TOKENS', description: '接下来：对齐 logits 与目标，只对有效 token 计算平均损失。', tags: ['交叉熵', 'loss mask'], file: '05_inspect_loss.py', complete: false },
] as const;

export const stages = [
  { id: 'M0', title: '环境与最小实验', status: '已验收', description: 'CUDA 计算、单参数更新、内存状态恢复与环境重建。' },
  { id: 'M1', title: '从零理解语言模型训练', status: '进行中', description: '从字符与张量开始，逐步实现小模型和训练循环。' },
  { id: 'M2', title: '评测协议与 SFT', status: '计划中', description: '在固定 Qwen 起点上，观察额外监督训练的作用。' },
  { id: 'M3', title: 'Countdown 与手写 GRPO', status: '计划中', description: '用可验证奖励，理解采样、相对优势与策略更新。' },
  { id: 'M4', title: '数学 SFT → GRPO', status: '计划中', description: '在统一评测下，检查收益、退化与奖励投机。' },
  { id: 'M5', title: '完整对照与研究报告', status: '计划中', description: '比较四组实验，解释效果、计算成本与失败原因。' },
] as const;
