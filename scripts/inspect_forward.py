"""M1 第三个练习：batch 张量、embedding 与 logits。

你要完成四个函数，请按 TODO 1～4 的顺序填写。
给定数据、自检、打印和边界说明已经写好，不需要修改。

运行方式（在项目根目录）：
    uv run --locked python scripts/inspect_forward.py

当前文件是练习骨架。函数未填写时返回 None，自检会在对应 TODO 给出中文提示。
本轮使用已有的 PyTorch，在 CPU 上运行，不需要下载数据或模型。

上一轮把文本切成输入和目标两批列表。本轮从给定的小批次出发，完成一次前向计算：
    ID 列表 → ID 张量 x → embedding 向量 h → 候选词分数 logits → 预测 ID pred。
    形状：      [B, T]       [B, T, D]          [B, T, V]        [B, T]
forward（前向计算）指根据输入和当前权重得到输出。
本轮的产物是每个输入位置的候选词分数和预测 ID，之后才能与目标 y 对照计算损失。
本轮使用固定权重观察计算过程，不计算 loss、不反向传播、不更新参数。
这是进入 M1-2 的形状与前向练习；完整 decoder-only 模型仍需后续逐步构建。
"""

import torch


# 以下数据与配置已经写好，不需要修改。
# 本文件独立运行，缩小词表和序列，让所有输入输出都能完整展示。
# 词表下标就是 ID：0 对应“。”，1 对应“小”，2 对应“猫”，3 对应“鱼”。
vocab = ["。", "小", "猫", "鱼"]
batch_size = 2
block_size = 3
embedding_dim = 2
vocab_size = len(vocab)

# B = batch_size 是行数；T = block_size 是每行的预测位置数。
# D = embedding_dim 是每个 token 的向量长度；V = vocab_size 是候选词数量。
# 本轮 B = 2、T = 3、D = 2、V = 4。这四个数描述不同的维度。
# 沿用上一轮起点前进 T 的规则，原始 ID 数组为 [1, 2, 3, 0, 1, 2, 0]：
# 第 0 条取 [1, 2, 3, 0] → x = [1, 2, 3]，y = [2, 3, 0]。
# 第 1 条取 [0, 1, 2, 0] → x = [0, 1, 2]，y = [1, 2, 0]。
# y 已经对齐到下一个 ID，后续转张量和计算分数都保持这些位置，不能再次位移。
xb = [
    [1, 2, 3],
    [0, 1, 2],
]
yb = [
    [2, 3, 0],
    [1, 2, 0],
]

# tensor（张量）是带有 shape（形状）和 dtype（元素类型）的多维数组。
# torch.tensor 用给定数据创建张量；dtype 指定元素类型。
# ID 用 torch.long，即 64 位整数，作为查表下标；参与数值计算的权重用 torch.float32。
# embedding（嵌入）为每个离散 ID 分配一个向量，供模型后续计算使用。
# E 有 V 行 D 列：第 i 行是 ID i 的向量。ID 本身只是编号，向量才承载数值特征。
# 为方便手算，本轮固定下面这些值；训练时 embedding 通常是需要学习的参数。
E = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)

# LM head 中的 LM 是 language model（语言模型）的缩写。
# 这里的输出层把一个 D 维向量转换成 V 个候选词分数：每个候选词对应 W 的一列。
# 本题将 W 按 D 行 V 列存储，并省略偏置，保持手算规模。
# logits 是这些尚未归一化的分数，数值越大表示当前计算更倾向于那个候选词。
# 分数可以为负，也不要求总和为 1；概率和损失将在后续练习中处理。
W = torch.tensor(
    [
        [0.0, 1.0, 2.0, -1.0],
        [0.0, -1.0, 1.0, 2.0],
    ],
    dtype=torch.float32,
)


def to_tensor(rows: list[list[int]]) -> torch.Tensor:
    """TODO 1：把一批 ID 列表转换成整数张量。

    后面的查表和矩阵运算需要张量。这里保留原数组的行、列和每个 ID，仅转换存储形式。

    参数：
        rows：等长整数列表组成的一批数据，可以是输入 xb，也可以是目标 yb。
    返回：
        CPU 上 dtype 为 torch.long 的二维张量，形状为 [行数, 每行长度]。
        第 b 行第 t 列仍是原数据第 b 条样本第 t 个位置的 ID。
    示例 1：
        输入：rows = [[1, 2, 3], [0, 1, 2]]
        输出：
            张量内容 = [
                [1, 2, 3],
                [0, 1, 2],
            ]
            shape = [2, 3]，dtype = torch.long
        解释：两条输入样本各有三个位置，转换前后的六个 ID 一一对应，没有切分或位移。
    示例 2：
        输入：rows = [[3], [1]]
        输出：
            张量内容 = [
                [3],
                [1],
            ]
            shape = [2, 1]，dtype = torch.long
        解释：每行只有一个位置时仍保留行、列两维，不能把结果压成 [3, 1]。
    约束：
        - 1 <= 行数 <= 8，1 <= 每行长度 <= 8；输入非空且各行等长。
        - 元素都是合法 ID，本轮不处理空输入、行长不齐或越界 ID。
        - 使用传入的 rows，不写死 xb，也不改变 ID 顺序或数值。
    思路：
        1. 找出创建张量所需的数据和整数类型。
        2. 用原始嵌套列表创建张量，保留它的两层结构并返回。
    提示：
        - torch.tensor 可以从嵌套列表创建张量；dtype 参数控制元素类型。
    """
    pass


# 以下 TODO 1 自检已经写好，不需要修改。
# isinstance 检查对象类型；shape 返回各维长度；torch.equal 比较整数张量是否完全相同。
x = to_tensor(xb)
assert isinstance(x, torch.Tensor), "请完成 TODO 1：返回一个张量"
assert x.dtype == torch.long, "请完成 TODO 1：ID 张量使用 torch.long"
assert x.device.type == "cpu", "请完成 TODO 1：本轮在 CPU 上运行"
assert tuple(x.shape) == (batch_size, block_size), "请完成 TODO 1：保留输入的行数和列数"
assert torch.equal(x, torch.tensor(xb, dtype=torch.long)), "请完成 TODO 1：保留每个输入 ID 的位置和值"
y = to_tensor(yb)
assert isinstance(y, torch.Tensor), "请完成 TODO 1：同一函数也要能转换目标列表"
assert y.dtype == torch.long and y.device.type == "cpu", "请完成 TODO 1：目标也使用 CPU 整数张量"
assert torch.equal(y, torch.tensor(yb, dtype=torch.long)), "请完成 TODO 1：目标已经位移过，不要再次位移"
column = to_tensor([[3], [1]])
assert isinstance(column, torch.Tensor), "请完成 TODO 1：也要能转换每行只有一项的列表"
assert torch.equal(column, torch.tensor([[3], [1]], dtype=torch.long)), "请完成 TODO 1：单列输入仍保留两维"


def lookup_embeddings(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """TODO 2：为批次里的每个 ID 查出它的 embedding 向量。

    每个 ID 都选择词表中的一行向量。原来的样本维 B、位置维 T 保留，每个位置增加 D 个分量。
    查出的向量将交给下一步的输出层，计算各候选词的分数。

    参数：
        ids：形状为 [B, T] 的整数张量。
        table：形状为 [V, D] 的浮点张量，第 i 行存放 ID i 对应的向量。
    返回：
        形状为 [B, T, D] 的浮点张量。
        第 b 条样本第 t 个位置存放的是 table 中对应 ID 的整行向量，顺序与 ids 一致。
    示例 1：
        输入：ids = [[1, 2, 3], [0, 1, 2]]
              table = [
                  [0.0, 0.0],
                  [1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0],
              ]
              以上数组表示张量内容，B = 2，T = 3，V = 4，D = 2。
        输出：
            h = [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            ]
            shape = [2, 3, 2]
        解释：第 0 条输入 [1, 2, 3] → 依次选 table 的第 1、2、3 行
            → [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]。
            第 1 条输入 [0, 1, 2] → 依次选第 0、1、2 行，得到输出的第二组向量。
            一共仍有 2 * 3 = 6 个 token 位置，每个位置变成长度为 2 的向量，共 12 个数。
            ID 1 两次出现时都查到 [1.0, 0.0]，本步骤还没有加入位置或上下文信息。
    约束：
        - 1 <= B, T, D <= 8，2 <= V <= 16；所有 ID 满足 0 <= ID < V。
        - ids 为 torch.long，table 为 torch.float32，二者都在 CPU 上。
        - 使用传入的 table；不能改动它，也不能重新编号、排序或对 ID 做乘法。
        - 返回张量，不把向量转回 Python 列表。
    思路：
        1. 把每个 ID 理解为 table 的行号。
        2. 按这些行号取出整行向量，保持原来的批次和位置排列。
    提示：
        - PyTorch 支持用整数张量作为下标，一次选择多行。
    """
    pass


# 以下 TODO 2 自检已经写好，不需要修改。
# torch.allclose 比较浮点结果，允许微小的数值误差。
expected_h = torch.tensor(
    [
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    ],
    dtype=torch.float32,
)
h = lookup_embeddings(x, E)
assert isinstance(h, torch.Tensor), "请完成 TODO 2：返回 embedding 张量"
assert tuple(h.shape) == (batch_size, block_size, embedding_dim), "请完成 TODO 2：保留 B、T，并增加向量维 D"
assert h.dtype == E.dtype and h.device == E.device, "请完成 TODO 2：保留词表向量的类型和设备"
assert torch.allclose(h, expected_h), "请完成 TODO 2：每个 ID 应查出对应的整行向量"
other_h = lookup_embeddings(
    torch.tensor([[2, 0]], dtype=torch.long),
    torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32),
)
assert isinstance(other_h, torch.Tensor), "请完成 TODO 2：也要能使用另一份词表"
assert tuple(other_h.shape) == (1, 2, 1), "请完成 TODO 2：维度由传入数据决定，不固定为 2、3、2"
assert torch.allclose(other_h, torch.tensor([[[6.0], [2.0]]])), "请完成 TODO 2：使用传入的 table 查表"


def project_logits(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """TODO 3：把每个位置的向量转换成 V 个候选词分数。

    一个向量有 D 个分量，输出需要给词表中的每个候选词各打一个分。
    对候选词 v，将向量各分量与 weight 第 v 列的对应权重相乘，再将 D 个乘积相加。
    数学上，s_v = sum(d = 0 到 D - 1, h_d * W_{d,v})；h_d 是向量第 d 个分量。
    对每个样本、每个位置都执行这一计算，就得到之后用来预测的 logits。

    参数：
        hidden：形状为 [B, T, D] 的浮点张量，每个位置存放一个 D 维向量。
        weight：形状为 [D, V] 的浮点张量，每一列对应一个候选词。
    返回：
        形状为 [B, T, V] 的浮点张量；第 b 行第 t 个位置包含 V 个候选词分数。
        最后一维的下标就是候选词 ID，前两维保持样本和位置的原顺序。
    示例 1：
        输入：hidden = [
                  [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                  [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
              ]
              weight = [
                  [0.0, 1.0, 2.0, -1.0],
                  [0.0, -1.0, 1.0, 2.0],
              ]
              以上数组表示张量内容，B = 2，T = 3，D = 2，V = 4。
        输出：
            logits = [
                [[0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 1.0, 2.0], [0.0, 0.0, 3.0, 1.0]],
                [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 1.0, 2.0]],
            ]
            shape = [2, 3, 4]
        解释：以第 0 条样本第 2 个位置的 [1.0, 1.0] 为例，四个候选词的分数是：
            ID 0：1 * 0 + 1 * 0 = 0。
            ID 1：1 * 1 + 1 * (-1) = 0。
            ID 2：1 * 2 + 1 * 1 = 3。
            ID 3：1 * (-1) + 1 * 2 = 1。
            因此 [1.0, 1.0] → [0.0, 0.0, 3.0, 1.0]。
            原来的 6 个位置各得到 4 个分数，共 24 个数；样本之间、位置之间都没有合并。
    约束：
        - 1 <= B, T, D <= 8，2 <= V <= 16；hidden 最后一维等于 weight 的行数。
        - 两个输入都是 CPU 上的 torch.float32 张量，数值有限。
        - 使用传入的 hidden 和 weight；保留 B、T，不混合不同位置的向量。
        - 本轮只做线性计算，不加偏置、不转换为概率、不直接选出预测 ID。
    思路：
        1. 核对向量长度 D 与权重行数 D 能对应。
        2. 对每个位置执行向量与输出权重的矩阵乘法，保留批次和位置两维。
    提示：
        - torch.matmul 或 @ 运算符支持带批次维的矩阵乘法。
        - 右侧二维权重可以由左侧的所有样本共用，不需要手动复制多份。
    """
    pass


# 以下 TODO 3 自检已经写好，不需要修改。
expected_logits = torch.tensor(
    [
        [[0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 1.0, 2.0], [0.0, 0.0, 3.0, 1.0]],
        [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 1.0, 2.0]],
    ],
    dtype=torch.float32,
)
logits = project_logits(h, W)
assert isinstance(logits, torch.Tensor), "请完成 TODO 3：返回候选词分数张量"
assert tuple(logits.shape) == (batch_size, block_size, vocab_size), "请完成 TODO 3：每个位置应有 V 个分数"
assert logits.dtype == h.dtype and logits.device == h.device, "请完成 TODO 3：保留浮点类型和设备"
assert torch.allclose(logits, expected_logits), "请完成 TODO 3：逐候选词核对乘积求和的结果"
other_logits = project_logits(
    torch.tensor([[[2.0, -1.0]]]),
    torch.tensor([[1.0, 3.0, 0.0], [2.0, -1.0, 4.0]]),
)
assert isinstance(other_logits, torch.Tensor), "请完成 TODO 3：也要能计算另一组向量和权重"
assert tuple(other_logits.shape) == (1, 1, 3), "请完成 TODO 3：候选词数量由 weight 决定"
assert torch.allclose(other_logits, torch.tensor([[[0.0, 7.0, -4.0]]])), "请完成 TODO 3：使用传入的权重计算"


def predict_ids(scores: torch.Tensor) -> torch.Tensor:
    """TODO 4：在每个位置选出分数最高的候选词 ID。

    上一题为每个位置保留了 V 个候选分数，这里为每个位置选一个预测结果。
    这种取最大分数的选择叫 greedy（贪心）选择，本轮用它直接观察前向计算的结果。

    参数：
        scores：形状为 [B, T, V] 的浮点张量，最后一维按 ID 顺序存放候选词分数。
    返回：
        形状为 [B, T]、dtype 为 torch.long 的张量。
        每个位置保存最大分数对应的 ID；分数并列时选择 ID 最小的那个。
    示例 1：
        输入：scores = [
                  [[0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 1.0, 2.0], [0.0, 0.0, 3.0, 1.0]],
                  [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, -1.0], [0.0, -1.0, 1.0, 2.0]],
              ]
              以上数组表示张量内容，B = 2，T = 3，V = 4。
        输出：
            pred = [
                [2, 3, 2],
                [0, 2, 3],
            ]
            shape = [2, 3]
        解释：第 0 条样本第 2 个位置的分数 [0.0, 0.0, 3.0, 1.0]
            → 最大分数为 3.0，它位于下标 2 → 预测 ID 为 2，对应“猫”。
            第 1 条样本第 0 个位置四个分数并列 → 选择最小 ID 0，对应“。”。
            六个位置各选一个 ID，候选词这一维被归约，样本和位置两维仍保留。
    示例 2：
        输入：scores = [[[-4.0, -2.0, -3.0, -5.0]]]，B = 1，T = 1，V = 4
        输出：pred = [[1]]，shape = [1, 1]
        解释：最大值是 -2.0，在下标 1，因此返回 ID 1；单条单位置的情况也保留两维。
    约束：
        - 1 <= B, T <= 8，2 <= V <= 16；分数有限，输入位于 CPU。
        - 在每个位置自己的 V 个候选词中比较，不跨样本或位置比较。
        - 返回整数 ID，不返回分数或字符；也不新增一维。
        - 直接根据传入的 scores 选择，不使用目标 yb，不进行随机采样。
    思路：
        1. 确认哪一维存放同一个位置的候选词分数。
        2. 沿着这一维查找最大分数所在的下标，保留其余维度并返回。
    提示：
        - argmax 返回最大值的下标；dim 参数指定在哪一维进行比较。
    """
    pass


# 以下 TODO 4 自检已经写好，不需要修改。
pred = predict_ids(logits)
assert isinstance(pred, torch.Tensor), "请完成 TODO 4：返回预测 ID 张量"
assert pred.dtype == torch.long and pred.device.type == "cpu", "请完成 TODO 4：返回 CPU 上的整数 ID"
assert tuple(pred.shape) == (batch_size, block_size), "请完成 TODO 4：每个位置选出一个 ID，保留 B、T"
assert torch.equal(pred, torch.tensor([[2, 3, 2], [0, 2, 3]])), "请完成 TODO 4：选最大分数的下标，并处理并列"
negative_pred = predict_ids(torch.tensor([[[-4.0, -2.0, -3.0, -5.0]]]))
assert isinstance(negative_pred, torch.Tensor), "请完成 TODO 4：全负分数也应返回张量"
assert torch.equal(negative_pred, torch.tensor([[1]])), "请完成 TODO 4：全负分数取最大值的下标，保留两维"


# 以下打印区已经写好，不需要修改。
# tolist() 将张量内容转成普通列表，方便对照题面的数组；不会改变张量本身。
print("输入形状 [B, T]：", list(x.shape))
print("输入 ID：", x.tolist())
print("目标 ID：", y.tolist())
print("embedding 形状 [B, T, D]：", list(h.shape))
print("embedding 向量：", h.tolist())
print("logits 形状 [B, T, V]：", list(logits.shape))
print("候选词分数：", logits.tolist())
print("预测形状 [B, T]：", list(pred.shape))
print("预测 ID：", pred.tolist())
pred_rows = pred.tolist()
for b in range(batch_size):
    print(f"第 {b} 条输入字符：", [vocab[i] for i in xb[b]])
    print(f"第 {b} 条目标字符：", [vocab[i] for i in yb[b]])
    print(f"第 {b} 条预测字符：", [vocab[i] for i in pred_rows[b]])

print("batch 张量、embedding 与 logits 前向检查通过")


# 本轮的边界：仅检查 CPU 上固定权重的查表、线性输出和贪心选择，不代表模型训练通过。
# 每个位置只使用当前 token 的向量，没有加入位置编码、attention、causal mask、FFN 或残差。
# FFN 是 feed-forward network（前馈网络），将在后续模型组件练习中介绍。
# 同一个输入 ID 在不同上下文中仍得到相同分数；这还不能表达依赖更长前文的预测。
# 没有计算概率、loss、梯度或参数更新，也没有训练/验证评测和保存恢复。
# A1 的完整训练、mask 和恢复验收仍待后续练习，不因本文件自检通过而勾选。
