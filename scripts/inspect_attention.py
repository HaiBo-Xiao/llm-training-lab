"""M1 下一步练习：单头因果自注意力。

你要完成四个函数，请按 TODO 1～4 的顺序填写。
给定 Q、K、V，依次计算注意力分数、因果遮罩、softmax 权重和加权输出。
检查区、打印区和已标注“已经写好”的代码不需要修改。

运行方式（在项目根目录）：
    uv run --locked python scripts/inspect_attention.py

当前文件是练习骨架。函数未填写时返回 None，自检会在对应 TODO 给出中文提示。
本轮使用 CPU 上的固定小张量，不构造完整模型，不计算 loss，不反向传播，也不更新参数。

数据依次经过：
    Q、K、V → 缩放点积 scores → causal mask → softmax 权重 → 加权输出。
本轮只处理一个样本和一个 attention head；后续再把它放进 decoder-only 模型。
"""

import math

import torch

# 以下数据与配置已经写好，不需要修改。
# T 是序列长度，D 是每个 token 向量的长度。
Q = torch.tensor(
    [
        [
            1.0,
            0.0
        ],
        [
            0.0,
            1.0
        ],
        [
            1.0,
            1.0
        ],
    ],
    dtype=torch.float32,
)
K = torch.tensor(
    [
        [
            1.0,
            0.0
        ],
        [
            0.0,
            1.0
        ],
        [
            1.0,
            1.0
        ],
    ],
    dtype=torch.float32,
)
V = torch.tensor(
    [
        [
            1.0,
            0.0
        ],
        [
            0.0,
            1.0
        ],
        [
            2.0,
            2.0
        ],
    ],
    dtype=torch.float32,
)
T, D = Q.shape


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """TODO 1：计算缩放点积注意力分数。

    每个查询向量都要和所有键向量比较，得到它对各位置的未归一化分数。
    除以 sqrt(D) 可以随着向量维度增大而控制点积的尺度，避免 softmax 输入过大。

    参数：
        q：形状为 [T, D] 的查询向量。
        k：形状为 [T, D] 的键向量。
    返回：
        形状为 [T, T] 的分数矩阵。第 i 行第 j 列表示位置 i 对位置 j 的分数。
    示例 1：

            q = [[1.0, 0.0], [0.0, 1.0]]
            k = [[1.0, 0.0], [0.0, 1.0]]
        输出：
            scores = [[0.7071, 0.0000], [0.0000, 0.7071]]
        解释：D = 2，sqrt(D) = sqrt(2)。第 0 个查询与第 0 个键的点积是 1，
            与第 1 个键的点积是 0；每个点积都除以 sqrt(2)。
    约束：
        - q、k 都是 CPU 上的 torch.float32 张量，形状均为 [T, D]。
        - q 和 k 的 T、D 必须对应；本轮不处理空张量或形状错误。
        - 只计算分数，不做 mask、softmax 或加权求和。
    思路：
        1. 让每个查询向量与每个键向量做点积。
        2. 用键向量的最后一维长度确定缩放因子。
        3. 返回形状为 [T, T] 的浮点分数矩阵。
    提示：
        - 转置 k 的最后两维后再做矩阵乘法。
        - math.sqrt(...) 可以计算平方根。
    """
    return q @ k.T / math.sqrt(k.shape[-1])


# 以下 TODO 1 自检已经写好，不需要修改。
scores = attention_scores(Q, K)
expected_scores = torch.tensor(
    [
        [
            1.0 / math.sqrt(2.0),
            0.0,
            1.0 / math.sqrt(2.0)
        ],
        [
            0.0,
            1.0 / math.sqrt(2.0),
            1.0 / math.sqrt(2.0)
        ],
        [
            1.0 / math.sqrt(2.0),
            1.0 / math.sqrt(2.0),
            2.0 / math.sqrt(2.0)
        ],
    ],
    dtype=torch.float32,
)
assert isinstance(scores, torch.Tensor), "请完成 TODO 1：返回分数张量"
assert tuple(scores.shape) == (
    T,
    T
), "请完成 TODO 1：分数矩阵应为 [T, T]"
assert torch.allclose(scores, expected_scores), "请完成 TODO 1：按缩放点积计算每个位置的分数"


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """TODO 2：屏蔽每个位置之后的未来信息。

    位置 i 只能读取位置 0 到 i 的分数，不能读取 j > i 的位置。
    被屏蔽的位置要在 softmax 前变成负无穷，使它们归一化后的权重为 0。

    参数：
        scores：形状为 [T, T] 的未遮罩分数矩阵。
    返回：
        形状仍为 [T, T] 的分数矩阵；下三角保留原分数，上三角表示不可见位置。
    示例 1：
        输入：scores = [[1.0, 2.0], [3.0, 4.0]]
        输出：masked = [[1.0, -inf], [3.0, 4.0]]
        解释：第 0 行只能看位置 0；第 1 行可以看位置 0 和 1。
    约束：
        - 输入位于 CPU，形状为 [T, T]，数值有限。
        - 不改变可见位置的原分数，不改变矩阵形状。
        - 本轮不处理 padding mask；只屏蔽未来位置。
    思路：
        1. 根据行号和列号判断哪些位置在当前位置之后。
        2. 只替换上三角的未来分数。
        3. 返回遮罩后的新矩阵。
    提示：
        - torch.triu 可以生成上三角区域；masked_fill 可以按布尔条件替换数值。
    """
    mask = torch.triu(#上下三角
        torch.ones_like(scores, dtype=torch.bool),
        diagonal=1)
    return scores.masked_fill(mask, float("-inf"))


# 以下 TODO 2 自检已经写好，不需要修改。
masked_scores = apply_causal_mask(scores)
assert isinstance(masked_scores, torch.Tensor), "请完成 TODO 2：返回遮罩后的分数张量"
assert tuple(masked_scores.shape) == (
    T,
    T
), "请完成 TODO 2：遮罩不能改变分数矩阵形状"
assert torch.allclose(
    masked_scores[torch.tril(torch.ones(T, T, dtype=torch.bool))],
    scores[torch.tril(torch.ones(T, T, dtype=torch.bool))],
), "请完成 TODO 2：保留下三角的可见分数"
assert torch.isneginf(masked_scores[0, 1:]).all(), "请完成 TODO 2：第 0 行不能读取未来位置"
assert torch.isneginf(masked_scores[1, 2:]).all(), "请完成 TODO 2：第 1 行不能读取位置 2"


def attention_weights(masked_scores: torch.Tensor) -> torch.Tensor:
    """TODO 3：把遮罩后的分数转换为注意力权重。

    softmax 将同一行的候选位置分数转换为非负权重，并让每行权重和为 1。
    未来位置已经被遮罩，因此这些位置的权重必须保持为 0。

    参数：
        masked_scores：形状为 [T, T] 的已遮罩分数矩阵。
    返回：
        形状为 [T, T] 的权重矩阵。第 i 行是位置 i 对所有可见位置的权重。
    示例 1：
        输入：masked_scores = [[1.0, -inf], [3.0, 4.0]]
        输出：weights = [[1.0, 0.0], [0.2689, 0.7311]]
        解释：第 0 行只有一个可见位置，所以权重为 1；第 1 行对两个位置做 softmax，
            exp(3) / (exp(3) + exp(4)) 约为 0.2689。
    约束：
        - 输入形状为 [T, T]，上三角已被设为负无穷。
        - 沿最后一维归一化；不跨行混合不同查询位置。
        - 返回浮点张量，不返回索引或 Python 列表。
    思路：
        1. 找到表示候选位置的维度。
        2. 沿该维度执行 softmax。
        3. 返回每个位置的注意力权重。
    提示：
        - torch.softmax 的 dim 参数指定归一化的维度。
    """
    return torch.softmax(masked_scores, dim=-1)


# 以下 TODO 3 自检已经写好，不需要修改。
weights = attention_weights(masked_scores)
assert isinstance(weights, torch.Tensor), "请完成 TODO 3：返回注意力权重张量"
assert tuple(weights.shape) == (
    T,
    T
), "请完成 TODO 3：权重矩阵应为 [T, T]"
assert torch.allclose(weights.sum(dim=-1), torch.ones(T)), "请完成 TODO 3：每行权重和应为 1"
assert torch.equal(weights.triu(1), torch.zeros(T, T)), "请完成 TODO 3：未来位置权重应为 0"


def attend_values(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """TODO 4：用注意力权重汇总 V 向量。

    每个位置输出一个 V 的加权平均向量；权重大的位置对结果贡献更大。
    这一步保留查询位置维度，并把最后一维从 T 个权重变回 D 个向量分量。

    参数：
        weights：形状为 [T, T] 的注意力权重矩阵。
        values：形状为 [T, D] 的值向量矩阵。
    返回：
        形状为 [T, D] 的输出矩阵。第 i 行是 weights 第 i 行对所有 values 的加权和。
    示例 1：
        输入：
            weights = [[1.0, 0.0], [0.25, 0.75]]
            values = [[2.0, 0.0], [0.0, 4.0]]
        输出：output = [[2.0, 0.0], [0.5, 3.0]]
        解释：第 0 行完全取第 0 个值向量；第 1 行得到
            0.25 * [2.0, 0.0] + 0.75 * [0.0, 4.0] = [0.5, 3.0]。
    约束：
        - weights 为 [T, T]，values 为 [T, D]，二者位于 CPU 且为浮点张量。
        - 只做加权求和，不再次计算分数或 softmax。
        - 返回张量，保留查询位置数 T 和向量维度 D。
    思路：
        1. 确认权重的列与 values 的行都表示同一个源位置。
        2. 对每个查询位置执行矩阵乘法。
        3. 返回形状为 [T, D] 的结果。
    提示：
        - 使用 @ 或 torch.matmul 做矩阵乘法。
    """
    return weights @ values


# 以下 TODO 4 自检已经写好，不需要修改。
output = attend_values(weights, V)
assert isinstance(output, torch.Tensor), "请完成 TODO 4：返回注意力输出张量"
assert tuple(output.shape) == (
    T,
    D
), "请完成 TODO 4：输出形状应为 [T, D]"
assert torch.allclose(output[0], V[0]), "请完成 TODO 4：第 0 个位置只能汇总第 0 个值向量"

# 改变未来位置的值，前面位置的输出不应改变；位置 2 可以看到自身，因此允许改变。
future_changed = V.clone()
future_changed[2] = torch.tensor(
    [
        100.0,
        100.0
    ],
)
changed_output = attend_values(weights, future_changed)
assert torch.allclose(changed_output[:2], output[:2]), (
    "请完成 TODO 4：因果遮罩应阻止未来值影响前两个位置"
)
assert not torch.allclose(changed_output[2], output[2]), (
    "请完成 TODO 4：当前位置可以读取自己的值"
)

# 以下打印区已经写好，不需要修改。
print("Q 形状：", list(Q.shape))
print("K 形状：", list(K.shape))
print("V 形状：", list(V.shape))
print("未遮罩分数：", scores.tolist())
print("因果遮罩后的分数：", masked_scores.tolist())
print("注意力权重：", weights.tolist())
print("注意力输出：", output.tolist())
print("因果自注意力检查通过")

# 本轮边界：只验证单样本、单头、固定权重的因果注意力。
# 没有加入 batch 维、多头投影、位置编码、dropout、FFN、残差、loss、梯度或参数更新。
# padding mask、有效 token 数和完整 decoder-only 模型留到后续练习。
