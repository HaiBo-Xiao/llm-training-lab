"""M1 下一步练习：next-token logits 与有效 token 的交叉熵损失。

你要完成四个函数，请按 TODO 1～4 的顺序填写。
给定已经对齐的 logits 和目标 ID，依次整理损失输入、计算逐位置损失、
构造有效 token mask，并只对有效位置求平均。
检查区、打印区和已标注“已经写好”的代码不需要修改。

运行方式（在项目根目录）：
    uv run --locked python scripts/05_inspect_loss.py

当前文件是练习骨架。函数未填写时返回 None，自检会在对应 TODO 给出中文提示。
本轮使用 CPU 上的固定小张量，不构造完整模型，不反向传播，也不更新参数。

上一轮已经得到形状为 [B, T, V] 的 logits。本轮将它与形状为 [B, T] 的目标对齐：
    logits、目标 → 展平为交叉熵输入 → 每个位置的损失 → 有效 token mask → 平均损失。
目标已经是 next-token 目标，本轮不能再次位移。`ignore_index` 表示不参与损失的
位置，例如 padding；平均时分母只能使用有效 token 数。
"""

import torch
import torch.nn.functional as F


# 以下数据与配置已经写好，不需要修改。
# B 是样本数，T 是每条样本的预测位置数，V 是候选 token 数。
B, T, V = 2, 3, 4
ignore_index = -100

# 每个位置有 V 个 logits；目标已经与这些位置一一对齐。
# 第二条样本的中间位置用 ignore_index 模拟 padding，不应计入平均损失。
logits = torch.tensor(
    [
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        [[0.0, 0.0, 0.0, 2.0], [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 2.0]],
    ],
    dtype=torch.float32,
)
targets = torch.tensor(
    [
        [0, 1, 2],
        [3, ignore_index, 0],
    ],
    dtype=torch.long,
)


def flatten_for_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TODO 1：整理交叉熵所需的二维 logits 和一维目标。

    交叉熵把每一行看作一个预测位置，把最后一维当作候选 token。这里把批次维和
    位置维合并，保留它们原来的行优先顺序；目标也按相同顺序展开，才能一一对应。

    参数：
        scores：形状为 [B, T, V] 的 logits，最后一维是候选 token 分数。
        labels：形状为 [B, T] 的目标 ID，已经完成 next-token 对齐。
    返回：
        `(flat_scores, flat_labels)`；前者形状为 [B*T, V]，后者形状为 [B*T]。
        第 n 行 logits 与第 n 个目标仍表示同一个样本、同一个位置。
    示例 1：
        输入：scores = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
              labels = [[0, 1], [1, 0]]，B = 2，T = 2，V = 2。
        输出：
            flat_scores = [[1, 2], [3, 4], [5, 6], [7, 8]]
            flat_labels = [0, 1, 1, 0]
        解释：先保留第 0 条样本的两个位置，再保留第 1 条样本的两个位置；
            只合并 B、T 两个维度，候选 token 维 V 仍保留为每行的列。
    约束：
        - `scores` 与 `labels` 的前两维对应，且输入非空。
        - `labels` 中可以出现 `ignore_index`，它仍要保留在对应位置，不能删除或改编号。
        - 不重新位移目标，不根据分数选择预测 ID。
    思路：
        1. 将每个样本的每个位置按原顺序排成连续的预测行。
        2. 用完全相同的顺序整理目标。
        3. 返回两份形状与位置对应关系一致的张量。
    提示：
        - `reshape` 可以合并前面的维度并保留元素顺序。
        - 目标的展平长度必须与 logits 的预测行数相同。
    """
    pass


# 以下 TODO 1 自检已经写好，不需要修改。
flat_logits, flat_targets = flatten_for_loss(logits, targets)
assert isinstance(flat_logits, torch.Tensor), "请完成 TODO 1：返回展平后的 logits"
assert isinstance(flat_targets, torch.Tensor), "请完成 TODO 1：返回展平后的目标"
assert tuple(flat_logits.shape) == (B * T, V), "请完成 TODO 1：logits 应为 [B*T, V]"
assert tuple(flat_targets.shape) == (B * T,), "请完成 TODO 1：目标应为 [B*T]"
assert flat_logits.dtype == logits.dtype and flat_targets.dtype == targets.dtype, (
    "请完成 TODO 1：保留 logits 和目标的 dtype"
)
assert torch.equal(
    flat_targets,
    torch.tensor([0, 1, 2, 3, ignore_index, 0], dtype=torch.long),
), "请完成 TODO 1：目标顺序应保持 next-token 对齐"
assert torch.equal(flat_logits, logits.reshape(B * T, V)), (
    "请完成 TODO 1：保留每个位置的候选分数"
)


def token_cross_entropy(
    scores: torch.Tensor,
    labels: torch.Tensor,
    ignored: int,
) -> torch.Tensor:
    """TODO 2：计算每个位置的交叉熵损失。

    交叉熵先在候选 token 维度上把 logits 转成目标类别的负对数似然，再还原为
    [B, T]，让每个位置的损失仍能和原来的样本、位置对应。被忽略的目标位置输出 0，
    但是否计入平均由下一步的有效 token mask 决定。

    参数：
        scores：形状为 [B, T, V] 的 logits。
        labels：形状为 [B, T] 的目标 ID，可能含有忽略标记。
        ignored：不参与损失的位置标记，本轮为 -100。
    返回：
        形状为 [B, T] 的逐位置损失。有效位置为非负浮点数，被忽略位置为 0。
    示例 1：
        输入：scores = [[[2, 0, 0], [0, 2, 0]]]
              labels = [[0, -100]]，ignored = -100。
        输出：losses = [[log(exp(2)+2)-2, 0.0]]
        解释：第 0 个位置监督 ID 0；第二个位置被忽略，所以它的损失不产生梯度贡献，
            这里用 0 表示该位置没有有效损失。
    约束：
        - `scores` 与 `labels` 的 B、T 对应，候选 token 在最后一维。
        - 使用传入的 `ignored`，不能把它当作有效词表 ID。
        - 本轮不做 label smoothing，不改变 logits，不再移动 labels。
    思路：
        1. 将批次和位置整理成交叉熵接口需要的布局。
        2. 让交叉熵按位置返回损失，而不是提前求整批平均。
        3. 将结果恢复为 [B, T]，以便后续按有效 token 筛选。
    提示：
        - `torch.nn.functional.cross_entropy` 的 `reduction="none"` 会保留每个位置的损失。
        - `ignore_index` 参数可以指定需要忽略的目标标记。
    """
    pass


# 以下 TODO 2 自检已经写好，不需要修改。
per_token_loss = token_cross_entropy(logits, targets, ignore_index)
v = torch.log(torch.exp(torch.tensor(2.0)) + 3.0) - 2.0
expected_loss = torch.tensor(
    [
        [v, v, v],
        [v, 0.0, v + 2.0],
    ],
)
assert isinstance(per_token_loss, torch.Tensor), "请完成 TODO 2：返回逐位置损失张量"
assert tuple(per_token_loss.shape) == (B, T), "请完成 TODO 2：损失应恢复为 [B, T]"
assert per_token_loss.dtype == logits.dtype, "请完成 TODO 2：损失使用浮点 dtype"
assert torch.allclose(per_token_loss, expected_loss), (
    "请完成 TODO 2：按目标 ID 计算每个位置的交叉熵，并忽略指定位置"
)


def valid_token_mask(labels: torch.Tensor, ignored: int) -> torch.Tensor:
    """TODO 3：标记哪些目标位置应参与损失平均。

    有效 token 是目标不等于忽略标记的位置。这个布尔矩阵与 [B, T] 的逐位置损失
    一一对应，后续可以用它排除 padding 或其它没有监督信号的位置。

    参数：
        labels：形状为 [B, T] 的目标 ID。
        ignored：不参与监督的目标标记。
    返回：
        形状为 [B, T] 的布尔张量；True 表示该位置有效，False 表示应排除。
    示例 1：
        输入：labels = [[2, -100, 3]]，ignored = -100。
        输出：mask = [[True, False, True]]
        解释：第 1 个位置是忽略标记；有效 token 数是两个 True，而不是三个位置总数。
    约束：
        - 返回值只表示位置是否有效，不修改 labels，也不计算损失。
        - 保留 labels 的 B、T 两维；本轮至少有一个有效位置。
    思路：
        1. 逐位置比较目标是否等于忽略标记。
        2. 将比较结果作为布尔 mask 返回。
    提示：
        - 张量比较会逐元素返回布尔结果。
    """
    pass


# 以下 TODO 3 自检已经写好，不需要修改。
valid_mask = valid_token_mask(targets, ignore_index)
expected_mask = torch.tensor(
    [
        [True, True, True],
        [True, False, True],
    ],
)
assert isinstance(valid_mask, torch.Tensor), "请完成 TODO 3：返回布尔张量"
assert valid_mask.dtype == torch.bool, "请完成 TODO 3：mask 的 dtype 应为 torch.bool"
assert tuple(valid_mask.shape) == (B, T), "请完成 TODO 3：mask 应保留 [B, T]"
assert torch.equal(valid_mask, expected_mask), "请完成 TODO 3：只有非 ignore_index 位置有效"
assert int(valid_mask.sum().item()) == 5, "请完成 TODO 3：有效 token 数应为 5"


def mean_valid_loss(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TODO 4：只对有效 token 的损失求平均。

    训练中的 loss 应反映有效监督位置的平均质量。若直接对整个 [B, T] 求平均，
    padding 数量变化会改变分母；这里必须使用 mask 选出的有效位置数作为分母。

    参数：
        losses：形状为 [B, T] 的逐位置损失。
        mask：形状为 [B, T] 的布尔有效位置标记。
    返回：
        一个 0 维浮点张量，等于所有有效位置损失之和除以有效 token 数。
    示例 1：
        输入：losses = [[1.0, 3.0, 5.0]]，mask = [[True, False, True]]。
        输出：loss = 3.0。
        解释：只保留 1.0 和 5.0；分母是 2，所以平均值为 (1.0 + 5.0) / 2。
    约束：
        - `losses` 与 `mask` 形状相同，且至少有一个有效位置。
        - 不把无效位置的 0 当作额外样本，不使用 B*T 作为分母。
        - 返回标量张量，不返回 Python float。
    思路：
        1. 用 mask 选出有效位置的损失。
        2. 统计有效位置数量。
        3. 用有效损失总和除以这个数量。
    提示：
        - 布尔索引可以选出有效位置；`numel()` 或 `sum()` 可以得到数量。
    """
    pass


# 以下 TODO 4 自检已经写好，不需要修改。
loss = mean_valid_loss(per_token_loss, valid_mask)
expected_mean = v + 0.4
assert isinstance(loss, torch.Tensor), "请完成 TODO 4：返回损失张量"
assert loss.ndim == 0, "请完成 TODO 4：平均损失应是 0 维标量"
assert loss.dtype == logits.dtype, "请完成 TODO 4：保留损失的浮点 dtype"
assert torch.allclose(loss, expected_mean), (
    "请完成 TODO 4：平均值的分母应是有效 token 数，而不是全部位置数"
)


# 以下打印区已经写好，不需要修改。
print("logits 形状 [B, T, V]：", list(logits.shape))
print("目标形状 [B, T]：", list(targets.shape))
print("展平 logits 形状 [B*T, V]：", list(flat_logits.shape))
print("展平目标：", flat_targets.tolist())
print("逐位置损失：", per_token_loss.tolist())
print("有效 token mask：", valid_mask.tolist())
print("有效 token 数：", int(valid_mask.sum().item()))
print(f"有效 token 平均损失：{loss.item():.6f}")
print("logits、目标与有效 token 损失检查通过")

# 本轮边界：只验证固定 logits、目标对齐、ignore_index 和有效 token 平均损失。
# 没有执行 backward、optimizer 更新、梯度累积、混合精度或完整 decoder-only 模型训练。
