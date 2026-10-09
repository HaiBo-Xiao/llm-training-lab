# 本脚本属于 M0 的环境与最小运行检查，还没有开始训练语言模型。
# 我们只训练一个数 w，让它从 0 向目标值 3 靠近，以检查：
# 1. GPU 上能否计算损失和梯度；
# 2. 优化器能否根据梯度修改参数；
# 3. 保存参数和优化器状态后，能否恢复出与连续训练一致的下一步。
#
# 阅读顺序：先看参数 w 和损失，再看 step()，最后看保存与恢复。
# assert 是检查语句：条件不满足就报错并停止，避免错误被悄悄忽略。

import io  # Python 标准库；BytesIO 可以充当保存在内存中的二进制文件。

import torch

assert torch.cuda.is_available()  # 确认当前 PyTorch 可以访问 CUDA GPU。
device = "cuda"

# tensor([0.0]) 创建形状为 [1] 的浮点张量，也就是只装一个数的容器。
# device=device 把它放到 GPU 上。
# Parameter 将张量标记为可训练参数，默认 requires_grad=True，允许求梯度。
# 这里没有神经网络，w 就是全部的可训练参数。
w = torch.nn.Parameter(torch.tensor([0.0], device=device))

# 优化器负责用梯度更新参数。[w] 是交给它管理的参数列表。
# SGD 是一种梯度下降优化器；lr=0.1 表示学习率，控制每次更新的幅度。
# momentum=0.9 表示保留上一步动量的 90%，使更新携带历史信息。
# 这里特意使用动量，是为了验证保存恢复时是否正确恢复了历史信息。
#
# 对本脚本的设置，可以用下面的公式理解更新（g 是当前梯度，b 是动量）：
#   第一步：b = g
#   后续步：b = 0.9 * 上一步的 b + g
#   更新参数：w = w - 0.1 * b
optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)


def step(parameter, optim):
    # 本函数完成一次训练更新。parameter、optim 是函数内使用的名字，
    # 调用 step(w, optimizer) 时，它们对应外面的 w 和 optimizer。

    # ① 清理上一步梯度。
    # PyTorch 默认会把 backward() 的梯度累加到 .grad 中，因此每一步先清理。
    # set_to_none=True 把 .grad 设为 None，随后 backward() 会重新生成梯度。
    # 这不会清除优化器的动量；梯度和动量是不同的状态。
    optim.zero_grad(set_to_none=True)

    # ② 前向计算：用平方误差衡量参数距离目标 3 有多远。
    # square() 对每个元素求平方，sum() 把结果加成一个标量损失。
    # 这里仅有一个元素，所以 loss = (w - 3)^2。
    # w=0 时 loss=9；w=3 时 loss=0，是这个问题的最小值。
    loss = (parameter - 3).square().sum()

    # ③ 反向传播：沿前向计算建立的计算图求导，将结果写入 parameter.grad。
    # 这个例子的导数是 d(loss)/dw = 2 * (w - 3)。
    # 第一步 w=0，所以梯度是 -6。此时只算出了梯度，w 还没有改变。
    loss.backward()

    # 检查损失和梯度是否为有限数值，排除 NaN（非数）和 Inf（无穷大）。
    # isfinite() 返回布尔张量；all() 要求所有元素都通过；
    # item() 把单元素张量转换为普通 Python 数值或布尔值。
    # 对 GPU 张量调用 item() 需要将数值取回 CPU，这里用于小规模检查。
    assert torch.isfinite(loss).item()
    assert torch.isfinite(parameter.grad).all().item()

    # abs().sum() 计算梯度绝对值之和；大于 0 表明本次有非零更新信号。
    # 这只适用于本例的两次受控更新。一般训练在最优点出现零梯度可以是正常的。
    assert parameter.grad.abs().sum().item() > 0

    # ④ 根据梯度和优化器状态更新参数；真正改变 w 的是这一行。
    # 第一步：g=-6，b=-6，w=0 - 0.1*(-6)=0.6。
    optim.step()
    assert torch.isfinite(parameter).all().item()

    # 返回的是②中计算的、更新前的损失；没有在更新后重新计算损失。
    return loss.item()


# 执行第一次更新，并与手算结果 0.6 比较。
# detach() 得到不参与自动求导的张量视图；这里比较数值不需要计算梯度。
# allclose() 允许很小的浮点误差，因为计算机中的小数运算不总是精确的。
loss = step(w, optimizer)
assert torch.allclose(w.detach(), torch.tensor([0.6], device=device))
# 所以这一行同时显示“更新前的 loss=9”和“更新后的 w=0.6”，并不矛盾。
print(f"第一次更新：loss={loss:.6f}, w={w.item():.6f}")

# 在第一次更新完成时保存快照，再分两条路径验证：
#   路径 A：当前参数与优化器直接继续第二步。
#   路径 B：新建参数与优化器，加载快照，再执行第二步。
# 两条路径应得到相同结果。
#
# BytesIO 创建内存中的文件对象。torch.save 将对象序列化为字节写进去。
# 这次没有生成磁盘上的 .pt 文件，程序退出后该内存快照就消失了。
checkpoint = io.BytesIO()
torch.save(
    # detach() 脱离自动求导，clone() 复制数值，避免保存对象共享 w 的存储。
    # state_dict() 是优化器的状态字典，包括动量、学习率及参数组等信息。
    # 它不代替参数值本身，所以这里分别保存 parameter 和 optimizer。
    # torch.save 在这里立即写出字节，后续训练不会改变已经写出的快照。
    {"parameter": w.detach().clone(), "optimizer": optimizer.state_dict()},
    checkpoint,
)

# 路径 A：从 w=0.6 继续训练。
#   当前梯度 g = 2*(0.6-3) = -4.8
#   新动量 b = 0.9*(-6) + (-4.8) = -10.2
#   新参数 w = 0.6 - 0.1*(-10.2) = 1.62
step(w, optimizer)
# 复制本次结果，作为路径 B 的预期值。
expected = w.detach().clone()

# 路径 B：读取第一次更新后的快照。
# 写入后文件位置在末尾，seek(0) 将读取位置移回开头。
checkpoint.seek(0)
# map_location=device 指定张量加载到 GPU。
# weights_only=True 使用受限加载方式，支持本例的张量和普通状态字典；
# 它不是“只读取 parameter、忽略 optimizer”的意思。
state = torch.load(checkpoint, map_location=device, weights_only=True)
# 新建可训练参数，初始值来自快照，所以 restored_w=0.6。
restored_w = torch.nn.Parameter(state["parameter"].clone())
# 新优化器必须绑定新参数 restored_w，才能更新恢复出来的这份参数。
restored_optimizer = torch.optim.SGD([restored_w], lr=0.1, momentum=0.9)
# 恢复第一步累积的动量 -6，以及保存的优化器配置。
# 如果省略这一行，新优化器没有历史动量，下一步会得到 1.08，而非 1.62。
restored_optimizer.load_state_dict(state["optimizer"])

# 对恢复出来的训练状态执行下一步，再与连续训练结果比较。
step(restored_w, restored_optimizer)
# 若差异超过默认浮点容差，assert_close 会报错；通过说明本例恢复结果一致。
torch.testing.assert_close(restored_w.detach(), expected)

# :.6f 是输出格式：保留小数点后六位。
print(f"连续训练第二步：w={expected.item():.6f}")
print(f"恢复训练第二步：w={restored_w.item():.6f}")
print("参数更新与训练状态恢复检查通过")

# 本检查的边界：只验证一个参数、两步更新和内存快照恢复。
# 它不代表完整训练已可复现。后续正式训练还需按实际使用情况保存模型、
# 随机数状态、数据读取位置、学习率调度器状态等，并验证磁盘保存和独立重启恢复。
