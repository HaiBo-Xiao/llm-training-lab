"""M1 第二个练习：训练/验证划分，以及定长样本与批的构造。

你要完成的是两个数据槽位和四个函数，共 6 个 TODO。
请按 TODO 1～6 的顺序填写。检查区、打印区和已标注“已经写好”的代码不需要修改。

运行方式（在项目根目录）：
    uv run --locked python scripts/split_dataset.py

当前文件是练习骨架。数据槽位未填写时报 NameError，函数槽位未填写时返回 None
并由后面的 assert 给出中文提示。这两种失败都是预期行为。

本轮使用普通 Python，不需要 PyTorch，不构造张量，也不会执行模型训练。
与上一个练习 scripts/inspect_tokens.py 的关键区别：词表只用训练集构建，
因此验证集里会出现词表外的字符，本轮必须处理它们，不能假设输入都在词表内。
"""
import math

# 语料。仍然规定一个字符就是一个 token，标点也算。
# 全文共 185 个字符，短到每一步都能用肉眼核对。
text = (
    "小猫住在河边的木屋里。每天清晨，它先到河滩上散步，再回屋里吃鱼。"
    "河水很清，能看见石头缝里的小虾。小猫不吃虾，只吃鱼，因为鱼刺少而虾壳硬。"
    "有一天，河水忽然涨起来，木屋前的石阶被淹掉两级。小猫站在门口看了很久，没有下水。"
    "它想起老猫说过的话：水涨的时候不要靠近河心，那里的水会把猫卷走。"
    "于是小猫回到屋里，把门关好，趴在窗台上等水退。"
    "第二天早上，水果然退了，河滩上留下一层细沙。"
)

# 训练集占的比例。剩下的部分作为验证集。
# 训练集用来更新参数，验证集只用来观察，不参与任何参数更新和词表构建。
# 为什么需要验证集：PLAN.md 的 M1-5 要求先在一个小批次上故意过拟合，
# 那时训练损失会降到接近 0。只有验证集能区分“学会了规律”和“背下了这段文本”。
train_ratio = 0.9

# 上下文长度，即一条训练样本包含多少个位置。
# 真实训练里它常取 256、512 这样的值，本轮取 8 是为了能手算核对。
block_size = 8

# 一批（batch）包含多少条样本。模型一次前向处理一整批，而不是一条。
batch_size = 4

# 词表外记号。unk 是 unknown 的缩写，用来表示“这个字符不在词表里”。
# 写成尖括号形式，是为了和语料里真正的字符区分开，肉眼一看就知道它不是原文内容。
unk = "<unk>"


# TODO 1：填写 train_text 和 val_text，这是数据，不是函数。
# 目标：把 text 按位置切成前后两段，前一段是训练集，后一段是验证集。
# 切分点取 int(len(text) * train_ratio)：185 乘 0.9 得 166.5，int 截断为 166。
# 因此 train_text 是前 166 个字符，val_text 是后 19 个字符。
# 约束：必须按位置切，不能打乱后再分。语言模型的相邻字符互为上下文，
#     打乱再切会让验证集每个位置的前文仍留在训练集里，等于提前看过答案。
# 约束：两段拼起来必须原样还原 text，不丢字符、不改顺序。
# 提示：字符串切片可以取出前 n 个字符和第 n 个字符之后的部分。
cut = math.floor(len(text)*train_ratio)
train_text: str=text[:cut]
val_text: str=text[cut:]



# 以下检查已经写好，不需要修改。
assert len(train_text) == 166, "请完成 TODO 1：训练集应为前 166 个字符"
assert len(val_text) == 19, "请完成 TODO 1：验证集应为剩下的 19 个字符"
assert train_text + val_text == text, "请完成 TODO 1：两段拼接应原样还原 text"
assert text.startswith(train_text), "请完成 TODO 1：训练集取原文靠前的一段，不能打乱后再抽取"


# TODO 2：填写词表 vocab，这是数据，不是函数。
# 目标：第 0 位放词表外记号 unk，之后按升序排列训练集里所有不同的字符。
# 训练集有 94 个不同字符，加上记号，词表共 95 项。
# 示例：若训练集为 "aab"，则 vocab 为 ["<unk>", "a", "b"]（与本例无关的例子）。
# 约束：只统计 train_text，不能用 text 或 val_text。用全量文本建词表，验证集的字符
#     就提前进入了编号体系，词表大小和每个 ID 都受它影响，这是一种数据泄漏。
# 约束：记号固定放在第 0 位，好处是它的 ID 恒为 0，不随语料变化。
# 提示：set(...) 可以去重，sorted(...) 可以固定顺序；两个列表可以直接相加。
vocab: list[str]=['<unk>']+sorted(set(train_text))
print(len(vocab))

# 以下检查已经写好，不需要修改。
assert vocab[0] == unk, "请完成 TODO 2：词表第 0 位应是词表外记号"
assert len(vocab) == 95, "请完成 TODO 2：一个记号加训练集里 94 个不同字符"
assert vocab[1:] == sorted(vocab[1:]), "请完成 TODO 2：记号之后的字符应按升序排列"
assert set(vocab[1:]) == set(train_text), "请完成 TODO 2：词表只能由训练集的字符构成"


# 以下两份映射沿用上一个练习的做法，已经写好，不需要修改。
# stoi 是 string-to-integer 的缩写，itos 是 integer-to-string 的缩写。
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def decode(ids: list[int]) -> str:
    """把 ID 列表还原成字符串。已经写好，不需要修改。

    与上一个练习的 decode 相同。注意记号的 ID 会还原成 "<unk>" 这五个字符，
    所以编码再解码不一定得回原文——词表外的字符已经在编码时丢失了信息。
    """
    return "".join(itos[i] for i in ids)


def unknown_chars(s: str) -> list[str]:
    """TODO 3：找出字符串 s 里所有不在词表中的字符。

    这一步只观察、不替换，目的是看清“词表只用训练集构建”带来的后果。

    参数：
        s：任意字符串。
    返回：
        s 中不在词表内的字符，去重后按升序排列；全部都在词表内时返回空列表。
    示例 1：
        输入：s = val_text
        输出：["层", "早", "果", "沙", "留", "细"]
        解释：这 6 个字符只出现在验证集里，训练集没有，所以词表里也没有。
    示例 2：
        输入：s = train_text
        输出：[]
        解释：词表就是由训练集的字符建起来的，训练集里不会有词表外字符。
    约束：
        - 1 <= len(s) <= 10^4
        - 用 stoi 判断，不要重新统计 train_text。
        - 记号 unk 不是语料里的字符，s 中不会出现它。
    思路：
        1. 逐个取出 s 中的字符，挑出不在 stoi 里的那些。
        2. 去重。
        3. 按升序排列后 return。
    提示：
        - in 可以判断一个键是否在字典里。
        - set(...) 去重，sorted(...) 排序。
    """
    ans=[]
    for ch in s:
        if ch not in vocab:
            ans.append(ch)
    # print(ans)
    return sorted(set(ans))


# 以下检查已经写好，不需要修改。
assert unknown_chars(train_text) == [], "请完成 TODO 3：训练集里不应有词表外字符"
assert unknown_chars(val_text) == ["层", "早", "果", "沙", "留", "细"], (
    "请完成 TODO 3：找出验证集里的词表外字符，去重并升序排列"
)


def encode(s: str) -> list[int]:
    """TODO 4：把字符串 s 转换成字符 ID 列表，词表外的字符映射到记号的 ID。

    与上一个练习的 encode 只有一处不同：那时约定输入都在词表内，直接查表即可；
    现在词表来自训练集，验证集必然有查不到的字符，必须给它们一个明确的归宿。

    参数：
        s：任意字符串，可以包含词表外字符。
    返回：
        与 s 等长的 ID 列表。
    示例 1：
        输入：s = "小猫吃鱼"
        输出：[34, 60, 25, 92]
        解释：四个字符都在词表内，按 stoi 逐个查出编号。
    示例 2：
        输入：s = "小猫吃沙"
        输出：[34, 60, 25, 0]
        解释："沙" 只出现在验证集里，词表内没有，映射到记号的 ID 0。
    约束：
        - 1 <= len(s) <= 10^4
        - 返回长度必须等于 len(s)。遇到词表外字符不能跳过：跳过会让长度悄悄变短，
          错误被隐藏，之后按长度对齐输入和目标时查不出原因。
        - 记号的 ID 恒为 0，因为 TODO 2 已把它固定在词表第 0 位。
    思路：
        1. 逐个取出 s 中的字符。
        2. 在词表内就取它的 ID，不在词表内就取记号的 ID。
        3. 按原顺序收集成列表后 return。
    提示：
        - 字典的 get 方法可以在键不存在时返回一个指定的默认值。
    """
    ans=[]
    for ch in s:
        if ch in stoi:
            ans.append(stoi[ch])
        else:
            ans.append(0)
    return ans


# 以下检查已经写好，不需要修改。
assert encode("小猫吃鱼") == [34, 60, 25, 92], "请完成 TODO 4：词表内的字符逐个查表"
assert encode("小猫吃沙") == [34, 60, 25, 0], "请完成 TODO 4：词表外的字符应映射到记号的 ID"
assert len(encode(val_text)) == len(val_text), (
    "请完成 TODO 4：返回长度必须与输入一致，不能跳过词表外字符"
)


# 以下编码结果和样本数已经写好，不需要修改。
train_ids = encode(train_text)
val_ids = encode(val_text)

# 样本数的手算过程：train_ids 有 166 个 ID。
# 一条样本要切出等长的输入和目标，需要连续 block_size + 1 = 9 个 ID（理由见 TODO 5），
# 而相邻两条样本的起点相差 block_size = 8 个 ID。
# 能切的条数是 (166 - 1) // 8 = 20，编号 0～19。第 19 条用到下标 152～160，
# 剩下的下标 161～165 共 5 个 ID 凑不满一条，本轮直接丢弃。
# 真实训练里这就是 dataloader 的 drop_last：丢掉尾部不完整的样本或批次。
n_samples = (len(train_ids) - 1) // block_size


def get_sample(ids: list[int], k: int) -> tuple[list[int], list[int]]:
    """TODO 5：从 ids 中取出第 k 条定长样本，返回输入和目标。

    上一个练习在整段文本上做过一次位移；这里把同一件事做在长度为 block_size 的片段上，
    并且要能按编号取出任意一条。

    参数：
        ids：一整段连续文本的 ID 列表。
        k：样本编号，从 0 开始；第 k 条的起点是 k * block_size。
    返回：
        (x, y) 两个列表，长度都是 block_size，y 是 x 向右移一位的结果。
    示例 1：
        输入：ids = train_ids，k = 0
        输出：x = [34, 60, 13, 28, 54, 83, 61, 48]，y = [60, 13, 28, 54, 83, 61, 48, 36]
        解释：取下标 0～8 共 9 个 ID，x 是前 8 个，y 是后 8 个。解码后 x 为
            "小猫住在河边的木"，y 为 "猫住在河边的木屋"，也就是每个位置要预测的下一个字符。
    约束：
        - 0 <= k < n_samples，本轮不需要检查越界。
        - 使用传入的 ids，不要写死 train_ids；验证集要能用同一个函数。
        - 只做切片，不要重新编号。
    思路：
        1. 算出起点，从 ids 中取连续 block_size + 1 个 ID。
           为什么多取一个：x 和 y 各占 block_size 个位置，但两者错开一位，
           合起来只覆盖 block_size + 1 个 ID，最后那个 ID 只作为目标出现。
        2. 去掉这段的最后一个 ID，得到 x。
        3. 去掉这段的第一个 ID，得到 y。
        4. return 这两个列表。
    提示：
        - 一个 return 语句可以用逗号同时返回两个值。
    """
    return ids[k*block_size:k*block_size+block_size],ids[1+k*block_size:k*block_size+block_size+1]


# 以下检查已经写好，不需要修改。
sample = get_sample(train_ids, 0)
assert sample is not None, "请完成 TODO 5：get_sample 应返回输入和目标两个列表"
x, y = sample
assert len(x) == len(y) == block_size, "请完成 TODO 5：x 和 y 的长度都应为 block_size"
assert x == [34, 60, 13, 28, 54, 83, 61, 48], "请完成 TODO 5：第 0 条样本从 ids 的开头切起"
assert x[1:] == y[:-1], "请完成 TODO 5：y 应是 x 向右移一位的结果"
assert get_sample(train_ids, 1)[0] == [36, 88, 1, 51, 30, 57, 46, 93], (
    "请完成 TODO 5：相邻两条样本的起点应相差 block_size"
)


def get_batch(ids: list[int], k0: int) -> tuple[list[list[int]], list[list[int]]]:
    """TODO 6：取出从第 k0 条开始、连续 batch_size 条样本，堆成一批。

    模型一次前向处理一整批，而不是一条。这一步把若干条样本按行摆在一起，
    得到 batch_size 行、block_size 列的形状，也就是 M1-2 里模型输入的形状。

    参数：
        ids：一整段连续文本的 ID 列表。
        k0：这一批第一条样本的编号。
    返回：
        (xb, yb) 两个嵌套列表，各有 batch_size 行，每行 block_size 个 ID。
        名字里的 b 是 batch 的缩写。
    示例 1：
        输入：ids = train_ids，k0 = 0
        输出：xb 为 4 行 8 列，第 0 行是 [34, 60, 13, 28, 54, 83, 61, 48]。
        解释：这一批由第 0、1、2、3 条样本组成，每条都由 TODO 5 的 get_sample 取得。
    约束：
        - 0 <= k0 且 k0 + batch_size <= n_samples，本轮不需要检查越界。
        - 复用 get_sample，不要重写切片逻辑。
        - 按编号顺序连续取样，不随机打乱；随机起点采样留到 M1-5。
    思路：
        1. 准备两个空列表，分别存放输入行和目标行。
        2. 依次取编号 k0 到 k0 + batch_size - 1 的样本。
        3. 把每条样本的 x 追加到输入列表，y 追加到目标列表。
        4. return 这两个嵌套列表。
    提示：
        - range(起点, 终点) 可以生成一段连续的编号。
    """
    xb1=[]
    yb1=[]
    for i in range(batch_size):
        xbb,ybb=get_sample(ids,k0+i)
        xb1.append(xbb)
        yb1.append(ybb)
    return xb1,yb1


# 以下检查已经写好，不需要修改。
batch = get_batch(train_ids, 0)
assert batch is not None, "请完成 TODO 6：get_batch 应返回输入和目标两批"
xb, yb = batch
assert len(xb) == len(yb) == batch_size, "请完成 TODO 6：一批应有 batch_size 行"
assert all(len(row) == block_size for row in xb), "请完成 TODO 6：输入每行应有 block_size 个 ID"
assert all(len(row) == block_size for row in yb), "请完成 TODO 6：目标每行应有 block_size 个 ID"
assert xb[0] == x and xb[3] == get_sample(train_ids, 3)[0], (
    "请完成 TODO 6：应按编号连续取第 k0 起的 batch_size 条样本"
)


# 以下打印区已经写好，不需要修改。
# 尾部凑不满一条样本、本轮丢弃的 ID 数：166 - (19 * 8 + 9) = 5。
n_dropped = len(train_ids) - ((n_samples - 1) * block_size + block_size + 1)

print("原文长度：", len(text))
print("训练集长度：", len(train_text), "验证集长度：", len(val_text))
print("验证集原文：", val_text)
print("词表大小：", len(vocab))
print("词表前 8 项：", vocab[:8])
print("验证集里的词表外字符：", unknown_chars(val_text))
print("验证集编码再解码：", decode(val_ids))
print(f"样本数：{n_samples}，每条 {block_size} 个位置，尾部丢弃 {n_dropped} 个 ID")
print("第 0 条样本的输入：", decode(x))
print("第 0 条样本的目标：", decode(y))
print(f"第 0 批的形状：{len(xb)} 行 {len(xb[0])} 列")
print("第 0 批各行的输入文本：")
for row in xb:
    print("   ", decode(row))

print("训练/验证划分与批次构造检查通过")


# 本轮的边界：只做了数据侧的准备。没有构造张量、没有模型、没有 loss、没有反向传播，
# 也没有实现 causal mask，这些属于 M1-2 和 M1-3。
# 定长切分下每条样本长度一致，因此不存在 padding；padding mask 与有效 token 数留到 M1-3。
# 本轮按编号顺序连续取批，没有随机打乱起点，也没有检查采样分布，随机采样留到 M1-5。
# 词表外字符只做了统计并映射到一个记号，没有实现 BPE 等子词切分；
# 按 PLAN.md，BPE 是 M1-1 的可选项，深入的自训练 tokenizer 属于 X1。
