"""M1 第一个练习：字符编码，以及“预测下一个字符”的输入和目标。

你要完成的是三份数据、两个函数，以及两份训练序列。
请按 TODO 1～6 的顺序填写。打印和检查部分已经写好。

运行方式（在项目根目录）：
    uv run --locked python scripts/inspect_tokens.py

当前文件是练习骨架，未填写时会主动报错提示。这是预期行为。
本轮使用普通 Python，不需要 PyTorch，也不会执行模型训练。
"""


# 原始文本。本轮规定一个字符就是一个 token，标点也算。
# 本例有五个不同的字符，因此词表应有五项。
text = "小猫吃鱼。"


# TODO 1：填写词表 vocab，这是一个列表，不是函数。
# 目标：包含 text 中所有不同的字符，每个字符只出现一次。
# 提示：set(text) 可以去重；sorted(...) 可以把字符按固定顺序排列。
# 排序是为了让编号顺序稳定，不代表字符的重要性。
vocab = []


# TODO 2：填写字符到 ID 的字典 stoi，这也是数据，不是函数。
# stoi 是 string-to-integer 的缩写。
# ID 从 0 开始，按 vocab 中的顺序编号。
# 举个与本例无关的例子：若词表为 ["a", "b"]，结果为 {"a": 0, "b": 1}。
# 提示：enumerate(vocab) 每次给你一对“编号、字符”。
# 可以用普通 for 循环逐项填入字典，不要求使用字典推导式。
stoi = {}


# TODO 3：填写反向字典 itos，即 integer-to-string。
# 它回答“给定一个 ID，对应哪个字符？”
# 上面的英文字母例子中，结果为 {0: "a", 1: "b"}。
# 它必须与 stoi 一一对应。
itos = {}


def encode(s: str) -> list[int]:
    """TODO 4：将字符串 s 转换为字符 ID 列表，并返回这个列表。

    str 和 list[int] 是类型提示：输入是字符串，输出是整数列表。
    本函数可以直接使用上面建立的 stoi 字典。

    思路：
    1. 准备一个空列表。
    2. 依次取出 s 中的每个字符，在 stoi 中查找它的 ID。
    3. 将 ID 加入列表，最后 return 该列表。

    注意使用传入的 s，而不是写死成全局变量 text。
    本轮假设输入字符都在词表中，不需要处理未知字符。
    """
    # 将下面的占位报错替换成你的实现。
    raise NotImplementedError("请先完成 TODO 4：encode")


def decode(ids: list[int]) -> str:
    """TODO 5：将 ID 列表还原成字符串，并返回字符串。

    本函数可以直接使用上面建立的 itos 字典。

    思路：
    1. 遍历传入的 ids，将每个 ID 查成对应字符。
    2. 按原顺序拼接这些字符。
    3. return 拼接结果；返回值是字符串，不是字符列表。

    提示：可以使用空字符串的 join 方法拼接字符列表。
    同样，不要把答案写死为全局变量 text。
    """
    # 将下面的占位报错替换成你的实现。
    raise NotImplementedError("请先完成 TODO 5：decode")


# 以下先检查前三份数据。assert 后的条件不成立时，程序会在这里停止。
# 这些提示能让你逐项填写，不会一下子跳到后面尚未完成的函数。
assert vocab == sorted(set(text)), "请完成 TODO 1：去重并排序的词表"
assert set(stoi) == set(vocab), "请完成 TODO 2：为每个字符建立 ID"
assert [stoi[ch] for ch in vocab] == list(range(len(vocab))), "ID 应按词表顺序从 0 编号"
assert all(itos.get(stoi[ch]) == ch for ch in vocab), "请完成 TODO 3：建立反向字典"

print("词表：", vocab)
print("字符到 ID：", stoi)
print("ID 到字符：", itos)


# 调用你写的两个函数，检查编码和解码能否对应。
ids = encode(text)
decoded_text = decode(ids)

assert len(ids) == len(text), "本例一个字符应对应一个 ID"
assert decoded_text == text, "编码再解码，应还原出原文"
assert decode(encode("鱼小猫")) == "鱼小猫", "函数也要能处理词表内的另一段文本"

print("完整 ID 序列：", ids)
print("解码后的文本：", decoded_text)


# TODO 6：从 ids 中构造两份等长的列表。
# x 是输入：去掉完整序列的最后一个 ID。
# y 是目标：去掉完整序列的第一个 ID。
# 提示：使用列表切片；不要重新编号，也不要手写具体 ID。
# 本例解码后应分别是：x -> "小猫吃鱼"，y -> "猫吃鱼。"。
x = []
y = []

assert len(x) == len(y) == len(ids) - 1, "请完成 TODO 6：输入和目标都应比原序列短一项"
assert decode(x) == text[:-1], "输入应去掉原文最后一个字符"
assert decode(y) == text[1:], "目标应去掉原文第一个字符"

print("输入文本：", decode(x))
print("目标文本：", decode(y))


# 展示逻辑已经写好。这里展示的是未来模型在各位置应使用的前缀，
# 并没有实际运行模型，也没有实现阻止模型看到后文的 causal mask。
print("各位置的预测任务：")
for i in range(len(x)):
    # 切片右端不包含在结果中，所以 i+1 才能包含位置 i。
    prefix = decode(x[: i + 1])
    # decode 接收列表；y[i] 是一个整数，因此先用 [] 包成单元素列表。
    target = decode([y[i]])
    print(f"位置 {i}：{prefix} -> {target}")

print("字符编码与输入/目标构造检查通过")
