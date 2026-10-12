"""每个 Scene 对应一个已完成的小节；输入来自源码，不执行学习练习。"""

import numpy as np
from manim import DOWN, LEFT, Create, FadeIn, FadeOut, Indicate, ReplacementTransform, Transform, VGroup

from drawing import BLUE, GOLD, GREEN, MUTED, Grid, LabScene, copy_to, focus, label, number, text
from examples import TEXT, VOCAB, IDS, CORPUS, CUT, TRAIN_VOCAB, X, Y, E, W, H, LOGITS, PRED, Q, K, V, SCORES, MASKED, WEIGHTS, OUT


class TokenVocabularyScene(LabScene):
    def construct(self) -> None:
        self.start("建立词表与双向字典", "01 · vocab / stoi / itos")
        source = Grid(list(TEXT), width=1.05).move_to([0, 1.4, 0])
        target = Grid(VOCAB, width=1.05).move_to([0, -0.3, 0])
        ids = Grid(list(range(len(VOCAB))), width=1.05, color=GOLD).move_to([0, -1.35, 0])
        self.step("原文按顺序是“小猫吃鱼。”，本例一个字符就是一个 token。", FadeIn(source), FadeIn(label("原始文本", source)))
        self.step("去重，再按字符升序排列；原文本身保持原来的顺序。", FadeIn(target.boxes()), FadeIn(label("vocab", target)))
        self.play(*[copy_to(source.cell(0, TEXT.index(ch))[1], target.cell(0, i)[1]) for i, ch in enumerate(VOCAB)], run_time=1.6)
        self.step("按位置从 0 编号。“小”的 ID 是 2，而不是第一个编号。", FadeIn(ids), FadeIn(label("ID", ids, direction=LEFT)), Indicate(target.cell(0, 2), color=GREEN))
        self.step("stoi 查字符到 ID；itos 查 ID 到字符，两份字典互为反向。", FadeIn(self.note('stoi["小"] = 2     ↔     itos[2] = "小"', y=-2.35)), hold=3)


class TokenEncodeScene(LabScene):
    def construct(self) -> None:
        self.start("编码：逐个字符查表", "01 · encode(s)")
        source = Grid(list(TEXT), width=1.02).move_to([0, 1.55, 0])
        mapping = Grid([VOCAB, list(range(5))], width=1.02).move_to([0, 0, 0])
        result = Grid(IDS, width=1.02, color=GREEN).move_to([0, -1.6, 0])
        self.step("遍历原文，用同一份 stoi 字典查出每个字符的 ID。", FadeIn(source), FadeIn(mapping), FadeIn(result.boxes()), hold=1)
        pointer = focus(source.cell(0, 0), BLUE)
        lookup = focus(mapping.rows[0][IDS[0]], GOLD)
        self.add(pointer, lookup)
        for i, ch in enumerate(TEXT):
            self.step(f"第 {i} 项：{ch} → ID {IDS[i]}，追加到输出的第 {i} 项。", Transform(pointer, focus(source.cell(0, i), BLUE)), Transform(lookup, focus(VGroup(mapping.cell(0, IDS[i]), mapping.cell(1, IDS[i])), GOLD)), copy_to(mapping.cell(1, IDS[i])[1], result.cell(0, i)[1]), hold=1.1)
        self.step("输出完整序列 [2, 3, 1, 4, 0]，字符顺序和序列长度都保留。", FadeOut(pointer), FadeOut(lookup), hold=2.5)


class TokenDecodeScene(LabScene):
    def construct(self) -> None:
        self.start("解码：ID 还原成文本", "01 · decode(ids)")
        source = Grid(IDS, width=1.02, color=BLUE).move_to([0, 1.55, 0])
        mapping = Grid([list(range(5)), VOCAB], width=1.02).move_to([0, 0, 0])
        result = Grid(list(TEXT), width=1.02, color=GREEN).move_to([0, -1.6, 0])
        self.step("解码沿着 itos 反向查表，按传入 ID 的顺序收集字符。", FadeIn(source), FadeIn(mapping), FadeIn(result.boxes()))
        pointer = focus(source.cell(0, 0), BLUE)
        self.add(pointer)
        for i, value in enumerate(IDS):
            self.step(f"ID {value} → “{VOCAB[value]}”，放回第 {i} 个位置。", Transform(pointer, focus(source.cell(0, i), BLUE)), Indicate(mapping.cell(1, value), color=GOLD), copy_to(mapping.cell(1, value)[1], result.cell(0, i)[1]), hold=0.9)
        self.step("最后拼成字符串；编码后再解码，得到原始文本。", FadeOut(pointer), FadeIn(self.note('join → "小猫吃鱼。"', y=-2.4)), hold=2.5)


class TokenShiftScene(LabScene):
    def construct(self) -> None:
        self.start("构造 next-token 输入与目标", "01 · x / y · 目标只位移一次")
        source = Grid(IDS, width=1.0).move_to([0, 1.45, 0])
        x = Grid(IDS[:-1], width=1.0, color=BLUE).move_to([0, 0, 0])
        y = Grid(IDS[1:], width=1.0, color=GOLD).move_to([0, -1.45, 0])
        self.step("完整序列有五个 ID，它将产生四个预测位置。", FadeIn(source), FadeIn(label("ids", source)))
        self.step("输入 x 去掉最后一项，保留前四个 ID。", FadeIn(x.boxes()), FadeIn(label("输入 x", x, BLUE, LEFT)), *[copy_to(source.cell(0, i)[1], x.cell(0, i)[1]) for i in range(4)])
        self.step("目标 y 去掉第一项，再移到与 x 相同的列上。", FadeIn(y.boxes()), FadeIn(label("目标 y", y, GOLD, LEFT)), *[copy_to(source.cell(0, i + 1)[1], y.cell(0, i)[1]) for i in range(4)])
        pointer = focus(VGroup(x.cell(0, 0), y.cell(0, 0)))
        self.add(pointer)
        for i in range(4):
            self.step(f"位置 {i}：输入前缀“{TEXT[:i + 1]}”，预测“{TEXT[i + 1]}”。", Transform(pointer, focus(VGroup(x.cell(0, i), y.cell(0, i)))), hold=1.1)
        self.step("这里准备预测任务；限制模型读取未来，需要后面的因果遮罩。", FadeOut(pointer), hold=2.5)


class BatchSplitScene(LabScene):
    def construct(self) -> None:
        self.start("划分训练集与验证集", "02 · train_text / val_text")
        source = Grid(list(TEXT), width=1.02).move_to([0, 1.35, 0])
        left = Grid(list(TEXT[:3]), width=1.02, color=BLUE).move_to([-2.2, -0.1, 0])
        right = Grid(list(TEXT[3:]), width=1.02, color=GOLD).move_to([2.1, -0.1, 0])
        self.step("先看独立小例子：5 个字符，训练比例为 0.6。", FadeIn(source))
        cut = self.note("cut = floor(5 × 0.6) = 3")
        self.step("切分点是下标 3，前 3 个归训练，剩下 2 个归验证。", FadeIn(cut), FadeIn(left.boxes()), FadeIn(right.boxes()), *[copy_to(source.cell(0, i)[1], left.cell(0, i)[1]) for i in range(3)], *[copy_to(source.cell(0, i + 3)[1], right.cell(0, i)[1]) for i in range(2)])
        self.step("连续切分保留顺序；两段分别建样本，不能跨边界取数据。", FadeIn(label("训练段", left, BLUE, DOWN)), FadeIn(label("验证段", right, GOLD, DOWN)))
        actual = self.note(f"实际语料：{len(CORPUS)} × 0.9 → {CUT} 训练 + {len(CORPUS) - CUT} 验证")
        self.step("回到脚本：185 × 0.9 向下取整为 166，验证段有 19 个字符。", ReplacementTransform(cut, actual), hold=3)


class BatchVocabularyScene(LabScene):
    def construct(self) -> None:
        self.start("只用训练文本建立词表", "02 · vocab · 独立小例子")
        source = Grid(["猫", "小", "猫"], width=1.1).move_to([0, 1.4, 0])
        unique = Grid(["小", "猫"], width=1.1).move_to([0, 0, 0])
        vocab = Grid(["<unk>", "小", "猫"], width=1.55).move_to([0, -1.4, 0])
        self.step("独立训练文本“猫小猫”，同一个字符只需要一个词表位置。", FadeIn(source))
        self.step("去重后按固定顺序排列，两次出现的“猫”合并到同一项。", FadeIn(unique.boxes()), copy_to(source.cell(0, 1)[1], unique.cell(0, 0)[1]), copy_to(source.cell(0, 0)[1], unique.cell(0, 1)[1]), FadeOut(source.cell(0, 2)))
        self.step("把未知字符记号放在第 0 项，其余字符的编号依次后移。", FadeIn(vocab.boxes()), FadeIn(vocab.cell(0, 0)[1]), copy_to(unique.cell(0, 0)[1], vocab.cell(0, 1)[1]), copy_to(unique.cell(0, 1)[1], vocab.cell(0, 2)[1]))
        ids = VGroup(*[text(str(i), 24, GOLD).next_to(vocab.cell(0, i), DOWN, buff=0.12) for i in range(3)])
        self.step("实际语料有 94 种训练字符，加上 <unk> 共 95 项。", FadeIn(ids), hold=2.5)


class BatchUnknownScene(LabScene):
    def construct(self) -> None:
        self.start("找出词表外的字符", "02 · unknown_chars(s)")
        source = Grid(list("沙猫沙层"), width=1.1).move_to([0, 1.4, 0])
        unknown = Grid(list("沙沙层"), width=1.1, color=GOLD).move_to([0, -0.1, 0])
        result = Grid(["层", "沙"], width=1.1, color=GREEN).move_to([0, -1.5, 0])
        self.step("输入“沙猫沙层”，逐个判断字符是否在训练词表中。", FadeIn(source))
        self.step("“猫”能查到；“沙”和“层”查不到，先收集三次未知出现。", FadeIn(unknown.boxes()), copy_to(source.cell(0, 0)[1], unknown.cell(0, 0)[1]), copy_to(source.cell(0, 2)[1], unknown.cell(0, 1)[1]), copy_to(source.cell(0, 3)[1], unknown.cell(0, 2)[1]), source.cell(0, 1).animate.set_opacity(0.3))
        self.step("报告的是未知字符的种类：去重、排序，得到“层、沙”。", FadeIn(result.boxes()), copy_to(unknown.cell(0, 2)[1], result.cell(0, 0)[1]), copy_to(unknown.cell(0, 0)[1], result.cell(0, 1)[1]))
        self.step("实际验证段有 6 种未知字符：层、早、果、沙、留、细。", FadeIn(self.note("验证文本 → 6 种词表外字符", y=-2.35)), hold=2.5)


class BatchEncodeScene(LabScene):
    def construct(self) -> None:
        self.start("未知字符也要保留一个位置", "02 · encode(s) · 使用实际训练词表")
        chars = list("小猫吃沙")
        ids = [TRAIN_VOCAB.index(ch) if ch in TRAIN_VOCAB else 0 for ch in chars]
        source = Grid(chars, width=1.2).move_to([0, 1.3, 0])
        out = Grid(ids, width=1.2, color=GREEN).move_to([0, -0.4, 0])
        self.step("这次词表来自训练段，编号和第一章不同。", FadeIn(source), FadeIn(out.boxes()))
        for i, ch in enumerate(chars):
            caption = f"“{ch}”在词表中 → ID {ids[i]}。" if i < 3 else "“沙”不在训练词表中 → <unk> 的 ID 0。"
            self.step(caption, Indicate(source.cell(0, i), color=BLUE if i < 3 else GOLD), copy_to(source.cell(0, i)[1], out.cell(0, i)[1]), hold=1.2)
        self.step("不能跳过未知字符：四个字符仍对应四个 ID，位置才不会错乱。", FadeIn(self.note("[34, 60, 25, 0]    长度保持为 4")), hold=3)


class BatchWindowScene(LabScene):
    def construct(self) -> None:
        self.start("从连续 ID 取出一条样本", "02 · get_sample(ids, k) · 独立示例 T = 3")
        source = Grid(list(range(10, 20)), width=0.88).move_to([0, 1.35, 0])
        indexes = VGroup(*[text(str(i), 18, MUTED).next_to(source.cell(0, i), [0, 1, 0], buff=0.13) for i in range(10)])
        x = Grid([10, 11, 12], width=1.02, color=BLUE).move_to([-2.35, -0.75, 0])
        y = Grid([11, 12, 13], width=1.02, color=GOLD).move_to([2.35, -0.75, 0])
        pointer = focus(VGroup(*source.rows[0][:4]))
        self.step("上方是下标，下方是示意 token ID；T = 3 需要连续 4 个 ID。", FadeIn(source), FadeIn(indexes))
        self.step("k = 0，从下标 0 取到 3：前 3 项是 x，后 3 项是 y。", Create(pointer), FadeIn(x.boxes()), FadeIn(y.boxes()), FadeIn(label("x", x, BLUE)), FadeIn(label("y", y, GOLD)), *[copy_to(source.cell(0, i)[1], x.cell(0, i)[1]) for i in range(3)], *[copy_to(source.cell(0, i + 1)[1], y.cell(0, i)[1]) for i in range(3)], hold=2.5)
        self.step("k = 1，起点前进 T = 3 个位置，选下标 3 到 6。", Transform(pointer, focus(VGroup(*source.rows[0][3:7]))), FadeOut(x.labels()), FadeOut(y.labels()))
        next_x = Grid([13, 14, 15], width=1.02, color=BLUE).move_to(x)
        next_y = Grid([14, 15, 16], width=1.02, color=GOLD).move_to(y)
        self.step("新的输入为 [13, 14, 15]，目标为 [14, 15, 16]。", *[copy_to(source.cell(0, i + 3)[1], next_x.cell(0, i)[1]) for i in range(3)], *[copy_to(source.cell(0, i + 4)[1], next_y.cell(0, i)[1]) for i in range(3)])
        self.step("ID 13 是上一条最后的目标，也是这一条最先的输入。", Indicate(source.cell(0, 3), color=GOLD), FadeIn(self.note("起点：k × T     片段：T + 1 项")), hold=3)


class BatchStackScene(LabScene):
    def construct(self) -> None:
        self.start("把样本按行放进 batch", "02 · get_batch(ids, k0) · 独立示例 B = 2，T = 3")
        source = Grid(list(range(10, 20)), width=0.88).move_to([0, 1.5, 0])
        xb = Grid([[10, 11, 12], [13, 14, 15]], width=1.0, color=BLUE).move_to([-2.5, -0.3, 0])
        yb = Grid([[11, 12, 13], [14, 15, 16]], width=1.0, color=GOLD).move_to([2.5, -0.3, 0])
        self.step("k0 = 0，连续取第 0、1 条样本，分别成为 batch 的两行。", FadeIn(source), FadeIn(xb.boxes()), FadeIn(yb.boxes()), FadeIn(label("xb [B, T]", xb, BLUE)), FadeIn(label("yb [B, T]", yb, GOLD)))
        for row in range(2):
            self.step(f"样本 {row} → 第 {row} 行；输入与目标按同样的顺序放置。", *[copy_to(source.cell(0, row * 3 + col)[1], xb.cell(row, col)[1]) for col in range(3)], *[copy_to(source.cell(0, row * 3 + col + 1)[1], yb.cell(row, col)[1]) for col in range(3)], hold=2)
        self.step("每行保留自己的上下文，这一批有 2 × 3 = 6 个预测位置。", Create(focus(xb.rows[1], BLUE)), Create(focus(yb.rows[1], GOLD)))
        shifted_x = Grid([[13, 14, 15], [16, 17, 18]], width=1.0, color=BLUE).move_to(xb)
        shifted_y = Grid([[14, 15, 16], [17, 18, 19]], width=1.0, color=GOLD).move_to(yb)
        self.step("k0 = 1 时，改取第 1、2 条样本；两边一起移动。", Transform(xb, shifted_x), Transform(yb, shifted_y), hold=2.5)
        self.step("实际训练段有 20 条完整样本；验证段只有 2 条，不足一整批。", FadeIn(self.note("实际 B = 4，T = 8 → xb、yb 都是 [4, 8]", y=-1.95)), hold=3)


class ForwardTensorScene(LabScene):
    def construct(self) -> None:
        self.start("列表变成整数张量", "03 · to_tensor(rows) · B = 2，T = 3")
        left_x = Grid(X, width=0.85, color=BLUE).move_to([-3, 1.0, 0])
        left_y = Grid(Y, width=0.85, color=GOLD).move_to([-3, -1.15, 0])
        right_x = Grid(X, width=0.85, color=BLUE).move_to([3, 1.0, 0])
        right_y = Grid(Y, width=0.85, color=GOLD).move_to([3, -1.15, 0])
        self.step("输入列表 xb 与目标列表 yb 已经对齐，转换时保持每一个位置。", FadeIn(left_x), FadeIn(left_y), FadeIn(label("xb · 列表", left_x, BLUE)), FadeIn(label("yb · 列表", left_y, GOLD)))
        self.step("把两层列表变成二维张量，六个 ID 的数值与顺序完全保留。", FadeIn(right_x.boxes()), FadeIn(right_y.boxes()), FadeIn(label("x · 张量", right_x, BLUE)), FadeIn(label("y · 张量", right_y, GOLD)), *[copy_to(a, b) for a, b in zip(left_x.labels(), right_x.labels())], *[copy_to(a, b) for a, b in zip(left_y.labels(), right_y.labels())], run_time=1.8, hold=2)
        self.step("shape = [2, 3]，dtype = torch.long；ID 是整数查表下标。", FadeIn(self.note("[B, T] = [2, 3]     dtype: int64", y=-2.25)))
        self.step("目标已经对应下一个 token，这里不能再移动一次。", Indicate(right_y, color=GOLD), hold=3)


class ForwardEmbeddingScene(LabScene):
    def construct(self) -> None:
        self.start("查表：每个 ID 变成一个向量", "03 · lookup_embeddings(ids, table)")
        x = Grid(X, width=0.77, color=BLUE).move_to([-4.5, 0.4, 0])
        table = Grid(E, width=0.65, height=0.66).move_to([-1.35, 0.4, 0])
        entries = [[f"[{number(v[0])}, {number(v[1])}]" for v in row] for row in H]
        out = Grid(entries, width=1.46, size=26, color=GREEN).move_to([3.5, 0.4, 0])
        row_ids = VGroup(*[text(str(i), 20, GOLD).next_to(table.rows[i], LEFT, buff=0.17) for i in range(4)])
        self.step("E 有 4 行、2 列；每个 ID 选择对应的一整行。", FadeIn(x), FadeIn(table), FadeIn(row_ids), FadeIn(out.boxes()), FadeIn(label("x [2, 3]", x, BLUE)), FadeIn(label("E [4, 2]", table)), FadeIn(label("h [2, 3, 2]", out, GREEN)))
        selector = focus(table.rows[1], GOLD)
        position = focus(x.cell(0, 0), BLUE)
        self.add(selector, position)
        for b in range(2):
            for t in range(3):
                token = int(X[b, t])
                self.step(f"x[{b}, {t}] = {token} → E 第 {token} 行 → {entries[b][t]}。", Transform(selector, focus(table.rows[token], GOLD)), Transform(position, focus(x.cell(b, t), BLUE)), copy_to(table.rows[token], out.cell(b, t)[1]), hold=1.0)
        self.step("B、T 两维保持不变，每个位置增加 D = 2 个分量。", FadeOut(selector), FadeOut(position), FadeIn(self.note("[B, T] → [B, T, D]")), hold=3)


class ForwardLogitsScene(LabScene):
    def construct(self) -> None:
        self.start("投影：逐列计算候选分数", "03 · project_logits(hidden, weight)")
        h = Grid(H[0, 0], width=0.8, color=BLUE).move_to([-4.9, 0.8, 0])
        weight = Grid(W, width=0.83).move_to([-0.7, 0.8, 0])
        output = Grid(LOGITS[0, 0], width=0.83, color=GREEN).move_to([4.1, 0.8, 0])
        headers = VGroup(label("h [D]", h, BLUE), label("W [D, V]", weight), label("logits [V]", output, GREEN))
        self.step("取输入“小”的向量 [1, 0]；W 的每一列对应一个候选 token。", FadeIn(h), FadeIn(weight), FadeIn(output.boxes()), FadeIn(headers))
        formula = self.note("1 × 0 + 0 × 0 = 0", y=-1.15)
        pointer = focus(VGroup(weight.cell(0, 0), weight.cell(1, 0)), GOLD)
        self.add(formula, pointer)
        for col in range(4):
            expression = self.note(f"1 × ({number(W[0, col])}) + 0 × ({number(W[1, col])}) = {number(LOGITS[0, 0, col])}", y=-1.15)
            self.step(f"第 {col} 列：对应分量相乘，再把两个乘积相加。", Transform(pointer, focus(VGroup(weight.cell(0, col), weight.cell(1, col)), GOLD)), Transform(formula, expression), FadeIn(output.cell(0, col)[1]), hold=1.8)
        self.step("得到 [0, 1, 2, −1]。logits 是分数，可以为负，还不是概率。", FadeOut(pointer), hold=2.5)
        all_logits = Grid([[str([int(v) for v in vec]) for vec in row] for row in LOGITS], width=3.4, height=0.76, size=24, color=GREEN).move_to([0, 0.3, 0])
        self.step("对 batch 里全部六个位置执行同样计算，得到 [2, 3, 4]。", FadeOut(h), FadeOut(weight), FadeOut(output), FadeOut(formula), FadeOut(headers), FadeIn(all_logits), FadeIn(label("完整 logits [2, 3, 4]", all_logits)), hold=3)


class ForwardPredictScene(LabScene):
    def construct(self) -> None:
        self.start("从分数中选出预测 ID", "03 · predict_ids(scores) · argmax 沿最后一维")
        score = Grid(LOGITS[0, 0], width=1.0).move_to([0, 1.25, 0])
        ids = VGroup(*[text(str(i), 22, GOLD).next_to(score.cell(0, i), [0, 1, 0], buff=0.18) for i in range(4)])
        self.step("四个候选分数为 [0, 1, 2, −1]，上方标的是候选 ID。", FadeIn(score), FadeIn(ids))
        chosen = text("ID 2 → 猫", 36, GREEN).move_to([0, -0.15, 0])
        self.step("最大分数是 2，返回它所在的位置 ID 2，而不是返回分数。", Create(focus(score.cell(0, 2))), copy_to(ids[2], chosen))
        result = Grid(PRED, width=1.0, color=GREEN).move_to([0, -1.55, 0])
        self.step("每个位置选一次：完整预测张量变回 [B, T] = [2, 3]。", FadeIn(result))
        self.step("全零的那一行有并列最大值，argmax 选择第一个候选，即 ID 0。", Indicate(result.cell(1, 0), color=GOLD), hold=2.5)
        self.step("这是固定权重的计算结果，还没有发生训练或参数更新。", hold=3)


class AttentionScoresScene(LabScene):
    def construct(self) -> None:
        self.start("计算缩放点积分数", "04 · attention_scores(q, k) · T = 3，D = 2")
        q = Grid(Q, width=0.75, color=BLUE).move_to([-4.7, 0.5, 0])
        kt = Grid(K.T, width=0.75, color=GOLD).move_to([-1.3, 0.5, 0])
        scores = Grid(SCORES, width=1.08, size=28, color=GREEN).move_to([3.3, 0.5, 0])
        self.step("Q 的每一行，与 K 转置后的每一列做点积，再除以 √2。", FadeIn(q), FadeIn(kt), FadeIn(scores.boxes()), FadeIn(label("Q [3, 2]", q, BLUE)), FadeIn(label("Kᵀ [2, 3]", kt, GOLD)), FadeIn(label("scores [3, 3]", scores, GREEN)))
        row_focus = focus(q.rows[0], BLUE)
        col_focus = focus(VGroup(kt.cell(0, 0), kt.cell(1, 0)), GOLD)
        self.add(row_focus, col_focus)
        equation = self.note("(1 × 1 + 0 × 0) / √2 = 0.707", y=-1.8)
        self.add(equation)
        for row in range(3):
            for col in range(3):
                expression = self.note(f"({number(Q[row, 0])} × {number(K[col, 0])} + {number(Q[row, 1])} × {number(K[col, 1])}) / √2 = {number(SCORES[row, col])}", y=-1.8)
                self.step(f"Q 的第 {row} 行 × K 的第 {col} 行 → scores[{row}, {col}]。", Transform(row_focus, focus(q.rows[row], BLUE)), Transform(col_focus, focus(VGroup(kt.cell(0, col), kt.cell(1, col)), GOLD)), Transform(equation, expression), FadeIn(scores.cell(row, col)[1]), hold=0.8)
        self.step("行是查询位置，列是被读取的位置；此时还没有遮住未来。", FadeOut(row_focus), FadeOut(col_focus), hold=3)


class AttentionMaskScene(LabScene):
    def construct(self) -> None:
        self.start("遮住未来位置", "04 · apply_causal_mask(scores)")
        scores = Grid(SCORES, width=1.12).move_to([-3.0, 0.4, 0])
        output = Grid(MASKED, width=1.12).move_to([3.0, 0.4, 0])
        self.step("分数矩阵有三个查询位置；第 i 行只能读取第 0 到 i 列。", FadeIn(scores), FadeIn(label("scores", scores)), FadeIn(output.boxes()), FadeIn(label("masked_scores", output)))
        for row in range(3):
            animations = [copy_to(scores.cell(row, col)[1], output.cell(row, col)[1]) for col in range(3)]
            for col in range(row + 1, 3):
                output.cell(row, col)[1].set_color(GOLD)
                animations.append(output.cell(row, col)[0].animate.set_fill("#34404A"))
            self.step(f"位置 {row}：保留列 0～{row}，把右侧所有未来分数设为 −∞。" if row < 2 else "位置 2：三个位置都已出现，所以这一行全部保留。", *animations, hold=2.2)
        self.step("屏蔽的是不含对角线的上三角；矩阵形状和可见分数不变。", FadeIn(self.note("j > i → −∞     j ≤ i → 原分数")), hold=2.5)
        self.step("不能替换成 0：exp(0) = 1，仍会在 softmax 中获得权重。", hold=3)


class AttentionSoftmaxScene(LabScene):
    def construct(self) -> None:
        self.start("把一行分数变成读取权重", "04 · attention_weights(masked_scores) · softmax(dim = −1)")
        row = Grid(MASKED[1], width=1.55).move_to([0, 1.6, 0])
        exps = Grid(np.exp(MASKED[1]), width=1.55, color=GOLD).move_to([0, 0.1, 0])
        weights = Grid(WEIGHTS[1], width=1.55, color=GREEN).move_to([0, -1.5, 0])
        self.step("先看位置 1：遮罩后的分数是 [0, 0.707, −∞]。", FadeIn(row))
        self.step("每项取指数，得到 [1, 2.028, 0]；未来位置的贡献变成 0。", FadeIn(exps.boxes()), *[copy_to(row.cell(0, i)[1], exps.cell(0, i)[1]) for i in range(3)], hold=2.5)
        formula = self.note("分母 = 1 + 2.028 + 0 = 3.028", y=-0.65)
        self.step("只把同一行的指数相加，作为这一行自己的分母。", FadeIn(formula))
        self.step("逐项除以 3.028，得到 [0.330, 0.670, 0]，这一行和为 1。", FadeIn(weights.boxes()), *[copy_to(exps.cell(0, i)[1], weights.cell(0, i)[1]) for i in range(3)], hold=2.5)
        all_rows = Grid(WEIGHTS, width=1.65, color=GREEN).move_to([0, 0.35, 0])
        self.step("其他行各自归一化，不跨行混合；未来位置的权重始终为 0。", FadeOut(row), FadeOut(exps), FadeOut(weights), FadeOut(formula), FadeIn(all_rows), FadeIn(label("完整 weights [3, 3] · 数值四舍五入", all_rows)), hold=3)


class AttentionValuesScene(LabScene):
    def construct(self) -> None:
        self.start("用权重汇总 V 向量", "04 · attend_values(weights, values)")
        weights = Grid(WEIGHTS[1], width=1.28, color=BLUE).move_to([-3.2, 1.2, 0])
        values = Grid(V, width=1.0, color=GOLD).move_to([3.1, 0.5, 0])
        parts = Grid(WEIGHTS[1, :, None] * V, width=1.2, color=GREEN).move_to([-2.9, -0.55, 0])
        headers = VGroup(label("位置 1 的权重", weights, BLUE), label("V [3, 2]", values, GOLD))
        self.step("位置 1 的三个权重，分别乘 V 的三行向量。", FadeIn(weights), FadeIn(values), FadeIn(parts.boxes()), FadeIn(headers))
        for i in range(3):
            contribution = WEIGHTS[1, i] * V[i]
            self.step(f"{number(WEIGHTS[1, i])} × [{number(V[i, 0])}, {number(V[i, 1])}] → [{number(contribution[0])}, {number(contribution[1])}]。", Indicate(weights.cell(0, i), color=BLUE), Indicate(values.rows[i], color=GOLD), *[copy_to(values.cell(i, j)[1], parts.cell(i, j)[1]) for j in range(2)], hold=1.8)
        sum_note = self.note("[0.330, 0] + [0, 0.670] + [0, 0]", y=-2.1)
        self.step("沿着向量分量相加：位置 1 的输出是 [0.330, 0.670]。", FadeIn(sum_note), hold=3)
        result = Grid(OUT, width=1.6, color=GREEN).move_to([0, 0.3, 0])
        self.step("每个位置都这样汇总，完整输出为 [T, D] = [3, 2]。", FadeOut(weights), FadeOut(values), FadeOut(parts), FadeOut(headers), FadeOut(sum_note), FadeIn(result), FadeIn(label("output [3, 2]", result, GREEN)), hold=3)


class AttentionCausalityScene(LabScene):
    def construct(self) -> None:
        self.start("验证未来 V 不影响前面的位置", "04 · 因果性自检 · Q、K 保持不变")
        values = Grid(V, width=1.05, color=GOLD).move_to([-3.4, 0.4, 0])
        out = Grid(OUT, width=1.6, color=GREEN).move_to([3.1, 0.4, 0])
        self.step("先保留原始 V 与完整输出，对照每个位置的变化。", FadeIn(values), FadeIn(out), FadeIn(label("V [3, 2]", values, GOLD)), FadeIn(label("output [3, 2]", out, GREEN)))
        changed = V.copy()
        changed[2] = 100
        new_values = Grid(changed, width=1.05, color=GOLD).move_to(values)
        new_out = Grid(WEIGHTS @ changed, width=1.6, color=GREEN).move_to(out)
        self.step("只把位置 2 的 V 从 [2, 2] 改成 [100, 100]。", Transform(values, new_values), Create(focus(values.rows[2], GOLD)), hold=2.5)
        self.step("Q、K 没变，权重也不变；重新汇总后，前两行输出完全相同。", Transform(out, new_out), Create(focus(VGroup(out.rows[0], out.rows[1]), BLUE)), hold=2.5)
        self.step("位置 2 可以读取自己，所以最后一行变为约 [50.597, 50.597]。", Indicate(out.rows[2], color=GOLD), hold=2.5)
        self.step("因果遮罩阻断了未来 V 到前面位置的读取路径。", FadeIn(self.note("未来变化 → 前两行不变")), hold=3)
