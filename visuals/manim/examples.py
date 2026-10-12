"""从练习源码读取已给定的示例数据，不导入或执行学习者的练习。"""

import ast
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def literal(filename: str, name: str) -> object:
    """仅接受字面量或 torch.tensor 的字面量参数，避免执行练习文件。"""
    tree = ast.parse((ROOT / "scripts" / filename).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "tensor":
                value = value.args[0]
            return ast.literal_eval(value)
    raise ValueError(f"{filename} 中没有找到字面量 {name}")


TEXT = str(literal("01_inspect_tokens.py", "text"))
VOCAB = sorted(set(TEXT))
IDS = [VOCAB.index(ch) for ch in TEXT]
CORPUS = str(literal("02_split_dataset.py", "text"))
CUT = int(len(CORPUS) * float(literal("02_split_dataset.py", "train_ratio")))
TRAIN = CORPUS[:CUT]
VALIDATION = CORPUS[CUT:]
TRAIN_VOCAB = ["<unk>", *sorted(set(TRAIN))]
UNKNOWN = sorted(set(VALIDATION) - set(TRAIN))
FORWARD_VOCAB = literal("03_inspect_forward.py", "vocab")
X = np.array(literal("03_inspect_forward.py", "xb"))
Y = np.array(literal("03_inspect_forward.py", "yb"))
E = np.array(literal("03_inspect_forward.py", "E"))
W = np.array(literal("03_inspect_forward.py", "W"))
H = E[X]
LOGITS = H @ W
PRED = LOGITS.argmax(axis=-1)
Q = np.array(literal("04_inspect_attention.py", "Q"))
K = np.array(literal("04_inspect_attention.py", "K"))
V = np.array(literal("04_inspect_attention.py", "V"))
SCORES = Q @ K.T / np.sqrt(Q.shape[-1])
MASK = np.triu(np.ones_like(SCORES, dtype=bool), k=1)
MASKED = np.where(MASK, -np.inf, SCORES)
EXP = np.exp(MASKED - MASKED.max(axis=-1, keepdims=True))
WEIGHTS = EXP / EXP.sum(axis=-1, keepdims=True)
OUT = WEIGHTS @ V


def check_examples() -> None:
    """分镜依赖小例子的具体规模；源码变化时要求同步调整分镜。"""
    assert TEXT == "小猫吃鱼。" and IDS == [2, 3, 1, 4, 0]
    assert (len(CORPUS), CUT, len(TRAIN_VOCAB)) == (185, 166, 95)
    assert UNKNOWN == ["层", "早", "果", "沙", "留", "细"]
    assert TRAIN_VOCAB.index("小") == 34 and TRAIN_VOCAB.index("猫") == 60
    assert X.shape == (2, 3) and E.shape == (4, 2) and W.shape == (2, 4)
    np.testing.assert_allclose(LOGITS[0, 0], [0, 1, 2, -1])
    np.testing.assert_array_equal(PRED, [[2, 3, 2], [0, 2, 3]])
    assert Q.shape == K.shape == V.shape == (3, 2)
    np.testing.assert_allclose(WEIGHTS.sum(axis=-1), 1)
    np.testing.assert_allclose(OUT[1], [0.33023845, 0.66976155])
    changed = V.copy()
    changed[2] = 100
    np.testing.assert_allclose((WEIGHTS @ changed)[:2], OUT[:2])


check_examples()
