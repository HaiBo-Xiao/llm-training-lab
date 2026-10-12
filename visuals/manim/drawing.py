"""Manim 场景共享的数组元素、字幕时间轴和中文排版。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import manimpango
import numpy as np
from manim import (
    LEFT, RIGHT, DOWN, UP, FadeIn, Line, Rectangle, ReplacementTransform,
    Scene, SurroundingRectangle, Text, VGroup, config,
)

BG = "#101D26"
PAPER = "#182A35"
INK = "#EDF2F0"
MUTED = "#ADC0CA"
LINE = "#405461"
GREEN = "#83D6B5"
BLUE = "#8DBBFA"
GOLD = "#F3C87F"
DIM = "#6E808B"


def choose_font(variable: str, candidates: list[str]) -> str:
    installed = set(manimpango.list_fonts())
    preferred = os.environ.get(variable)
    if preferred:
        if preferred not in installed:
            raise RuntimeError(f"字体未安装：{preferred}")
        return preferred
    for candidate in candidates:
        if candidate in installed:
            return candidate
    raise RuntimeError(f"请安装字体 {candidates[0]}，或设置 {variable}")


FONT = choose_font("LLM_LAB_MANIM_FONT", ["PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei"])
MONO = choose_font("LLM_LAB_MANIM_MONO_FONT", ["Menlo", "DejaVu Sans Mono", "Monospace"])


def text(value: object, size: int = 28, color: str = INK, mono: bool = False, width: float = 12.6) -> Text:
    result = Text(str(value), font=MONO if mono else FONT, font_size=size, color=color)
    if result.width > width:
        result.scale_to_fit_width(width)
    return result


def number(value: object) -> str:
    if isinstance(value, (str, np.str_)):
        return str(value)
    if np.isneginf(value):
        return "−∞"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3f}"


class Grid(VGroup):
    """每个格子分别保留边框和标签，供元素移动与高亮使用。"""

    def __init__(self, values: object, width: float = 0.9, height: float = 0.66, size: int = 30, color: str = INK) -> None:
        super().__init__()
        values = np.asarray(values, dtype=object)
        if values.ndim == 1:
            values = values[None, :]
        self.rows = VGroup()
        for row in values:
            cells = VGroup()
            for value in row:
                box = Rectangle(width=width, height=height, stroke_width=1, stroke_color=LINE, fill_color=PAPER, fill_opacity=1)
                item = text(number(value), size=size, color=color, width=width - 0.14).move_to(box).set_z_index(2)
                cells.add(VGroup(box, item))
            cells.arrange(RIGHT, buff=0.06)
            self.rows.add(cells)
        self.rows.arrange(DOWN, buff=0.07)
        self.add(self.rows)

    def cell(self, row: int, column: int) -> VGroup:
        return self.rows[row][column]

    def labels(self) -> VGroup:
        return VGroup(*[cell[1] for row in self.rows for cell in row])

    def boxes(self) -> VGroup:
        return VGroup(*[cell[0] for row in self.rows for cell in row])


def label(value: str, mob: VGroup, color: str = MUTED, direction: np.ndarray = UP) -> Text:
    return text(value, 23, color).next_to(mob, direction, buff=0.22)


def focus(mob: VGroup, color: str = GREEN) -> SurroundingRectangle:
    return SurroundingRectangle(mob, color=color, buff=0.045, stroke_width=2.5).set_z_index(3)


def copy_to(source: VGroup, target: VGroup) -> ReplacementTransform:
    return ReplacementTransform(source.copy(), target)


class LabScene(Scene):
    def setup(self) -> None:
        self.camera.background_color = BG
        self.cues: list[dict] = []
        self.caption_mob = None

    def start(self, title: str, function: str) -> None:
        title_mob = text(title, 36).move_to([-6.3, 3.35, 0], aligned_edge=LEFT)
        function_mob = text(function, 21, GREEN).move_to([-6.3, 2.75, 0], aligned_edge=LEFT)
        self.add(title_mob, function_mob, Line([-6.3, 2.35, 0], [6.3, 2.35, 0], color=LINE, stroke_width=1))
        self.add(text("LLM TRAINING LAB  ·  M1", 16, DIM, mono=True).move_to([4.8, -3.76, 0]))

    def step(self, caption: str, *animations: object, hold: float = 1.6, run_time: float = 0.8) -> None:
        self.cues.append({"start": round(float(self.time), 3), "text": caption})
        new_caption = text(caption, 25, INK).move_to([0, -3.2, 0])
        change = FadeIn(new_caption) if self.caption_mob is None else ReplacementTransform(self.caption_mob, new_caption)
        self.play(change, *animations, run_time=run_time)
        self.caption_mob = new_caption
        self.wait(hold)

    def note(self, value: str, color: str = GREEN, y: float = -2.15) -> Text:
        return text(value, 29, color, width=12).move_to([0, y, 0])

    def tear_down(self) -> None:
        directory = Path(config.media_dir) / "timelines"
        directory.mkdir(parents=True, exist_ok=True)
        cues = [{**cue, "end": self.cues[i + 1]["start"] if i + 1 < len(self.cues) else round(float(self.time), 3)} for i, cue in enumerate(self.cues)]
        (directory / f"{type(self).__name__}.json").write_text(json.dumps({"duration": round(float(self.time), 3), "cues": cues}, ensure_ascii=False, indent=2))
        super().tear_down()
