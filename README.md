# LLM Training Lab

用单张 RTX 4070 SUPER 开展从零预训练、SFT 和数学 RLVR 实验，建立能够解释、复现、评测和诊断训练的能力。

## 文档入口

| 文档 | 用途 |
|---|---|
| [PLAN.md](PLAN.md) | 修订后的执行路线、阶段任务、资源约束和参考资料 |
| [ACCEPTANCE.md](ACCEPTANCE.md) | 各阶段验收标准，以及需要提交的证据 |
| [TASKS.md](TASKS.md) | 当前进度、下一步和验收记录 |
| [ENVIRONMENT.md](ENVIRONMENT.md) | 已验证的环境、重建命令和存储约定 |
| [STYLE.md](STYLE.md) | 练习骨架、题面、注释、自检与文档的统一风格约定 |
| [实验报告模板](reports/TEMPLATE.md) | 每轮实验的问题、配置、结果与结论 |
| [本地学习网站](site/) | 项目概览、交互式章节、实验记录与源码对照；启动方式见下文 |

所有协作者和 AI 助手（Claude Code、Codex 或其它）在写代码或文档前先读 [STYLE.md](STYLE.md)，它是风格的唯一出处，第 0 节是核心约定速览。

当前 M0 / A0 已通过，证据见 [M0 报告](reports/m0-20260910-025345/REPORT.md)。M1-1 的字符编码与批次构造练习，以及 [M1-2 前向](scripts/03_inspect_forward.py) 和 [因果自注意力](scripts/04_inspect_attention.py) 已通过自检；注意力结果见提交 `ef29806`。下一步是 [logits、目标与有效 token 损失](scripts/05_inspect_loss.py)，随后组合 decoder-only 模型并接入最小训练步；M1 尚未运行模型训练，A1 待验收。训练学习由学习者执行，纯环境校验由助手代办；每完成一项，将证据路径记入 TASKS.md，再更新状态。

## 复习顺序

脚本按文件名前缀排序，依次复习：

| 脚本 | 内容 |
|---|---|
| [00_check_training.py](scripts/00_check_training.py) | CUDA 环境、单参数更新与状态恢复 |
| [01_inspect_tokens.py](scripts/01_inspect_tokens.py) | 字符编码与 next-token 目标位移 |
| [02_split_dataset.py](scripts/02_split_dataset.py) | 训练/验证划分、定长样本与批次 |
| [03_inspect_forward.py](scripts/03_inspect_forward.py) | ID 张量、embedding 与 logits |
| [04_inspect_attention.py](scripts/04_inspect_attention.py) | 单头因果自注意力 |
| [05_inspect_loss.py](scripts/05_inspect_loss.py) | 交叉熵与有效 token 平均损失（当前待填写） |

## 主线

1. M0：环境与最小实验记录。
2. M1：从随机初始化训练小语言模型，理解训练循环。
3. M2：固定评测协议，完成短解答 SFT。
4. M3：Countdown 奖励验证、手写 GRPO、框架对照。
5. M4：GSM8K 数学 SFT → GRPO 闭环。
6. M5：补齐四组对照、关键消融与可复现报告。

完整 BabyLM、偏好分类、Reward Model、DPO 和更复杂数学任务作为扩展。它们不阻塞数学 RLVR 主线。

预训练练习的小模型和后训练使用的 Qwen 是不同的模型谱系。项目完成标准是具备经过实验验证的单卡小模型训练能力；覆盖范围以实际完成的任务为准。

## 本地学习网站

网站位于 `site/`，使用 Astro、MDX 与 React，包含项目首页、现有四项练习的 Manim 实现动画、阅读与交互、loss 待完成任务、M0 实验记录和项目路线。源码与归档材料在构建时从本仓库读取；网站不会执行 Python 练习或启动训练。

使用 Node.js 22.12 以上版本；`site/.nvmrc` 固定本次使用的 24.16.0。已使用 nvm 时，在项目根目录执行：

```bash
cd site
nvm use
npm ci
npm run dev
```

默认地址为 `http://127.0.0.1:4321/`。若端口被占用，以终端实际输出为准。未使用 nvm 时，先确保 `node --version` 满足要求，再执行安装和启动命令。

| 入口 | 内容 |
|---|---|
| `/` | 项目概览、当前练习与两条模型路线 |
| `/learn/` | 学习目录、本地保存的最近阅读位置 |
| `/learn/attention/` | 按实现小节观看 Manim 动画；展开因果遮罩与未来 V 交互、手算和复习题 |
| `/experiments/` | M0 真实结果与原始证据 |
| `/roadmap/` | M0～M5 的实际状态与后续计划 |
| `/source/<slug>/` | 仓库源文件的只读展示与下载 |

在 `site/` 中执行 `npm run build` 完成类型检查和静态构建，结果在 `site/dist/`；后续可由服务器直接提供这个目录。执行 `npm run preview` 可以预览构建结果。练习章节使用 `site/src/pages/learn/*.mdx`，共享目录数据在 `site/src/data/chapters.ts`。

数值检查使用 `npm test`。浏览器检查使用 Playwright：准备 Chromium（`npx playwright install chromium`），在一个终端启动默认端口的本地网站，再在另一终端执行 `npm run test:browser`。如需使用现有浏览器，可通过 `LAB_BROWSER_EXECUTABLE` 指定可执行文件路径。截图写入已忽略的 `site/.playwright/`。

### 每小节的 Manim 动画

四项已完成练习共有 19 段独立动画：字符与词表 4 段、数据准备 6 段、前向计算 4 段、注意力与因果性验证 5 段。每段展示对应实现的输入、元素变化和完整小例子的输出。页面支持选节、倍速、连播、按文字稿跳转与 MP4 下载；详细笔记和交互实验默认折叠。当前视频为 720p 中文字幕版，无配音。Loss 练习尚未完成，未生成实现动画。

分镜在 [subsections.py](visuals/manim/subsections.py)，小节目录在 [catalog.json](visuals/manim/catalog.json)。[examples.py](visuals/manim/examples.py) 通过语法树读取练习中的固定数据，校验分镜适用范围，不导入或运行练习文件。语料或张量更新后需要重新渲染并核对分镜。

在项目根目录执行：

```bash
uv run --locked --script visuals/manim/render.py
# 只重做一个小节
uv run --locked --script visuals/manim/render.py --only attention-mask
```

渲染工具使用独立的脚本依赖与锁文件，固定 Manim Community 0.19.2，不改变训练环境的依赖。渲染需 Python 3.12、ffmpeg（含 ffprobe）、Cairo/Pango 和中文字体；默认检测 PingFang SC、Noto Sans CJK SC 或 Microsoft YaHei，也可用 `LLM_LAB_MANIM_FONT` 指定已安装的字体。`--quality 1080p` 可生成更高分辨率，`--quality preview` 用于分镜预览。

生成的 MP4、海报和 WebVTT 字幕在 `site/public/animations/`，时间轴与网页目录在 `site/src/data/animations.json`；这些产物随网站一起构建。中间文件和渲染日志位于被忽略的 `.cache/manim/`。服务器只需托管完整的 `site/dist/`，观看视频无需 Python 或 Manim。

更新训练进度后，同时核对网站目录与对应章节；数值演示标明教学示例，实验记录引用实际运行证据。网站的展示工作不改变训练阶段的验收状态。
