"""生成 README 用的两张图（纯 Pillow 绘制，无额外依赖）。

输出：
    docs/images/architecture.png  系统架构与检索链路
    docs/images/ablation.png      消融实验结果柱状图

运行：python scripts/make_figures.py
"""

import os

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "docs", "images")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhl.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# 配色
INK = (46, 52, 64)
MUTED = (94, 105, 122)
BLUE = (94, 129, 172)
LBLUE = (136, 192, 208)
GREEN = (163, 190, 140)
RED = (191, 97, 106)
BG_SOFT = (245, 247, 250)
WHITE = (255, 255, 255)
LINE = (200, 208, 218)


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def center_text(draw, box, text, font, fill):
    x0, y0, x1, y1 = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x0 + (x1 - x0 - w) / 2 - bbox[0],
               y0 + (y1 - y0 - h) / 2 - bbox[1]), text, font=font, fill=fill)


def box(draw, xy, text, font, fill, outline, text_fill=WHITE, radius=14):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)
    center_text(draw, xy, text, font, text_fill)


def arrow(draw, x1, y1, x2, y2, color=MUTED, width=3, head=11):
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):  # 水平
        sign = 1 if x2 > x1 else -1
        draw.polygon([(x2, y2), (x2 - sign * head, y2 - head * 0.6),
                      (x2 - sign * head, y2 + head * 0.6)], fill=color)
    else:  # 垂直
        sign = 1 if y2 > y1 else -1
        draw.polygon([(x2, y2), (x2 - head * 0.6, y2 - sign * head),
                      (x2 + head * 0.6, y2 - sign * head)], fill=color)


def make_architecture():
    W, H = 1680, 862
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)

    f_title = load_font(40)
    f_stage = load_font(24)
    f_small = load_font(19)
    f_tiny = load_font(16)

    d.text((60, 40), "胡桃助手 · 系统架构与检索链路", font=f_title, fill=INK)
    d.text((60, 96), "Streamlit 前端 · FastAPI 服务 · ReAct Agent · 混合检索（BM25 + FAISS + RRF + Reranker）",
           font=f_small, fill=MUTED)

    # ---------- 主链路：5 个阶段 ----------
    stages = [
        ("Streamlit 前端", "chat_input / write_stream"),
        ("FastAPI 接口", "ChatRequest 校验 · SSE"),
        ("ReAct Agent", "决定是否调用工具"),
        ("检索工具", "search_hu_tao_knowledge"),
        ("DeepSeek LLM", "基于检索结果生成"),
    ]
    y0, h = 170, 108
    bw, gap = 292, 34
    x = 60
    boxes = []
    for i, (name, sub) in enumerate(stages):
        x0 = x + i * (bw + gap)
        color = BLUE if i % 2 == 0 else (79, 111, 152)
        box(d, (x0, y0, x0 + bw, y0 + h), "", f_stage, color, color)
        center_text(d, (x0, y0 + 8, x0 + bw, y0 + 62), name, f_stage, WHITE)
        center_text(d, (x0, y0 + 58, x0 + bw, y0 + h - 6), sub, f_tiny, LBLUE)
        boxes.append((x0, y0, x0 + bw, y0 + h))
        if i < len(stages) - 1:
            arrow(d, x0 + bw + 4, y0 + h / 2, x0 + bw + gap - 6, y0 + h / 2)

    # 回传箭头（下方绕回）
    d.line((boxes[-1][2] - 40, y0 + h, boxes[-1][2] - 40, y0 + h + 34), fill=GREEN, width=3)
    d.line((boxes[0][0] + 40, y0 + h + 34, boxes[-1][2] - 40, y0 + h + 34), fill=GREEN, width=3)
    arrow(d, boxes[0][0] + 40, y0 + h + 34, boxes[0][0] + 40, y0 + h + 6, color=GREEN)
    center_text(d, (boxes[0][0], y0 + h + 42, boxes[-1][2], y0 + h + 70),
                "SSE 流式返回：on_chat_model_stream → data: {...} → st.write_stream",
                f_tiny, GREEN)

    # ---------- 检索细节面板 ----------
    py0, py1 = 400, 800
    d.rounded_rectangle((60, py0, W - 60, py1), radius=18, fill=BG_SOFT, outline=LINE, width=2)
    d.text((90, py0 + 22), "检索链路四步（search_hu_tao）", font=load_font(27), fill=INK)

    detail = [
        ("① BM25 关键词", "jieba 分词\nTF-IDF 打分", "Top-10", (163, 190, 140)),
        ("② BGE 向量", "bge-small-zh\n余弦相似度", "Top-3", LBLUE),
        ("③ RRF 融合", "1/(60+rank)\n只看名次", "Top-10", (235, 203, 139)),
        ("④ BGE 精排", "bge-reranker-base\nCross-Encoder", "Top-8", RED),
    ]
    dy0, dh = py0 + 90, 210
    dw, dgap = 330, 58
    dx = 90
    for i, (name, sub, topk, color) in enumerate(detail):
        x0 = dx + i * (dw + dgap)
        box(d, (x0, dy0, x0 + dw, dy0 + dh), "", f_stage, WHITE, color, radius=16)
        center_text(d, (x0, dy0 + 14, x0 + dw, dy0 + 58), name, f_stage, INK)
        d.multiline_text((x0 + 24, dy0 + 70), sub, font=f_tiny, fill=MUTED, spacing=8)
        center_text(d, (x0, dy0 + dh - 52, x0 + dw, dy0 + dh - 14), topk, f_stage, color)
        if i < len(detail) - 1:
            arrow(d, x0 + dw + 6, dy0 + dh / 2, x0 + dw + dgap - 8, dy0 + dh / 2)

    d.text((90, dy0 + dh + 26),
           "融合提升召回但损伤排序（R@1 55% < 单路 60%），所以需要第四步精排把 Top-1 拉回来 → 80%",
           font=f_tiny, fill=MUTED)

    path = os.path.join(OUT_DIR, "architecture.png")
    img.save(path)
    print(f"已生成 {path}  ({W}x{H})")


def make_ablation():
    data = [
        ("只用 BM25", (0.60, 0.75, 0.85)),
        ("只用向量", (0.50, 0.90, 0.90)),
        ("RRF 融合", (0.55, 0.95, 0.95)),
        ("融合 + 精排", (0.80, 1.00, 1.00)),
    ]
    metrics = ["Recall@1", "Recall@3", "Recall@5"]
    colors = [(136, 192, 208), (94, 129, 172), (191, 97, 106)]

    W, H = 1280, 760
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    f_title = load_font(36)
    f_axis = load_font(20)
    f_val = load_font(19)

    d.text((70, 44), "检索消融实验（20 条评测集 / 全量索引 / k=5）", font=f_title, fill=INK)

    left, right, top, bottom = 130, W - 70, 150, H - 160
    plot_w, plot_h = right - left, bottom - top

    # 网格与 y 轴刻度
    for i in range(0, 6):
        v = i / 5
        y = bottom - v * plot_h
        d.line((left, y, right, y), fill=(228, 233, 240), width=1)
        label = f"{int(v * 100)}%"
        bbox = d.textbbox((0, 0), label, font=f_axis)
        d.text((left - 18 - (bbox[2] - bbox[0]), y - 12), label, font=f_axis, fill=MUTED)

    group_w = plot_w / len(data)
    bar_w = 52
    inner = 12
    total_w = len(metrics) * bar_w + (len(metrics) - 1) * inner

    for gi, (name, values) in enumerate(data):
        gx = left + gi * group_w + (group_w - total_w) / 2
        for mi, value in enumerate(values):
            x0 = gx + mi * (bar_w + inner)
            y0 = bottom - value * plot_h
            d.rounded_rectangle((x0, y0, x0 + bar_w, bottom), radius=6,
                                fill=colors[mi], outline=None)
            text = f"{int(value * 100)}%"
            bbox = d.textbbox((0, 0), text, font=f_val)
            d.text((x0 + (bar_w - (bbox[2] - bbox[0])) / 2, y0 - 26), text,
                   font=f_val, fill=INK)
        # x 轴标签
        bbox = d.textbbox((0, 0), name, font=f_axis)
        d.text((gx + (total_w - (bbox[2] - bbox[0])) / 2, bottom + 16), name,
               font=f_axis, fill=INK)

    # 图例
    lx = left
    for mi, metric in enumerate(metrics):
        d.rounded_rectangle((lx, top - 44, lx + 22, top - 22), radius=4, fill=colors[mi])
        d.text((lx + 30, top - 46), metric, font=f_axis, fill=INK)
        lx += 190

    d.text((70, H - 52),
           "结论：两路错误不相关 → 融合把 R@3 抬到 95%；融合损伤排序（R@1 55%）→ 精排修复到 80%",
           font=f_axis, fill=MUTED)

    path = os.path.join(OUT_DIR, "ablation.png")
    img.save(path)
    print(f"已生成 {path}  ({W}x{H})")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    make_architecture()
    make_ablation()
