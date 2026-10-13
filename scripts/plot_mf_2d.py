#!/usr/bin/env python3
"""
把 MfVisualizer 导出的 CSV 画成二维散点图(SVG),零第三方依赖。
用法: python3 scripts/plot_mf_2d.py data/mf-2d-embedding.csv docs/images/mf-2d.svg
"""
import csv
import sys
import math


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "data/mf-2d-embedding.csv"
    out_svg = sys.argv[2] if len(sys.argv) > 2 else "docs/images/mf-2d.svg"

    movies = []
    users = []
    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                "id": int(row["id"]),
                "label": row["label"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "bias": float(row["bias"]),
                "tag": row["tag"],
                "color": row["color"],
            }
            if row["type"] == "movie":
                movies.append(rec)
            else:
                users.append(rec)

    # 计算边界,留 15% 边距
    xs = [p["x"] for p in movies + users]
    ys = [p["y"] for p in movies + users]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = (xmax - xmin) * 0.15 or 0.1
    ypad = (ymax - ymin) * 0.15 or 0.1
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    width, height = 900, 700
    pad_left, pad_right = 80, 180
    pad_top, pad_bottom = 60, 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def tx(x):
        return pad_left + (x - xmin) / (xmax - xmin) * plot_w

    def ty(y):
        return height - pad_bottom - (y - ymin) / (ymax - ymin) * plot_h

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append('<rect width="100%" height="100%" fill="#fafafa"/>')
    svg.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#222">k=2 矩阵分解:电影与用户在同一品味空间</text>')

    # 网格
    for i in range(6):
        gx = xmin + (xmax - xmin) * i / 5
        svg.append(f'<line x1="{tx(gx)}" y1="{pad_top}" x2="{tx(gx)}" y2="{height-pad_bottom}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{tx(gx)}" y="{height-pad_bottom+18}" text-anchor="middle" font-size="10" fill="#666">{gx:.2f}</text>')
    for i in range(6):
        gy = ymin + (ymax - ymin) * i / 5
        svg.append(f'<line x1="{pad_left}" y1="{ty(gy)}" x2="{width-pad_right}" y2="{ty(gy)}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{pad_left-8}" y="{ty(gy)+4}" text-anchor="end" font-size="10" fill="#666">{gy:.2f}</text>')

    # 坐标轴
    ox, oy = tx(0), ty(0)
    svg.append(f'<line x1="{pad_left}" y1="{oy}" x2="{width-pad_right}" y2="{oy}" stroke="#333" stroke-width="1.5"/>')
    svg.append(f'<line x1="{ox}" y1="{pad_top}" x2="{ox}" y2="{height-pad_bottom}" stroke="#333" stroke-width="1.5"/>')
    svg.append(f'<text x="{width-pad_right+10}" y="{oy+4}" font-size="12" fill="#333">dim 0</text>')
    svg.append(f'<text x="{ox-4}" y="{pad_top-10}" text-anchor="middle" font-size="12" fill="#333">dim 1</text>')

    # 用户点(灰色小点)
    for u in users:
        svg.append(f'<circle cx="{tx(u["x"])}" cy="{ty(u["y"])}" r="3" fill="#333" opacity="0.5"/>')

    # 电影点(按类型着色,大一点)
    label_positions = []
    for m in movies:
        cx, cy = tx(m["x"]), ty(m["y"])
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{m["color"]}" stroke="#fff" stroke-width="1.5"/>')
        label_positions.append((cx, cy, m["label"], m["color"]))

    # 电影标签:放在点上方,太长截断
    for cx, cy, label, color in label_positions:
        short = label if len(label) <= 22 else label[:19] + "..."
        # 标签背景,避免和网格重叠难以阅读
        svg.append(f'<text x="{cx}" y="{cy-12}" text-anchor="middle" font-size="10" fill="{color}" font-weight="bold">{escape(short)}</text>')

    # 图例
    legend_x = width - pad_right + 20
    legend_y = pad_top + 20
    svg.append(f'<text x="{legend_x}" y="{legend_y}" font-size="13" font-weight="bold" fill="#222">电影类型</text>')
    seen = set()
    for i, m in enumerate(movies):
        if m["tag"] in seen:
            continue
        seen.add(m["tag"])
        ly = legend_y + 22 + len(seen) * 22
        svg.append(f'<circle cx="{legend_x+6}" cy="{ly}" r="6" fill="{m["color"]}"/>')
        svg.append(f'<text x="{legend_x+20}" y="{ly+4}" font-size="12" fill="#333">{escape(m["tag"])}</text>')

    svg.append(f'<text x="{legend_x}" y="{legend_y + 22 + (len(seen)+2) * 22}" font-size="12" fill="#555">灰点 = 用户</text>')
    svg.append('</svg>')

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"SVG 已保存: {out_svg}")


def escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
