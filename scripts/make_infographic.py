#!/usr/bin/env python3
"""Build the study's summary infographic from results/*.json and nothing else.

One figure, one point: the two pairs do not agree. Effect sizes are drawn on a
shared axis against the pre-registered 5 pp SESOI band, so a reader can see at
a glance which effects clear the threshold the study committed to in advance.

Hand-rolled SVG, no plotting dependency, matching report.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESOI = 0.05
W, LEFT, RIGHT, TOP, ROW = 960, 290, 150, 96, 38
BG, FG, MUTED = "#ffffff", "#13161a", "#6b7280"
BAND, ZERO = "#eef2ff", "#9ca3af"
POS, NEG, NIL = "#b91c1c", "#1d4ed8", "#6b7280"

TESTS = [("H1", "tool-call accuracy"), ("H2", "injection attack success"),
         ("H4", "degradation control")]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def collect(results_dir: Path):
    rows = []
    for f in sorted(results_dir.glob("*.json")):
        doc = json.loads(f.read_text())
        if doc.get("mock"):
            continue
        for pair_id, pair in doc["pairs"].items():
            for tid, label in TESTS:
                t = pair["tests"].get(tid)
                if not t:
                    continue
                rows.append({
                    "pair": pair_id, "test": tid, "label": label,
                    "rd": t["rd"], "lo": t["rd_ci"][0], "hi": t["rd_ci"][1],
                    "p": t.get("rd_boot_p", t.get("p_boot")),
                    "status": pair["status"],
                })
    return rows


def svg(rows) -> str:
    if not rows:
        raise SystemExit("no real results found")
    height = TOP + len(rows) * ROW + 86
    vals = [abs(v) for r in rows for v in (r["rd"], r["lo"], r["hi"])] + [SESOI * 1.4]
    span = max(vals) * 1.15
    midx = LEFT + (W - LEFT - RIGHT) / 2

    def x(v: float) -> float:
        return midx + (v / span) * ((W - LEFT - RIGHT) / 2)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
         f'viewBox="0 0 {W} {height}" font-family="Inter,Segoe UI,system-ui,sans-serif">',
         f'<rect width="{W}" height="{height}" fill="{BG}"/>',
         f'<text x="28" y="36" font-size="19" font-weight="650" fill="{FG}">'
         f'Abliteration: two model pairs, opposite results</text>',
         f'<text x="28" y="60" font-size="13" fill="{MUTED}">Risk difference, ablated minus stock, '
         f'with 95% bootstrap interval. Shaded band = the 5 pp effect size</text>',
         f'<text x="28" y="78" font-size="13" fill="{MUTED}">pre-registered as the smallest of '
         f'interest. Both pairs are ablation-unverified, so these are descriptive.</text>']

    # SESOI band and zero line
    o.append(f'<rect x="{x(-SESOI):.1f}" y="{TOP - 8}" width="{x(SESOI) - x(-SESOI):.1f}" '
             f'height="{len(rows) * ROW + 10}" fill="{BAND}"/>')
    o.append(f'<line x1="{midx:.1f}" y1="{TOP - 8}" x2="{midx:.1f}" '
             f'y2="{TOP + len(rows) * ROW + 2}" stroke="{ZERO}" stroke-width="1.5"/>')

    for i, r in enumerate(rows):
        y = TOP + i * ROW + ROW / 2
        colour = NIL if r["lo"] <= 0 <= r["hi"] else (POS if r["rd"] > 0 else NEG)
        o.append(f'<text x="28" y="{y + 4:.0f}" font-size="13" fill="{FG}">'
                 f'{esc(r["pair"])} <tspan fill="{MUTED}">{esc(r["test"])} · {esc(r["label"])}</tspan></text>')
        o.append(f'<line x1="{x(r["lo"]):.1f}" y1="{y:.1f}" x2="{x(r["hi"]):.1f}" y2="{y:.1f}" '
                 f'stroke="{colour}" stroke-width="2.5"/>')
        o.extend(
            f'<line x1="{x(e):.1f}" y1="{y - 5:.1f}" x2="{x(e):.1f}" y2="{y + 5:.1f}" '
            f'stroke="{colour}" stroke-width="2.5"/>'
            for e in (r["lo"], r["hi"])
        )
        o.append(f'<circle cx="{x(r["rd"]):.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>')
        # Effect size and p-value together, right-aligned in one column: floating
        # the p-value above each bar collided with the interval whiskers.
        # Two separate right-aligned text elements, not nested tspans: a tspan
        # inside a text-anchor="end" element does not reliably keep the anchor
        # through rendering and the p-value was clipped at the canvas edge.
        pv = "" if r["p"] is None else ("p&lt;0.001" if r["p"] < 0.001 else f"p={r['p']:.3f}")
        o.append(f'<text x="{W - 96}" y="{y + 4:.0f}" font-size="12.5" font-weight="600" '
                 f'fill="{FG}" text-anchor="end">{r["rd"] * 100:+.1f} pp</text>')
        o.append(f'<text x="{W - 24}" y="{y + 4:.0f}" font-size="11" fill="{MUTED}" '
                 f'text-anchor="end">{pv}</text>')

    base = TOP + len(rows) * ROW + 14
    o.extend(
        f'<text x="{x(v):.1f}" y="{base + 14:.0f}" font-size="11" fill="{MUTED}" '
        f'text-anchor="middle">{v * 100:+.0f} pp</text>'
        for v in (-SESOI, 0, SESOI)
    )
    o.append(f'<text x="28" y="{height - 20}" font-size="11.5" fill="{MUTED}">'
             f'Grey = interval crosses zero. Generated from results/*.json by '
             f'scripts/make_infographic.py; no number is typed by hand.</text>')
    o.append("</svg>")
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "diagrams" / "summary.svg"))
    a = ap.parse_args()
    rows = collect(Path(a.results))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg(rows), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} effects from {len(set(r['pair'] for r in rows))} pair(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
