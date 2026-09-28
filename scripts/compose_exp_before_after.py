#!/usr/bin/env python
"""Stack the zero-shot and fine-tuned Roy v2 overlay grids of a dt-0.2 L5 model into one
before/after PNG and print a markdown table of the per-family scores.
  python scripts/compose_exp_before_after.py <zero_shot_dir> <finetune_dir> <out.png> [tagA] [tagB]
"""
import sys, csv, os
from PIL import Image, ImageDraw, ImageFont

a, b, out = sys.argv[1:4]
tagA = sys.argv[4] if len(sys.argv) > 4 else "pilot zero-shot"
tagB = sys.argv[5] if len(sys.argv) > 5 else "after exp fine-tune"


def table(d):
    rows = list(csv.DictReader(open(os.path.join(d, "roy_summary.csv"))))
    return {r["amp"]: r for r in rows}


ta, tb = table(a), table(b)
print(f"| family | n | spikes data | spikes {tagA} | spikes {tagB} | mse_z {tagA} | mse_z {tagB} | dtw_z {tagA} | dtw_z {tagB} |")
print("|---|---|---|---|---|---|---|---|---|")
for amp in ["500", "1000", "1500", "2000"]:
    ra, rb = ta.get(amp), tb.get(amp)
    if not ra or not rb:
        continue
    print(f"| Roy{amp} | {ra['n']} | {float(ra['spikes_data']):.1f} | {float(ra['spikes_sim']):.1f} | "
          f"{float(rb['spikes_sim']):.1f} | {float(ra['mse_z_mean']):.2f} | {float(rb['mse_z_mean']):.2f} | "
          f"{float(ra['dtw_z_mean']):.2f} | {float(rb['dtw_z_mean']):.2f} |")

ia = Image.open(os.path.join(a, "roy_overlay_grid.png")).convert("RGB")
ib = Image.open(os.path.join(b, "roy_overlay_grid.png")).convert("RGB")
w = max(ia.width, ib.width)
band = 36
canvas = Image.new("RGB", (w, ia.height + ib.height + 2 * band), "white")
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
except Exception:
    font = ImageFont.load_default()
draw.text((12, 8), f"A. {tagA}", fill="black", font=font)
canvas.paste(ia, (0, band))
draw.text((12, band + ia.height + 8), f"B. {tagB}", fill="black", font=font)
canvas.paste(ib, (0, 2 * band + ia.height))
canvas.save(out)
print("wrote", out, canvas.size)
