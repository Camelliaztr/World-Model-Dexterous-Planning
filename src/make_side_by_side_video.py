import argparse
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def resize_keep_aspect(frame, width):
    img = Image.fromarray(frame).convert("RGB")
    w, h = img.size
    new_h = int(h * width / w)
    return img.resize((width, new_h))


def add_label(img, label):
    img = img.convert("RGBA")
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    draw.rectangle([0, 0, w, 36], fill=(0, 0, 0, 170))
    draw.text((14, 11), label, font=font, fill=(255, 255, 255, 255))

    return Image.alpha_composite(img, overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left_label", default="BC Policy")
    parser.add_argument("--right_label", default="Planning-WV")
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel_width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    left_reader = imageio.get_reader(args.left)
    right_reader = imageio.get_reader(args.right)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out_path, fps=args.fps, quality=8, macro_block_size=16) as writer:
        while True:
            try:
                lf = left_reader.get_next_data()
                rf = right_reader.get_next_data()
            except Exception:
                break

            li = resize_keep_aspect(lf, args.panel_width)
            ri = resize_keep_aspect(rf, args.panel_width)

            li = add_label(li, args.left_label)
            ri = add_label(ri, args.right_label)

            h = max(li.size[1], ri.size[1])
            canvas = Image.new("RGB", (li.size[0] + ri.size[0], h), (20, 20, 20))

            canvas.paste(li, (0, 0))
            canvas.paste(ri, (li.size[0], 0))

            writer.append_data(np.asarray(canvas))

    left_reader.close()
    right_reader.close()

    print("Saved comparison video to:", out_path)


if __name__ == "__main__":
    main()
