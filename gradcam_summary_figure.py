# gradcam_summary_figure.py
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

def collect_images(folder, keyword, k=5):
    # 忽略中文文件名
    def is_ascii(s): return all(ord(c) < 128 for c in s)
    pattern = os.path.join(folder, f"*{keyword.lower()}*gradcam.png")
    paths = sorted([p for p in glob.glob(pattern) if is_ascii(os.path.basename(p))])[:k]
    images = []
    for p in paths:
        try:
            img = Image.open(p).resize((224, 224))
            images.append(img)
        except Exception as e:
            print(f"[ERROR loading image]: {p} -> {e}")
    return images

def stitch_gradcam_rows(input_folder, output_file, k=5, mode="combined"):
    pd_imgs = collect_images(input_folder, "pd", k)
    hc_imgs = collect_images(input_folder, "hc", k)

    if len(pd_imgs) == 0 or len(hc_imgs) == 0:
        print("❌ Error: Not enough images found. Please check gradcam_outputs folder.")
        print(f"PD found: {len(pd_imgs)}, HC found: {len(hc_imgs)}")
        return

    if mode == "pd_only":
        row = pd_imgs
    elif mode == "hc_only":
        row = hc_imgs
    elif mode == "combined":
        # Alternating PD / HC
        combined = [None] * (2 * min(len(pd_imgs), len(hc_imgs)))
        combined[::2] = hc_imgs[:len(combined)//2]
        combined[1::2] = pd_imgs[:len(combined)//2]
        row = combined
    else:
        raise ValueError("Invalid mode")

    widths, heights = zip(*(i.size for i in row))
    total_width, max_height = sum(widths), max(heights)
    new_img = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in row:
        new_img.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    safe_name = os.path.splitext(output_file)[0]
    safe_name = "".join([c if ord(c) < 128 else "_" for c in safe_name]) + ".png"
    new_img.save(safe_name)
    print(f"✅ GradCAM summary image saved as: {safe_name}")

if __name__ == "__main__":
    stitch_gradcam_rows("gradcam_outputs", "gradcam_topk_summary.png", k=5, mode="combined")
