from PIL import Image, ImageSequence, ImageDraw, ImageFont
import sys

if len(sys.argv) < 4:
    print('Usage: compare_gifs_better.py left.gif right.gif out.gif')
    sys.exit(1)

left_path, right_path, out_path = sys.argv[1:4]

left = Image.open(left_path)
right = Image.open(right_path)

left_frames = [frame.convert('RGBA') for frame in ImageSequence.Iterator(left)]
right_frames = [frame.convert('RGBA') for frame in ImageSequence.Iterator(right)]

height = max(left_frames[0].height, right_frames[0].height)

def resize_keep_aspect(img, target_h):
    w, h = img.size
    if h == target_h:
        return img
    nw = int(w * (target_h / h))
    return img.resize((nw, target_h), Image.BICUBIC)

left_frames = [resize_keep_aspect(f, height) for f in left_frames]
right_frames = [resize_keep_aspect(f, height) for f in right_frames]

n = max(len(left_frames), len(right_frames))
left_seq = [left_frames[i % len(left_frames)] for i in range(n)]
right_seq = [right_frames[i % len(right_frames)] for i in range(n)]

# Prepare font
try:
    font = ImageFont.truetype('DejaVuSans-Bold.ttf', 20)
except Exception:
    font = ImageFont.load_default()

combined = []
for idx, (L, R) in enumerate(zip(left_seq, right_seq)):
    label_h = 36
    pad = 8
    new_w = L.width + R.width
    new_h = height + label_h
    new = Image.new('RGBA', (new_w, new_h), (40, 40, 40, 255))
    draw = ImageDraw.Draw(new)
    # paste frames
    new.paste(L, (0, label_h))
    new.paste(R, (L.width, label_h))

    # left label
    left_label = 'Baseline'
    def _text_size(draw, text, font):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            try:
                return draw.textsize(text, font=font)
            except Exception:
                try:
                    bbox = font.getbbox(text)
                    return bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    return (len(text) * 6, 12)

    text_w, text_h = _text_size(draw, left_label, font)
    rect_x0 = pad
    rect_y0 = pad
    rect_x1 = rect_x0 + text_w + pad*2
    rect_y1 = rect_y0 + text_h + pad
    draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], radius=6, fill=(30,30,30,200))
    draw.text((rect_x0+pad, rect_y0+pad//2), left_label, font=font, fill=(255,255,255,255))

    # right label
    right_label = 'Improved: CPG mix=0.25, smooth=0.85'
    text_w, text_h = _text_size(draw, right_label, font)
    rect_x0 = L.width + pad
    rect_y0 = pad
    rect_x1 = rect_x0 + text_w + pad*2
    rect_y1 = rect_y0 + text_h + pad
    draw.rounded_rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], radius=6, fill=(0,120,0,220))
    draw.text((rect_x0+pad, rect_y0+pad//2), right_label, font=font, fill=(255,255,255,255))

    # highlight border around improved side
    border_color = (20,200,80,255)
    border_width = 6
    rx0 = L.width
    ry0 = label_h
    rx1 = new_w-1
    ry1 = new_h-1
    for b in range(border_width):
        draw.rectangle([ (rx0+b, ry0+b), (rx1-b, ry1-b) ], outline=border_color)

    combined.append(new.convert('P', palette=Image.ADAPTIVE))

# save with duration from original if available
duration = left.info.get('duration', 40)
combined[0].save(out_path, save_all=True, append_images=combined[1:], loop=0, duration=duration)
print('Saved', out_path)
