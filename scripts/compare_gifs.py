from PIL import Image, ImageSequence
import sys

if len(sys.argv) < 4:
    print('Usage: compare_gifs.py left.gif right.gif out.gif')
    sys.exit(1)

left_path, right_path, out_path = sys.argv[1:4]

left = Image.open(left_path)
right = Image.open(right_path)

# ensure RGBA
left_frames = [frame.convert('RGBA') for frame in ImageSequence.Iterator(left)]
right_frames = [frame.convert('RGBA') for frame in ImageSequence.Iterator(right)]

# match heights by resizing while preserving aspect
height = max(left_frames[0].height, right_frames[0].height)

def resize_keep_aspect(img, target_h):
    w, h = img.size
    if h == target_h:
        return img
    nw = int(w * (target_h / h))
    return img.resize((nw, target_h), Image.BICUBIC)

left_frames = [resize_keep_aspect(f, height) for f in left_frames]
right_frames = [resize_keep_aspect(f, height) for f in right_frames]

# make both sequences same length by looping shorter
n = max(len(left_frames), len(right_frames))
left_seq = [left_frames[i % len(left_frames)] for i in range(n)]
right_seq = [right_frames[i % len(right_frames)] for i in range(n)]

combined = []
for L, R in zip(left_seq, right_seq):
    new_w = L.width + R.width
    new = Image.new('RGBA', (new_w, height))
    new.paste(L, (0, 0))
    new.paste(R, (L.width, 0))
    combined.append(new.convert('P', palette=Image.ADAPTIVE))

# save
combined[0].save(out_path, save_all=True, append_images=combined[1:], loop=0, duration=combined[0].info.get('duration', 40))
print('Saved', out_path)
