import numpy as np
import matplotlib.pyplot as plt

path = r"Enter address :"
with np.load(path, allow_pickle=True) as data:
    imgs = data["samples"]

if imgs.ndim == 4 and imgs.shape[1] in (1,3):
    imgs = np.transpose(imgs, (0,2,3,1))

def to_01(x):
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)

imgs = to_01(imgs)

idx = 0
fig, ax = plt.subplots()
def show(i):
    ax.clear()
    if imgs[i].shape[-1] == 1:
        ax.imshow(imgs[i][...,0], cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(imgs[i])
    ax.set_title(f"Index: {i}")
    ax.axis("off")
    fig.canvas.draw_idle()

def on_key(event):
    global idx
    if event.key in ["right","down"," "]:
        idx = min(idx+1, imgs.shape[0]-1)
        show(idx)
    elif event.key in ["left","up","backspace"]:
        idx = max(idx-1, 0)
        show(idx)

fig.canvas.mpl_connect("key_press_event", on_key)
show(idx)
plt.show()