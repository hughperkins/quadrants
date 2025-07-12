# type: ignore

import taichi as ti
from taichi.math import cmul, dot, log2, vec2, vec3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def run_render_loop(render_fn, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    img = ax.imshow(np.zeros((height, width, 3)), vmin=0, vmax=1)
    t = 0
    def render_wrapper(frame):
        nonlocal t
        im_ti = render_fn(t)
        image_data = im_ti.to_numpy()
        img.set_array(image_data)
        t += 0.03
        return [img]

    ani = animation.FuncAnimation(fig, render_wrapper, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


ti.init(arch=ti.gpu)

MAXITERS = 100
width, height = 800, 640
pixels = ti.Vector.field(3, ti.f32, shape=(width, height))


@ti.func
def setcolor(z, i):
    v = log2(i + 1 - log2(log2(z.norm()))) / 5
    col = vec3(0.0)
    if v < 1.0:
        col = vec3(v**4, v**2.5, v)
    else:
        v = ti.max(0.0, 2 - v)
        col = vec3(v, v**1.5, v**3)
    return col


@ti.kernel
def render(time: ti.f32):
    zoo = 0.64 + 0.36 * ti.cos(0.02 * time)
    zoo = ti.pow(zoo, 8.0)
    ca = ti.cos(0.15 * (1.0 - zoo) * time)
    sa = ti.sin(0.15 * (1.0 - zoo) * time)
    for i, j in pixels:
        c = 2.0 * vec2(i, j) / height - vec2(1)
        # c *= 1.16
        xy = vec2(c.x * ca - c.y * sa, c.x * sa + c.y * ca)
        c = vec2(-0.745, 0.186) + xy * zoo
        z = vec2(0.0)
        count = 0.0
        while count < MAXITERS and dot(z, z) < 50:
            z = cmul(z, z) + c
            count += 1.0

        if count == MAXITERS:
            pixels[i, j] = [0, 0, 0]
        else:
            pixels[i, j] = setcolor(z, count)


def main():
    def animate(t: float):
        render(t)
        return pixels

    run_render_loop(render_fn=animate, width=width, height=height)


if __name__ == "__main__":
    main()
