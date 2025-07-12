from functools import partial
import taichi as ti
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def run_render_loop(render_fn, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    img = ax.imshow(np.zeros((height, width)), cmap='hot', vmin=0, vmax=1)
    t = 0
    def render_wrapper(frame):
        nonlocal t
        im_ti = render_fn(t)
        image_data = im_ti.to_numpy().T
        img.set_array(image_data)
        t += 0.05
        return [img]

    ani = animation.FuncAnimation(fig, render_wrapper, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


@ti.func
def complex_sqr(z: ti.template()) -> None:
    return ti.Vector([z[0] ** 2 - z[1] ** 2, z[1] * z[0] * 2])


@ti.kernel
def paint(n: int, t: float, pixels: ti.Template) -> None:
    for i, j in pixels:  # Parallelized over all pixels
        c = ti.Vector([-0.8, ti.cos(t) * 0.2])
        z = ti.Vector([i / n - 1, j / n - 0.5]) * 2
        iterations = 0
        while z.norm() < 20 and iterations < 50:
            z = complex_sqr(z) + c
            iterations += 1
        pixels[i, j] = 1 - iterations * 0.02


def main():
    ti.init(arch=ti.gpu)

    n = 320
    pixels = ti.field(dtype=float, shape=(n * 2, n))

    def animate(t: float):
        paint(n, t, pixels)
        return pixels

    run_render_loop(render_fn=animate, width=n * 2, height=n)


if __name__ == "__main__":
    main()
