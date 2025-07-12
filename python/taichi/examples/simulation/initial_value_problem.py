# type: ignore

import time
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def run_render_loop(render_fn, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    quiver = ax.quiver([], [], [], [], scale=50, scale_units='inches')
    
    def render_wrapper(frame):
        locations, directions = render_fn()
        if len(locations) > 0:
            quiver.set_offsets(locations)
            quiver.set_UVC(directions[:, 0], directions[:, 1])
        return [quiver]

    ani = animation.FuncAnimation(fig, render_wrapper, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def init():
    a = []
    for i in np.linspace(0, 1, n, False):
        for j in np.linspace(0, 1, n, False):
            a.append([i, j])
    return np.array(a).astype(np.float32)


ti.init(arch=ti.gpu)
n = 50
dirs = ti.field(dtype=float, shape=(n * n, 2))
locations_np = init()

locations = ti.field(dtype=float, shape=(n * n, 2))
locations.from_numpy(locations_np)


@ti.kernel
def paint(t: float):
    (o, p) = locations_np.shape
    for i in range(0, o):  # Parallelized over all pixels
        x = locations[i, 0]
        y = locations[i, 1]
        dirs[i, 0] = ti.sin((t * x - y))
        dirs[i, 1] = ti.cos(t * y - x)
        l = (dirs[i, 0] ** 2 + dirs[i, 1] ** 2) ** 0.5
        dirs[i, 0] /= l * 40
        dirs[i, 1] /= l * 40


def main():
    beginning = time.time_ns()
    
    def animate():
        t = (time.time_ns() - beginning) * 0.00000001
        paint(t)
        dirs_np = dirs.to_numpy()
        # Scale coordinates for display
        scaled_locations = locations_np * 500
        return scaled_locations, dirs_np

    run_render_loop(render_fn=animate, width=500, height=500)


if __name__ == "__main__":
    main()
