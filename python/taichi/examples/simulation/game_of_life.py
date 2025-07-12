# type: ignore

# Game of Life written in 100 lines of Taichi
# In memory of John Horton Conway (1937 - 2020)

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

    img = ax.imshow(np.zeros((height, width)), cmap='gray', vmin=0, vmax=1)
    
    def render_wrapper(frame):
        im_ti = render_fn()
        image_data = im_ti.to_numpy()
        img.set_array(image_data)
        return [img]

    ani = animation.FuncAnimation(fig, render_wrapper, interval=100, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


ti.init()

n = 64
cell_size = 8
img_size = n * cell_size
alive = ti.field(int, shape=(n, n))  # alive = 1, dead = 0
count = ti.field(int, shape=(n, n))  # count of neighbours


@ti.func
def get_alive(i, j):
    return alive[i, j] if 0 <= i < n and 0 <= j < n else 0


@ti.func
def get_count(i, j):
    return (
        get_alive(i - 1, j)
        + get_alive(i + 1, j)
        + get_alive(i, j - 1)
        + get_alive(i, j + 1)
        + get_alive(i - 1, j - 1)
        + get_alive(i + 1, j - 1)
        + get_alive(i - 1, j + 1)
        + get_alive(i + 1, j + 1)
    )


# See https://www.conwaylife.com/wiki/Cellular_automaton#Rules for more rules
B, S = [3], [2, 3]
# B, S = [2], [0]


@ti.func
def calc_rule(a, c):
    if a == 0:
        for t in ti.static(B):
            if c == t:
                a = 1
    elif a == 1:
        a = 0
        for t in ti.static(S):
            if c == t:
                a = 1
    return a


@ti.kernel
def run():
    for i, j in alive:
        count[i, j] = get_count(i, j)

    for i, j in alive:
        alive[i, j] = calc_rule(alive[i, j], count[i, j])


@ti.kernel
def init():
    for i, j in alive:
        if ti.random() > 0.8:
            alive[i, j] = 1
        else:
            alive[i, j] = 0


def main():
    init()
    
    def animate():
        run()
        return alive

    run_render_loop(render_fn=animate, width=n, height=n)


if __name__ == "__main__":
    main()
