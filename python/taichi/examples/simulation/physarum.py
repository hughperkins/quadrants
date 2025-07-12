# type: ignore

"""Physarum simulation example.

See https://sagejenson.com/physarum for the details."""

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

    img = ax.imshow(np.zeros((height, width)), cmap='viridis', vmin=0, vmax=1)
    
    def render_wrapper(frame):
        im_ti = render_fn()
        image_data = im_ti.to_numpy()
        img.set_array(image_data)
        return [img]

    ani = animation.FuncAnimation(fig, render_wrapper, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


ti.init(arch=ti.gpu)

PARTICLE_N = 1024
GRID_SIZE = 512
SENSE_ANGLE = 0.20 * np.pi
SENSE_DIST = 4.0
EVAPORATION = 0.95
MOVE_ANGLE = 0.1 * np.pi
MOVE_STEP = 2.0

grid = ti.field(dtype=ti.f32, shape=[2, GRID_SIZE, GRID_SIZE])
position = ti.Vector.field(2, dtype=ti.f32, shape=[PARTICLE_N])
heading = ti.field(dtype=ti.f32, shape=[PARTICLE_N])


@ti.kernel
def init():
    for p in ti.grouped(grid):
        grid[p] = 0.0
    for i in position:
        position[i] = ti.Vector([ti.random(), ti.random()]) * GRID_SIZE
        heading[i] = ti.random() * np.pi * 2.0


@ti.func
def sense(phase, pos, ang):
    p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * SENSE_DIST
    return grid[phase, p.cast(int) % GRID_SIZE]


@ti.kernel
def step(phase: ti.i32):
    # move
    for i in position:
        pos, ang = position[i], heading[i]
        l = sense(phase, pos, ang - SENSE_ANGLE)
        c = sense(phase, pos, ang)
        r = sense(phase, pos, ang + SENSE_ANGLE)
        if l < c < r:
            ang += MOVE_ANGLE
        elif l > c > r:
            ang -= MOVE_ANGLE
        elif c < l and c < r:
            ang += MOVE_ANGLE * (2 * (ti.random() < 0.5) - 1)
        pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * MOVE_STEP
        position[i], heading[i] = pos, ang

    # deposit
    for i in position:
        ipos = position[i].cast(int) % GRID_SIZE
        grid[phase, ipos] += 1.0

    # diffuse
    for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
        a = 0.0
        for di in ti.static(range(-1, 2)):
            for dj in ti.static(range(-1, 2)):
                a += grid[phase, (i + di) % GRID_SIZE, (j + dj) % GRID_SIZE]
        a *= EVAPORATION / 9.0
        grid[1 - phase, i, j] = a


def main():
    init()
    i = 0
    
    def animate():
        nonlocal i
        for _ in range(10):  # Multiple steps per frame for faster simulation
            step(i % 2)
            i += 1
        return grid.to_numpy()[0]

    run_render_loop(render_fn=animate, width=GRID_SIZE, height=GRID_SIZE)


if __name__ == "__main__":
    main()
