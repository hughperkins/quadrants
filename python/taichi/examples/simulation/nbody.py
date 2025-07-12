# type: ignore

# Authored by Tiantian Liu, Taichi Graphics.
import math
import taichi as ti
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def run_render_loop(render_fn, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    scatter = ax.scatter([], [], c='white', s=planet_radius**2, alpha=0.7)
    
    def render_wrapper(frame):
        positions = render_fn()
        if len(positions) > 0:
            scatter.set_offsets(positions)
        return [scatter]

    ani = animation.FuncAnimation(fig, render_wrapper, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


ti.init(arch=ti.cpu)

# global control
paused = ti.field(ti.i32, ())

# gravitational constant 6.67408e-11, using 1 for simplicity
G = 1

# number of planets
N = 3000
# unit mass
m = 1
# galaxy size
galaxy_size = 0.4
# planet radius (for rendering)
planet_radius = 2
# init vel
init_vel = 120

# time-step size
h = 1e-4
# substepping
substepping = 10

# center of the screen
center = ti.Vector.field(2, ti.f32, ())

# pos, vel and force of the planets
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N)
vel = ti.Vector.field(2, ti.f32, N)
force = ti.Vector.field(2, ti.f32, N)


@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    for i in range(N):
        theta = ti.random() * 2 * math.pi
        r = (ti.sqrt(ti.random()) * 0.6 + 0.4) * galaxy_size
        offset = r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        pos[i] = center[None] + offset
        vel[i] = [-offset.y, offset.x]
        vel[i] *= init_vel


@ti.kernel
def compute_force():
    # clear force
    for i in range(N):
        force[i] = [0.0, 0.0]

    # compute gravitational force
    for i in range(N):
        p = pos[i]
        for j in range(N):
            if i != j:  # double the computation for a better memory footprint and load balance
                diff = p - pos[j]
                r = diff.norm(1e-5)

                # gravitational force -(GMm / r^2) * (diff/r) for i
                f = -G * m * m * (1.0 / r) ** 3 * diff

                # assign to each particle
                force[i] += f


@ti.kernel
def update():
    dt = h / substepping
    for i in range(N):
        # symplectic euler
        vel[i] += dt * force[i] / m
        pos[i] += dt * vel[i]


def main():
    initialize()
    
    def animate():
        for i in range(substepping):
            compute_force()
            update()
        return pos.to_numpy() * 800  # Scale to pixel coordinates

    run_render_loop(render_fn=animate, width=800, height=800)


if __name__ == "__main__":
    main()
