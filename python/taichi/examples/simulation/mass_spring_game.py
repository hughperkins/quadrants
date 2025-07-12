# type: ignore

# Tutorials (Chinese):
# - https://www.bilibili.com/video/BV1UK4y177iH
# - https://www.bilibili.com/video/BV1DK411A771

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
    ax.set_facecolor('#DDDDDD')

    # For springs (lines)
    lines = []
    # For particles (scatter)
    scatter = ax.scatter([], [], s=50, alpha=0.8)
    
    def render_wrapper(frame):
        positions, connections, fixed_indices = render_fn()
        
        # Clear previous lines
        for line in lines:
            line.remove()
        lines.clear()
        
        # Draw springs
        for start, end in connections:
            line, = ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2, alpha=0.5)
            lines.append(line)
        
        # Draw particles
        if len(positions) > 0:
            colors = ['red' if i in fixed_indices else 'black' for i in range(len(positions))]
            scatter.set_offsets(positions)
            scatter.set_color(colors)
        
        return [scatter] + lines

    ani = animation.FuncAnimation(fig, render_wrapper, interval=50, blit=True, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


ti.init(arch=ti.cpu)

spring_Y = ti.field(dtype=ti.f32, shape=())  # Young's modulus
paused = ti.field(dtype=ti.i32, shape=())
drag_damping = ti.field(dtype=ti.f32, shape=())
dashpot_damping = ti.field(dtype=ti.f32, shape=())

max_num_particles = 1024
particle_mass = 1.0
dt = 1e-3
substeps = 10

num_particles = ti.field(dtype=ti.i32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
v = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
f = ti.Vector.field(2, dtype=ti.f32, shape=max_num_particles)
fixed = ti.field(dtype=ti.i32, shape=max_num_particles)

# rest_length[i, j] == 0 means i and j are NOT connected
rest_length = ti.field(dtype=ti.f32, shape=(max_num_particles, max_num_particles))


@ti.kernel
def substep():
    n = num_particles[None]

    # Compute force
    for i in range(n):
        # Gravity
        f[i] = ti.Vector([0, -9.8]) * particle_mass
        for j in range(n):
            if rest_length[i, j] != 0:
                x_ij = x[i] - x[j]
                d = x_ij.normalized()

                # Spring force
                f[i] += -spring_Y[None] * (x_ij.norm() / rest_length[i, j] - 1) * d

                # Dashpot damping
                v_rel = (v[i] - v[j]).dot(d)
                f[i] += -dashpot_damping[None] * v_rel * d

    # We use a semi-implicit Euler (aka symplectic Euler) time integrator
    for i in range(n):
        if not fixed[i]:
            v[i] += dt * f[i] / particle_mass
            v[i] *= ti.exp(-dt * drag_damping[None])  # Drag damping

            x[i] += v[i] * dt
        else:
            v[i] = ti.Vector([0, 0])

        # Collide with four walls
        for d in ti.static(range(2)):
            # d = 0: treating X (horizontal) component
            # d = 1: treating Y (vertical) component

            if x[i][d] < 0:  # Bottom and left
                x[i][d] = 0  # move particle inside
                v[i][d] = 0  # stop it from moving further

            if x[i][d] > 1:  # Top and right
                x[i][d] = 1  # move particle inside
                v[i][d] = 0  # stop it from moving further


@ti.kernel
def new_particle(pos_x: ti.f32, pos_y: ti.f32, fixed_: ti.i32):
    # Taichi doesn't support using vectors as kernel arguments yet, so we pass scalars
    new_particle_id = num_particles[None]
    x[new_particle_id] = [pos_x, pos_y]
    v[new_particle_id] = [0, 0]
    fixed[new_particle_id] = fixed_
    num_particles[None] += 1

    # Connect with existing particles
    for i in range(new_particle_id):
        dist = (x[new_particle_id] - x[i]).norm()
        connection_radius = 0.15
        if dist < connection_radius:
            # Connect the new particle with particle i
            rest_length[i, new_particle_id] = 0.1
            rest_length[new_particle_id, i] = 0.1


@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(num_particles[None]):
        p = ti.Vector([pos_x, pos_y])
        v[i] += -dt * substeps * (x[i] - p) * 100


def main():
    spring_Y[None] = 1000
    drag_damping[None] = 1
    dashpot_damping[None] = 100

    new_particle(0.3, 0.3, False)
    new_particle(0.3, 0.4, False)
    new_particle(0.4, 0.4, False)

    def animate():
        for step in range(substeps):
            substep()

        X = x.to_numpy()
        n = num_particles[None]

        # Get connections for springs
        connections = []
        for i in range(n):
            for j in range(i + 1, n):
                if rest_length[i, j] != 0:
                    connections.append((X[i] * 512, X[j] * 512))  # Scale to pixel coordinates

        # Get fixed particle indices
        fixed_indices = [i for i in range(n) if fixed[i]]

        # Scale positions to pixel coordinates
        positions = X[:n] * 512

        return positions, connections, fixed_indices

    run_render_loop(render_fn=animate, width=512, height=512)


if __name__ == "__main__":
    main()
