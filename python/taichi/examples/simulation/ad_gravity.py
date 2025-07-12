# type: ignore

def render_loop():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=50)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.title("Autodiff gravity")

    def update():
        for _ in range(50):
            substep()
        pos = x.to_numpy()
        sc.set_offsets(pos)
        plt.draw()
        plt.pause(0.01)

    # Initial draw
    sc.set_offsets(x.to_numpy())
    plt.draw()
    plt.pause(0.01)

    while plt.fignum_exists(fig.number):
        update()

import taichi as ti

ti.init()

N = 8
dt = 1e-5

x = ti.Vector.field(2, dtype=ti.f32, shape=N, needs_grad=True)  # particle positions
v = ti.Vector.field(2, dtype=ti.f32, shape=N)  # particle velocities
U = ti.field(dtype=ti.f32, shape=(), needs_grad=True)  # potential energy


@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        # This is to prevent 1/0 error which can cause wrong derivative
        U[None] += -1 / r.norm(1e-3)  # U += -1 / |r|


@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
    for i in x:
        x[i] += dt * v[i]  # dx/dt = v


def substep():
    with ti.ad.Tape(loss=U):
        # Kernel invocations in this scope will later contribute to partial derivatives of
        # U with respect to input variables such as x.
        compute_U()  # The tape will automatically compute dU/dx and save the results in x.grad
    advance()


@ti.kernel
def init():
    for i in x:
        x[i] = [ti.random(), ti.random()]


def main():
    init()
    render_loop()


if __name__ == "__main__":
    main()
    main()
