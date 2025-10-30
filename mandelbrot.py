# mandelbrot.py
import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(Nr=800, Nc=800, N_max=200, threshold=50.0):
    # Create complex plane grid
    x, y = np.mgrid[-2:1:Nr*1j, -1.5:1.5:Nc*1j]
    c = x + 1j * y

    # Initialize z and mask
    z = np.zeros_like(c, dtype=np.complex128)
    mask = np.ones(c.shape, dtype=np.uint8)

    # Iterate and update only non-diverged points
    for _ in range(N_max):
        active = np.abs(z) < threshold
        mask &= active.astype(np.uint8)
        z[active] = z[active] * z[active] + c[active]

    return mask

if __name__ == "__main__":
    mask = mandelbrot(Nr=800, Nc=800, N_max=200, threshold=50.0)
    plt.imshow(mask.T, extent=[-2, 1, -1.5, 1.5], origin="lower")
    plt.gray()
    plt.title("Mandelbrot Set")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")
    plt.savefig("mandelbrot.png", dpi=200, bbox_inches="tight")
    plt.show()
