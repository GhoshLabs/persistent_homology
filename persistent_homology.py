import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.representations
from griewank import griewank_function
from matplotlib.animation import FuncAnimation

# Step 2: Discretize the domain
grid_size = 100
x = np.linspace(-100, 100, 400)
y = np.linspace(-100, 100, 400)
X, Y = np.meshgrid(x, y)
Z = griewank_function(X, Y)

# Step 3: Visualize the function surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Filtration values to animate
heights = np.linspace(np.min(Z), np.max(Z), 60)

def update(frame):
    ax.clear()
    mask = np.where(Z < heights[frame], Z, np.nan)
    surf = ax.plot_surface(X, Y, mask, cmap='Blues', edgecolor='none', alpha=0.8)
    ax.set_title(f"Sublevel Set Surface: f(x, y) < {heights[frame]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.set_zlim(np.min(Z), np.max(Z))
    return surf,

ani = FuncAnimation(fig, update, frames=len(heights), blit=False, repeat=True)
plt.show()

# Step 4: Compute persistent homology using sublevel set filtration
# GUDHI expects top-dimensional cells as a flattened 1D array
Z_flat = Z.flatten()

# CubicalComplex requires shape
cc = gd.CubicalComplex(dimensions=Z.shape, top_dimensional_cells=Z_flat)
cc.compute_persistence()

# Step 5: Plot the persistence diagram
gd.plot_persistence_diagram(cc.persistence(), legend=True)
#gd.plot_persistence_barcode(cc.persistence(), legend=True)

# ...existing code...

print("Persistence intervals:")
for dim, (birth, death) in cc.persistence():
    print(f"H_{dim} feature: Birth = {birth:.3f}, Death = {death:.3f}, Lifetime = {death - birth if death != float('inf') else 'inf'}")
# ...existing code...

