import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Import the Griewank function from your existing file
from griewank import griewank_function

# --- 1. Discretize the function's domain ---
print("Setting up the grid for the Griewank function...")
grid_resolution = 200
x_range = np.linspace(-100, 100, grid_resolution)
y_range = np.linspace(-100, 100, grid_resolution)
X, Y = np.meshgrid(x_range, y_range)

# --- 2. Calculate the function value for each point on the grid ---
print("Calculating function values...")
Z = griewank_function(X, Y)
print("...done.")

# --- 3. Visualize the function surface ---
print("Displaying the 3D surface plot...")
fig_surf = plt.figure(figsize=(8, 6))
ax_surf = fig_surf.add_subplot(111, projection='3d')
ax_surf.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax_surf.set_title('Griewank Function Surface')
ax_surf.set_xlabel('x')
ax_surf.set_ylabel('y')
ax_surf.set_zlabel('f(x, y)')
plt.show()

# --- 4. Animate the sublevel set filtration ---
print("Preparing sublevel set animation...")
fig_anim = plt.figure(figsize=(8, 6))
ax_anim = fig_anim.add_subplot(111, projection='3d')
# Filtration values to animate
heights = np.linspace(np.min(Z), np.max(Z), 60)

def update(frame):
    ax_anim.clear()
    mask = np.where(Z < heights[frame], Z, np.nan)
    surf = ax_anim.plot_surface(X, Y, mask, cmap='Blues', edgecolor='none', alpha=0.8)
    ax_anim.set_title(f"Sublevel Set Surface: f(x, y) < {heights[frame]:.3f}")
    ax_anim.set_xlabel("x")
    ax_anim.set_ylabel("y")
    ax_anim.set_zlabel("f(x, y)")
    ax_anim.set_zlim(np.min(Z), np.max(Z))
    return surf,

ani = FuncAnimation(fig_anim, update, frames=len(heights), blit=False, repeat=True)
plt.show()

# --- 5. Compute persistent homology using sublevel set filtration ---
# GUDHI's CubicalComplex expects top-dimensional cells as a flattened 1D array
Z_flat = Z.flatten()

print("Computing persistence...")
# Create a CubicalComplex from the grid values
cc = gd.CubicalComplex(dimensions=Z.shape, top_dimensional_cells=Z_flat)
cc.compute_persistence()
print("...done.")

# --- 6. Plot the persistence diagram ---
print("Displaying the persistence diagram...")
gd.plot_persistence_diagram(cc.persistence(), legend=True)
plt.title("Persistence Diagram of the Griewank Function")
plt.show()

# --- 7. Analyze the persistence intervals ---
print("\n--- Persistence Intervals (Homology Features) ---")
h0_features = [p for p in cc.persistence() if p[0] == 0]
h1_features = [p for p in cc.persistence() if p[0] == 1]

print(f"\nFound {len(h0_features)} H_0 features (connected components/local minima).")
print(f"Found {len(h1_features)} H_1 features (cycles/holes).")

# The feature with infinite persistence corresponds to the global minimum
inf_feature = next((p for p in h0_features if p[1][1] == float('inf')), None)
if inf_feature:
    print(f"\nGlobal minimum feature (H_0): Birth = {inf_feature[1][0]:.3f}, Death = inf")

# Sort other features by lifetime to find the most prominent ones
finite_h0 = sorted([p for p in h0_features if p[1][1] != float('inf')], key=lambda p: p[1][1] - p[1][0], reverse=True)
h1_features.sort(key=lambda p: p[1][1] - p[1][0], reverse=True)

print("\nTop 5 most persistent H_0 features (prominent local minima):")
for i, (dim, (birth, death)) in enumerate(finite_h0[:5]):
    print(f"  {i+1}. Birth: {birth:.3f}, Death: {death:.3f}, Lifetime: {death - birth:.3f}")

print("\nTop 5 most persistent H_1 features (prominent cycles):")
for i, (dim, (birth, death)) in enumerate(h1_features[:5]):
    print(f"  {i+1}. Birth: {birth:.3f}, Death: {death:.3f}, Lifetime: {death - birth:.3f}")