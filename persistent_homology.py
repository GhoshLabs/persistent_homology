import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import gudhi.representations
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# --- 1. Define Toy Problem: Dataset and Model ---

# Toy dataset for a simple regression task
X_train = np.array([[0.1], [0.4], [0.7], [0.9]])
y_train = np.array([[0.2], [0.7], [0.5], [0.1]])

# Toy neural network model: y = w1 * x + w2 (a simple linear model)
# The two parameters are w1 and w2.
def model(X, w1, w2):
    return w1 * X + w2

# Loss function: Mean Squared Error (MSE)
def loss_function(w1, w2):
    y_pred = model(X_train, w1, w2)
    return np.mean((y_pred - y_train)**2)

# --- 2. Discretize the parameter space (the loss landscape) ---
param_resolution = 100
w_range = np.linspace(-2, 2, param_resolution)
W1, W2 = np.meshgrid(w_range, w_range)

# Calculate the loss for each point on the grid
print("Calculating loss landscape...")
vectorized_loss = np.vectorize(loss_function)
Z = vectorized_loss(W1, W2)
print("...done.")

# --- 3. Visualize the loss landscape surface ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Z, cmap='viridis', edgecolor='none')
ax.set_title('Loss Landscape of a Simple Linear Model')
ax.set_xlabel('Weight w1')
ax.set_ylabel('Weight w2')
ax.set_zlabel('Loss (MSE)')
plt.show()

# --- 4. Animate the sublevel set filtration of the loss landscape ---
fig_anim = plt.figure(figsize=(8, 6))
ax_anim = fig_anim.add_subplot(111, projection='3d')
# Filtration values to animate
heights = np.linspace(np.min(Z), np.max(Z), 60)

def update(frame):
    ax_anim.clear()
    mask = np.where(Z < heights[frame], Z, np.nan)
    surf = ax_anim.plot_surface(W1, W2, mask, cmap='Blues', edgecolor='none', alpha=0.8)
    ax_anim.set_title(f"Sublevel Set Surface: Loss < {heights[frame]:.3f}")
    ax_anim.set_xlabel("Weight w1")
    ax_anim.set_ylabel("Weight w2")
    ax_anim.set_zlabel("Loss (MSE)")
    ax_anim.set_zlim(np.min(Z), np.max(Z))
    return surf,

ani = FuncAnimation(fig_anim, update, frames=len(heights), blit=False, repeat=True)
plt.show()

# --- 5. Compute persistent homology using sublevel set filtration ---
# GUDHI expects top-dimensional cells as a flattened 1D array
Z_flat = Z.flatten()

# CubicalComplex requires shape
print("Computing persistence...")
cc = gd.CubicalComplex(dimensions=Z.shape, top_dimensional_cells=Z_flat)
cc.compute_persistence()
print("...done.")

# --- 6. Plot the persistence diagram ---
gd.plot_persistence_diagram(cc.persistence(), legend=True)
plt.title("Persistence Diagram of the Loss Landscape")
plt.show()

# --- 7.
for dim, (birth, death) in cc.persistence():
    print(f"H_{dim} feature: Birth = {birth:.3f}, Death = {death:.3f}, Lifetime = {death - birth if death != float('inf') else 'inf'}")
# ...existing code...
