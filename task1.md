# System Prompt: 2x2 Orthogonal Ablation Study for Robust-PINN (Optimal Control Problems)

## Role and Objective
You are an expert PyTorch developer and researcher in Physics-Informed Neural Networks (PINNs). 
Your objective is to write a highly modular, clean PyTorch codebase to conduct a 2x2 orthogonal ablation study. This study validates a scaling strategy (Robust-PINN) for solving PDE-constrained optimal control problems. 

The code must be optimized to run on macOS using Apple Silicon, strictly setting the device to Metal Performance Shaders (`mps`).

## Mathematical Context
We are solving an optimal control problem with a regularization parameter $\alpha$. Directly deriving the first-order optimality conditions yields a parameter-dependent saddle-point system (**Equation 1.4**), which is highly unstable for small $\alpha$ (e.g., $\alpha = 10^{-4}$):
$$-\Delta\overline{y} = f + u_d - \alpha^{-1}\overline{p}$$
$$-\Delta\overline{p} = \overline{y} - y_d$$
with homogeneous Dirichlet boundary conditions: $\overline{y}=0, \overline{p}=0$ on $\partial\Omega$.

To resolve this, the authors introduce a scaling transformation $p = \alpha^{-1/4}\overline{p}$ and $y = \alpha^{1/4}\overline{y}$, resulting in a well-balanced system (**Equation 1.5**):
$$-\alpha^{1/2}\Delta y + p = \alpha^{3/4}(f + u_d)$$
$$-\alpha^{1/2}\Delta p - y = -\alpha^{1/4}y_d$$
with scaled boundary conditions: $y=0, p=0$ on $\partial\Omega$.

## Experimental Design: The 2x2 Matrix
Please design a modular solver that can cross-combine the following two dimensions to form 4 distinct experimental groups:

**Dimension 1: Equation System**
* **System A (Unscaled)**: Computes the PDE residual loss using Equation 1.4.
* **System B (Scaled / Robust)**: Computes the PDE residual loss using Equation 1.5.

**Dimension 2: Boundary Condition (BC) Treatment**
* **Soft BC (Penalty Method)**: Network outputs are unrestricted. The loss function is $\mathcal{L} = \mathcal{L}_{PDE} + \omega \mathcal{L}_{BC}$, where $\omega = 1.0$.
* **Hard BC (Ansatz / Distance Function)**: Network outputs are strictly forced to be 0 on the boundary using a spatial distance function $D(x)$, e.g., $Output = D(x) \cdot NN(x)$. The BC loss is entirely removed: $\mathcal{L} = \mathcal{L}_{PDE}$.

**The 4 Experimental Groups to Run:**
1.  System A + Soft BC
2.  System A + Hard BC
3.  System B + Soft BC
4.  System B + Hard BC

## Data Collection and Visualization Requirements
The code must automatically evaluate these 4 groups and generate comparisons across the following three dimensions:

1.  **Loss Evolution Curves**: Plot the training iterations against the total loss and its individual components ($\mathcal{L}_{PDE1}$, $\mathcal{L}_{PDE2}$, $\mathcal{L}_{BC}$). We expect System A to show extreme instability or gradient domination, while System B remains smooth.
2.  **Point-wise Error Heatmaps**: Generate 2D spatial contour plots of the absolute error $|\overline{y}_{pred} - \overline{y}_{exact}|$. 
3.  **Sensitivity to $\alpha$**: Run the 4 groups across a range of alphas (e.g., $\alpha \in \{10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}\}$). Plot $\alpha$ (log scale) on the x-axis and the final relative $L^2$ error on the y-axis to demonstrate the robustness of System B across different parameter magnitudes.

## Technical & Implementation Constraints
1.  **Hardware**: Must include `device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")`. Ensure all tensors and models are properly sent to the MPS device.
2.  **Manufactured Solutions (MMS)**: Implement a function to generate a synthetic exact solution ($\overline{y}_{exact}, \overline{p}_{exact}$) and compute the corresponding forcing term $f$ and target $y_d$ analytically using SymPy or PyTorch auto-diff, so we have a ground truth for error calculation.
3.  **Architecture**: Use a standard Multi-Layer Perceptron (MLP) with Swish/Tanh activations. 
4.  **Modularity**: Do not hardcode the 4 groups as monolithic scripts. Write a unified `OptimalControlSolver` class where the system type and BC type are passed as arguments (e.g., `solver = OptimalControlSolver(system='scaled', bc='hard', alpha=1e-4)`).

Please provide the complete, runnable Python script with inline comments explaining the automatic differentiation (`torch.autograd.grad`) for the Laplacian operator.