# Role
You are an expert in Scientific Machine Learning (SciML), PyTorch, and Physics-Informed Neural Networks (PINNs).

# Context
I am conducting a hyperparameter sensitivity and robustness analysis for an optimal control problem solved using PINNs. Previously, I implemented a "Unified Network" approach (a single neural network outputting both the state variable `y` and the adjoint variable `p` as a vector) and compared an `unscaled` system with a `scaled` (Robust-PINN) system across different $\alpha$ values.

# Task
Please write a complete, runnable PyTorch script to perform a new experiment based on my existing Unified Network code. The goal is to test the robustness of the systems to the **global loss weight** ($\omega$).

## Experiment Specifications
1. **Fix Alpha**: Fix the regularization parameter to a constant, small value: $\alpha = 10^{-4}$.
2. **Sweep Global Weight ($\omega$)**: Introduce a global weight parameter $\omega$ (omega) that multiplies the *entire* combined loss. Iterate through the following values for $\omega$: `[0.01, 0.1, 1.0, 10.0, 100.0]`.
3. **Loss Formulation**: The loss function should be updated to reflect this global amplification:
   `Loss = omega * torch.mean(residual_vec ** 2)`
   *(Note: Ensure the `scaled` system still properly normalizes its residuals by dividing by $\alpha^{3/4}$ and $\alpha^{1/4}$ before concatenating them into `residual_vec`.)*
4. **Systems to Compare**: For each value of $\omega$, train and evaluate both the `unscaled` and `scaled` systems.
5. **Evaluation Metrics**: Use the same rigorous evaluation function as before, calculating the relative $L^2$ error, $L^\infty$ (maximum absolute) error, and relative $H^1$ error for both $y$ and $p$.
6. **Plotting**: Generate a 3x2 grid of subplots (similar to the previous script) but with $\omega$ on the x-axis (in log scale) and the Error on the y-axis (in log scale). Title the figure appropriately to reflect the $\omega$-Sensitivity Analysis.

# Base Code Reference
Please base your script on the structural logic of my previous Unified Network code. 

\`\`\`python
[PASTE_YOUR_CURRENT_UNIFIED_NETWORK_CODE_HERE]
\`\`\`

# Expected Output
Provide the **complete, well-commented Python script**. Ensure the L-BFGS optimizer logic handles potential numerical instabilities gracefully, as large $\omega$ values might cause exploding gradients.


