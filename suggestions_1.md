# Role
You are an expert in Scientific Machine Learning (SciML), PyTorch, and Physics-Informed Neural Networks (PINNs). 

# Context
I have a PyTorch script for a numerical experiment evaluating scaling methods to reduce sensitivity in solving PDE-constrained optimization problems. Currently, the script uses two separate neural networks for the state variable `y` and the adjoint variable `p`, evaluates only the $L^2$ norm error, and computes the loss using separate terms.

# Task
Please modify the provided Python script according to the following two specific requirements:

## Requirement 1: Unified Vector Output and Single-Weight Loss
1. **Combine Neural Networks**: Replace the two separate single-output networks (`net_y` and `net_p`) with a single dual-output neural network. The network should output a 2-dimensional vector representing both `y` and `p`.
2. **Unified Loss Function**: In the loss computation, treat the PDE residuals for `y` and `p` as a single concatenated vector. Use only **one global weight** to adjust this combined vector's related terms in the loss function, rather than summing two separately weighted loss components.

## Requirement 2: Comprehensive Error Evaluation ($L^\infty$ and $H^1$ norms)
1. **Add Exact Gradients**: Update the `ManufacturedSolution` class to include the exact analytical gradients (derivatives with respect to spatial coordinates) for both `y` and `p`.
2. **Calculate Multiple Norms**: Modify the evaluation function. In addition to the existing relative $L^2$ error, compute and output:
   - **$L^\infty$ (Infinity) Norm Error**: The maximum absolute difference between the predicted and exact values.
   - **$H^1$ Norm Error**: The error considering both the function values and their first-order derivatives (gradients). You will need to use `torch.autograd.grad` to compute the gradients of the neural network predictions during evaluation.

# Input Code
Here is my current PyTorch script:

\`\`\`python
[PASTE_YOUR_ORIGINAL_CODE_HERE]
\`\`\`

# Expected Output
Please provide the **complete, runnable Python script** incorporating all the above modifications. Ensure the code is well-commented, especially where the exact gradients, $H^1$ norm calculations, and the single-weight vector loss are implemented. After the code, briefly summarize the key changes made.
