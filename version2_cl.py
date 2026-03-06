"""
2x2 Orthogonal Ablation Study for Robust-PINN
==============================================
Dimension 1: Equation System — Unscaled (1.4) vs Scaled (1.5, Robust-PINN)
Dimension 2: BC Treatment   — Soft (Penalty) vs Hard (Distance Function)

Manufactured solution on Ω = [0,1]²:
    y_true(x) = sin(πx₁)sin(πx₂)
    p_true(x) = sin(πx₁)sin(πx₂)

Outputs:
    ablation_loss_evolution.png   — 4-panel loss component curves
    ablation_error_heatmaps_y.png — 4-panel pointwise error for y
    ablation_error_heatmaps_p.png — 4-panel pointwise error for p
    alpha_sensitivity.png         — α sensitivity analysis
"""

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Device setup: MPS > CUDA > CPU
# ---------------------------------------------------------------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ===========================================================================
# Laplacian via Automatic Differentiation
# ===========================================================================
def compute_laplacian(u, x):
    """
    计算 u 关于 x 的拉普拉斯算子 (Δu)
    要求输入 x 必须设置 requires_grad=True
    """
    grad_u = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]

    laplacian = torch.zeros_like(u)
    for i in range(x.shape[1]):
        u_xx_i = torch.autograd.grad(
            outputs=grad_u[:, i:i+1],
            inputs=x,
            grad_outputs=torch.ones_like(grad_u[:, i:i+1]),
            create_graph=True,
            retain_graph=True
        )[0][:, i:i+1]
        laplacian += u_xx_i

    return laplacian


# ===========================================================================
# Manufactured Solution
# ===========================================================================
class ManufacturedSolution:
    def __init__(self, alpha, device=device):
        self.alpha = alpha
        self.device = device
        self.pi = math.pi

    def exact_y(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.sin(self.pi * x1) * torch.sin(self.pi * x2)

    def exact_p(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.sin(self.pi * x1) * torch.sin(self.pi * x2)

    def target_yd(self, x):
        return (1.0 - 2.0 * self.pi**2) * self.exact_y(x)

    def source_f(self, x):
        return (2.0 * self.pi**2 + 1.0 / self.alpha) * self.exact_y(x)

    def prior_ud(self, x):
        return torch.zeros_like(x[:, 0:1])


# ===========================================================================
# MLP Network: [2, 50, 50, 50, 2] + Swish
# ===========================================================================
class PINN_Net(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 2]):
        super(PINN_Net, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.activation = nn.SiLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x[:, 0:1], x[:, 1:2]


# ===========================================================================
# Unified Solver for 2x2 Ablation
# ===========================================================================
class OptimalControlSolver:
    def __init__(self, system_type, bc_type, alpha, mms, device):
        """
        system_type: 'unscaled' (Eq 1.4) or 'scaled' (Eq 1.5, Robust-PINN)
        bc_type: 'soft' (penalty) or 'hard' (distance function)
        """
        self.system_type = system_type
        self.bc_type = bc_type
        self.alpha = alpha
        self.mms = mms
        self.device = device
        self.net = PINN_Net().to(device)
        self.omega_bc = 1.0

    def apply_ansatz(self, x, raw_y, raw_p):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        D_x = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        return raw_y * D_x, raw_p * D_x

    def forward_eval(self, x):
        raw_y, raw_p = self.net(x)
        if self.bc_type == 'hard':
            y_pred, p_pred = self.apply_ansatz(x, raw_y, raw_p)
        else:
            y_pred, p_pred = raw_y, raw_p
        return y_pred, p_pred

    def compute_loss(self, x_interior, x_boundary):
        x_interior.requires_grad_(True)
        y_pred, p_pred = self.forward_eval(x_interior)
        laplace_y = compute_laplacian(y_pred, x_interior)
        laplace_p = compute_laplacian(p_pred, x_interior)

        f = self.mms.source_f(x_interior)
        y_d = self.mms.target_yd(x_interior)
        u_d = self.mms.prior_ud(x_interior)

        if self.system_type == 'unscaled':
            res_pde1 = -laplace_y - (f + u_d) + (1.0 / self.alpha) * p_pred
            res_pde2 = -laplace_p - y_pred + y_d
        elif self.system_type == 'scaled':
            alpha_pow_1_2 = self.alpha ** 0.5
            alpha_pow_3_4 = self.alpha ** 0.75
            alpha_pow_1_4 = self.alpha ** 0.25
            res_pde1 = -alpha_pow_1_2 * laplace_y + p_pred - alpha_pow_3_4 * (f + u_d)
            res_pde2 = -alpha_pow_1_2 * laplace_p - y_pred + alpha_pow_1_4 * y_d
        else:
            raise ValueError("Invalid system_type")

        loss_pde1 = torch.mean(res_pde1 ** 2)
        loss_pde2 = torch.mean(res_pde2 ** 2)

        if self.bc_type == 'soft':
            y_bc_pred, p_bc_pred = self.forward_eval(x_boundary)
            loss_bc = torch.mean(y_bc_pred ** 2) + torch.mean(p_bc_pred ** 2)
        else:
            loss_bc = torch.tensor(0.0, device=self.device)

        total_loss = loss_pde1 + loss_pde2 + self.omega_bc * loss_bc
        return total_loss, loss_pde1, loss_pde2, loss_bc


# ===========================================================================
# Sampling
# ===========================================================================
def sample_points(N_interior, N_boundary, device):
    x_interior = torch.rand(N_interior, 2, device=device)
    N_per_edge = N_boundary // 4
    edge_1 = torch.cat([torch.rand(N_per_edge, 1), torch.zeros(N_per_edge, 1)], dim=1)
    edge_2 = torch.cat([torch.rand(N_per_edge, 1), torch.ones(N_per_edge, 1)], dim=1)
    edge_3 = torch.cat([torch.zeros(N_per_edge, 1), torch.rand(N_per_edge, 1)], dim=1)
    edge_4 = torch.cat([torch.ones(N_per_edge, 1), torch.rand(N_per_edge, 1)], dim=1)
    x_boundary = torch.cat([edge_1, edge_2, edge_3, edge_4], dim=0).to(device)
    return x_interior, x_boundary


# ===========================================================================
# Training
# ===========================================================================
def train_pinn(solver, epochs, N_int=2000, N_bc=400, lr=1e-3):
    optimizer = optim.Adam(solver.net.parameters(), lr=lr)
    history = {'total': [], 'pde1': [], 'pde2': [], 'bc': []}

    solver.net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_int, x_bc = sample_points(N_int, N_bc, solver.device)
        total_loss, loss_pde1, loss_pde2, loss_bc = solver.compute_loss(x_int, x_bc)
        total_loss.backward()
        optimizer.step()

        history['total'].append(total_loss.item())
        history['pde1'].append(loss_pde1.item())
        history['pde2'].append(loss_pde2.item())
        history['bc'].append(loss_bc.item())

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch:05d} | Total: {total_loss.item():.4e} "
                  f"| PDE1: {loss_pde1.item():.4e} | PDE2: {loss_pde2.item():.4e} "
                  f"| BC: {loss_bc.item():.4e}")

    return history


# ===========================================================================
# Enhanced Evaluation
# ===========================================================================
def evaluate_model(solver, mms, resolution=100):
    """Return predictions, exact solutions, pointwise errors, and L2 errors for both y and p."""
    solver.net.eval()

    x1 = torch.linspace(0, 1, resolution)
    x2 = torch.linspace(0, 1, resolution)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_test = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(solver.device)

    with torch.no_grad():
        raw_y, raw_p = solver.forward_eval(x_test)

        # Unscale if using scaled system
        if solver.system_type == 'scaled':
            y_pred = raw_y * (solver.alpha ** -0.25)
            p_pred = raw_p * (solver.alpha ** 0.25)
        else:
            y_pred = raw_y
            p_pred = raw_p

        y_exact = mms.exact_y(x_test)
        p_exact = mms.exact_p(x_test)

        error_y_l2 = torch.sqrt(torch.sum((y_pred - y_exact)**2) / torch.sum(y_exact**2)).item()
        error_p_l2 = torch.sqrt(torch.sum((p_pred - p_exact)**2) / torch.sum(p_exact**2)).item()

    label = f"{solver.system_type.upper()} + {solver.bc_type.upper()} BC"
    print(f"  [{label}] Relative L2 Error — y: {error_y_l2:.4e}, p: {error_p_l2:.4e}")

    res = resolution
    return {
        'X1': X1.numpy(),
        'X2': X2.numpy(),
        'y_pred': y_pred.reshape(res, res).cpu().numpy(),
        'p_pred': p_pred.reshape(res, res).cpu().numpy(),
        'y_exact': y_exact.reshape(res, res).cpu().numpy(),
        'p_exact': p_exact.reshape(res, res).cpu().numpy(),
        'error_y_grid': np.abs(y_pred.reshape(res, res).cpu().numpy()
                               - y_exact.reshape(res, res).cpu().numpy()),
        'error_p_grid': np.abs(p_pred.reshape(res, res).cpu().numpy()
                               - p_exact.reshape(res, res).cpu().numpy()),
        'error_y_l2': error_y_l2,
        'error_p_l2': error_p_l2,
    }


# ===========================================================================
# Visualization: Loss Evolution (2x2 subplot)
# ===========================================================================
EXPERIMENT_KEYS = ['unscaled_soft', 'unscaled_hard', 'scaled_soft', 'scaled_hard']
EXPERIMENT_TITLES = [
    '(A) Unscaled + Soft BC', '(B) Unscaled + Hard BC',
    '(C) Scaled + Soft BC',   '(D) Scaled + Hard BC',
]


def plot_loss_evolution(results, alpha, filename='ablation_loss_evolution.png'):
    """2x2 subplot: each panel shows total/pde1/pde2/bc loss curves (semilogy)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Loss Component Evolution ($\\alpha={alpha}$)', fontsize=16)

    for idx, ax in enumerate(axes.flatten()):
        key = EXPERIMENT_KEYS[idx]
        hist = results[key]['history']
        epochs_arr = np.arange(1, len(hist['total']) + 1)

        ax.semilogy(epochs_arr, hist['total'], label='Total', color='black', lw=1.5)
        ax.semilogy(epochs_arr, hist['pde1'],  label='PDE1 (state)',   color='steelblue', alpha=0.8)
        ax.semilogy(epochs_arr, hist['pde2'],  label='PDE2 (adjoint)', color='darkorange', alpha=0.8)
        if max(hist['bc']) > 0:
            ax.semilogy(epochs_arr, hist['bc'], label='BC', color='forestgreen', alpha=0.8)

        ax.set_title(EXPERIMENT_TITLES[idx])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ===========================================================================
# Visualization: Error Heatmaps (2x2 subplot, one figure per variable)
# ===========================================================================
def plot_error_heatmaps(results, alpha, variable='y',
                        filename='ablation_error_heatmaps_y.png'):
    """2x2 subplot of pointwise absolute error heatmaps for y or p."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    var_label = r'$\overline{y}$' if variable == 'y' else r'$\overline{p}$'
    fig.suptitle(f'Point-wise Absolute Error of {var_label} ($\\alpha={alpha}$)', fontsize=16)

    error_key = f'error_{variable}_grid'

    for idx, ax in enumerate(axes.flatten()):
        key = EXPERIMENT_KEYS[idx]
        eval_data = results[key]['eval']
        X1, X2 = eval_data['X1'], eval_data['X2']
        error_grid = eval_data[error_key]

        im = ax.pcolormesh(X1, X2, error_grid, cmap='jet', shading='auto')
        ax.set_title(EXPERIMENT_TITLES[idx])
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, format='%.1e')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ===========================================================================
# Visualization: Alpha Sensitivity (dual panel)
# ===========================================================================
def plot_alpha_sensitivity(sweep_results, filename='alpha_sensitivity.png'):
    """Dual-panel log-log plot: relative L2 error vs alpha for all 4 groups."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(r'$\alpha$-Sensitivity Analysis: Relative $L^2$ Error', fontsize=14)

    colors = {
        'unscaled_soft': 'tomato',
        'unscaled_hard': 'salmon',
        'scaled_soft':   'steelblue',
        'scaled_hard':   'navy',
    }
    labels = {
        'unscaled_soft': 'Unscaled + Soft BC',
        'unscaled_hard': 'Unscaled + Hard BC',
        'scaled_soft':   'Scaled + Soft BC',
        'scaled_hard':   'Scaled + Hard BC',
    }

    alphas = sorted(sweep_results.keys())

    for var_idx, (ax, var_name, var_label) in enumerate(zip(
            axes,
            ['error_y_l2', 'error_p_l2'],
            [r'$\overline{y}$', r'$\overline{p}$'])):

        for key in EXPERIMENT_KEYS:
            errors = [sweep_results[a][key]['eval'][var_name] for a in alphas]
            ax.loglog(alphas, errors, 'o-', color=colors[key], label=labels[key], lw=1.5)

        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(f'Relative $L^2$ Error of {var_label}')
        ax.set_title(f'Error of {var_label}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        ax.invert_xaxis()  # smaller alpha on the right

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


# ===========================================================================
# Orchestration: run 2x2 ablation for a single alpha
# ===========================================================================
def run_2x2_ablation(alpha, epochs, seed=42):
    """Run all 4 experiment groups for a given alpha. Returns results dict."""
    mms = ManufacturedSolution(alpha, device)
    experiments = [
        ('unscaled', 'soft'),
        ('unscaled', 'hard'),
        ('scaled',   'soft'),
        ('scaled',   'hard'),
    ]
    results = {}

    for sys_type, bc_type in experiments:
        key = f"{sys_type}_{bc_type}"
        print(f"\n{'='*60}")
        print(f"  α={alpha:.0e} | System={sys_type} | BC={bc_type}")
        print(f"{'='*60}")

        # Fixed seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        solver = OptimalControlSolver(sys_type, bc_type, alpha, mms, device)
        t0 = time.time()
        history = train_pinn(solver, epochs=epochs)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")

        eval_data = evaluate_model(solver, mms)
        results[key] = {
            'history': history,
            'eval': eval_data,
            'time': elapsed,
        }

    return results


# ===========================================================================
# Orchestration: alpha sweep
# ===========================================================================
def run_alpha_sweep(alpha_list, epochs, seed=42):
    """Run 2x2 ablation for each alpha. Returns {alpha: results}."""
    sweep = {}
    for alpha in alpha_list:
        print(f"\n{'#'*70}")
        print(f"#  Alpha Sweep: α = {alpha:.0e}")
        print(f"{'#'*70}")
        sweep[alpha] = run_2x2_ablation(alpha, epochs, seed=seed)
    return sweep


# ===========================================================================
# Main
# ===========================================================================
def main():
    # ----------------------------------------------------------------
    # Part 1: 2x2 Ablation at α = 1e-4
    # ----------------------------------------------------------------
    alpha_main = 1e-4
    epochs_main = 5000

    print("\n" + "=" * 70)
    print("  PART 1: 2x2 Ablation Study (α = {:.0e}, {} epochs)".format(alpha_main, epochs_main))
    print("=" * 70)

    results = run_2x2_ablation(alpha_main, epochs_main)

    # Generate loss evolution and error heatmap figures
    plot_loss_evolution(results, alpha_main)
    plot_error_heatmaps(results, alpha_main, variable='y',
                        filename='ablation_error_heatmaps_y.png')
    plot_error_heatmaps(results, alpha_main, variable='p',
                        filename='ablation_error_heatmaps_p.png')

    # Summary table for Part 1
    print("\n" + "=" * 70)
    print("  PART 1 SUMMARY (α = {:.0e})".format(alpha_main))
    print("=" * 70)
    print(f"{'Config':<25s} {'err_y (L2)':>12s} {'err_p (L2)':>12s} {'Time (s)':>10s}")
    print("-" * 60)
    for key in EXPERIMENT_KEYS:
        r = results[key]
        print(f"  {key:<23s} {r['eval']['error_y_l2']:>12.4e} "
              f"{r['eval']['error_p_l2']:>12.4e} {r['time']:>10.1f}")

    # ----------------------------------------------------------------
    # Part 2: Alpha Sensitivity Sweep
    # ----------------------------------------------------------------
    alpha_list = [1e-2, 1e-3, 1e-4, 1e-5]
    epochs_sweep = 5000

    print("\n" + "=" * 70)
    print("  PART 2: α-Sensitivity Sweep")
    print("=" * 70)

    sweep = run_alpha_sweep(alpha_list, epochs_sweep)

    # Reuse Part 1 results for α=1e-4 to avoid redundant training
    sweep[alpha_main] = results

    plot_alpha_sensitivity(sweep)

    # Summary table for Part 2
    print("\n" + "=" * 70)
    print("  PART 2 SUMMARY: α-Sensitivity")
    print("=" * 70)
    for alpha in sorted(alpha_list):
        print(f"\n  α = {alpha:.0e}")
        print(f"  {'Config':<25s} {'err_y':>12s} {'err_p':>12s}")
        print(f"  {'-'*50}")
        for key in EXPERIMENT_KEYS:
            ey = sweep[alpha][key]['eval']['error_y_l2']
            ep = sweep[alpha][key]['eval']['error_p_l2']
            print(f"    {key:<23s} {ey:>12.4e} {ep:>12.4e}")

    print("\n\nDone. Generated files:")
    print("  ablation_loss_evolution.png")
    print("  ablation_error_heatmaps_y.png")
    print("  ablation_error_heatmaps_p.png")
    print("  alpha_sensitivity.png")


if __name__ == "__main__":
    main()
