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
# 1. Device Setup
# ---------------------------------------------------------------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Soft BC Alpha-Sensitivity Test starting on: {device}")

# ---------------------------------------------------------------------------
# 2. Core Math & MMS
# ---------------------------------------------------------------------------
def compute_laplacian(u, x):
    grad_u = torch.autograd.grad(outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    laplacian = torch.zeros_like(u)
    for i in range(x.shape[1]):
        u_xx_i = torch.autograd.grad(outputs=grad_u[:, i:i+1], inputs=x,
                                     grad_outputs=torch.ones_like(grad_u[:, i:i+1]),
                                     create_graph=True, retain_graph=True)[0][:, i:i+1]
        laplacian += u_xx_i
    return laplacian

class ManufacturedSolution:
    def __init__(self, alpha, device=device):
        self.alpha = alpha
        self.pi = math.pi
    def exact_y(self, x):
        return torch.sin(self.pi * x[:, 0:1]) * torch.sin(self.pi * x[:, 1:2])
    def exact_p(self, x):
        return self.alpha * torch.sin(self.pi * x[:, 0:1]) * torch.sin(self.pi * x[:, 1:2])
    def target_yd(self, x):
        return (1.0 - 2.0 * self.pi**2 * self.alpha) * self.exact_y(x)
    def source_f(self, x):
        return (2.0 * self.pi**2 + 1.0) * self.exact_y(x)
    def prior_ud(self, x):
        return torch.zeros_like(x[:, 0:1])

    def grad_exact_y(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        dy_dx1 = self.pi * torch.cos(self.pi * x1) * torch.sin(self.pi * x2)
        dy_dx2 = self.pi * torch.sin(self.pi * x1) * torch.cos(self.pi * x2)
        return dy_dx1, dy_dx2

    def grad_exact_p(self, x):
        x1, x2 = x[:, 0:1], x[:, 1:2]
        dp_dx1 = self.alpha * self.pi * torch.cos(self.pi * x1) * torch.sin(self.pi * x2)
        dp_dx2 = self.alpha * self.pi * torch.sin(self.pi * x1) * torch.cos(self.pi * x2)
        return dp_dx1, dp_dx2

class SingleOutputNet(nn.Module):
    def __init__(self):
        super(SingleOutputNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# 3. Sampling
# ---------------------------------------------------------------------------
def sample_points(N_interior, N_boundary, device):
    x_interior = torch.rand(N_interior, 2, device=device)
    N_per_edge = N_boundary // 4
    edge_1 = torch.cat([torch.rand(N_per_edge, 1), torch.zeros(N_per_edge, 1)], dim=1)
    edge_2 = torch.cat([torch.rand(N_per_edge, 1), torch.ones(N_per_edge, 1)], dim=1)
    edge_3 = torch.cat([torch.zeros(N_per_edge, 1), torch.rand(N_per_edge, 1)], dim=1)
    edge_4 = torch.cat([torch.ones(N_per_edge, 1), torch.rand(N_per_edge, 1)], dim=1)
    x_boundary = torch.cat([edge_1, edge_2, edge_3, edge_4], dim=0).to(device)
    return x_interior, x_boundary

# ---------------------------------------------------------------------------
# 4. Soft BC Solver
# ---------------------------------------------------------------------------
class SoftBCSolver:
    def __init__(self, system_type, alpha, mms):
        self.system_type = system_type
        self.alpha = alpha
        self.mms = mms
        self.net_y = SingleOutputNet().to(device)
        self.net_p = SingleOutputNet().to(device)

    def forward_eval(self, x):
        raw_y = self.net_y(x)
        raw_p = self.net_p(x)
        # No Ansatz D(x) — soft BC, direct output + feature scaling
        if self.system_type == 'scaled':
            y_pred = raw_y * (self.alpha ** 0.25)
            p_pred = raw_p * (self.alpha ** 0.75)
        else:
            y_pred, p_pred = raw_y, raw_p
        return y_pred, p_pred

    def compute_loss(self, x_interior, x_boundary):
        x_interior.requires_grad_(True)
        y_pred, p_pred = self.forward_eval(x_interior)
        lap_y = compute_laplacian(y_pred, x_interior)
        lap_p = compute_laplacian(p_pred, x_interior)

        f = self.mms.source_f(x_interior)
        y_d = self.mms.target_yd(x_interior)
        u_d = self.mms.prior_ud(x_interior)

        # --- PDE Loss ---
        if self.system_type == 'unscaled':
            res_pde1 = -lap_y - (f + u_d) + (1.0 / self.alpha) * p_pred
            res_pde2 = -lap_p - y_pred + y_d
            loss_pde1 = torch.mean(res_pde1 ** 2)
            loss_pde2 = torch.mean(res_pde2 ** 2)
        else:
            a_1_2 = self.alpha ** 0.5
            a_3_4 = self.alpha ** 0.75
            a_1_4 = self.alpha ** 0.25
            res_pde1 = -a_1_2 * lap_y + p_pred - a_3_4 * (f + u_d)
            res_pde2 = -a_1_2 * lap_p - y_pred + a_1_4 * y_d
            # Per-residual normalization
            loss_pde1 = torch.mean((res_pde1 / a_3_4) ** 2)
            loss_pde2 = torch.mean((res_pde2 / a_1_4) ** 2)

        # --- BC Loss ---
        y_bnd, p_bnd = self.forward_eval(x_boundary)
        if self.system_type == 'scaled':
            # Strip feature scaling so BC loss ~ O(1)
            loss_bc = torch.mean((y_bnd / (self.alpha ** 0.25)) ** 2) \
                    + torch.mean((p_bnd / (self.alpha ** 0.75)) ** 2)
        else:
            # y_bnd ~ O(1), p_bnd ~ O(alpha) -> loss_bc ~ O(1)
            loss_bc = torch.mean(y_bnd ** 2) + torch.mean(p_bnd ** 2)

        # omega_bc = 1.0 (balanced: PDE ~ O(1), BC ~ O(1))
        total_loss = loss_pde1 + loss_pde2 + loss_bc
        return total_loss, loss_pde1, loss_pde2, loss_bc

# ---------------------------------------------------------------------------
# 5. Hybrid Training (Adam dynamic resampling + L-BFGS static)
# ---------------------------------------------------------------------------
def hybrid_train(solver, adam_epochs=2000, lbfgs_iters=1000):
    params = list(solver.net_y.parameters()) + list(solver.net_p.parameters())

    # --- Phase 1: Adam with dynamic resampling ---
    optimizer_adam = optim.Adam(params, lr=1e-3)
    solver.net_y.train(); solver.net_p.train()

    for epoch in range(adam_epochs):
        # Dynamic resampling each epoch
        x_int, x_bc = sample_points(2500, 400, device)
        optimizer_adam.zero_grad()
        total_loss, l_pde1, l_pde2, l_bc = solver.compute_loss(x_int, x_bc)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1e4)
        optimizer_adam.step()

        if epoch % 500 == 0:
            print(f"    Adam [{epoch:04d}/{adam_epochs}] total={total_loss.item():.4e} "
                  f"pde1={l_pde1.item():.4e} pde2={l_pde2.item():.4e} bc={l_bc.item():.4e}")

    # --- Phase 2: L-BFGS with static points ---
    x_int_static, x_bc_static = sample_points(2500, 400, device)

    optimizer_lbfgs = optim.LBFGS(
        params,
        lr=1.0,
        max_iter=lbfgs_iters,
        max_eval=int(lbfgs_iters * 1.25),
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        total_loss, _, _, _ = solver.compute_loss(x_int_static, x_bc_static)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1e4)
        return total_loss

    try:
        optimizer_lbfgs.step(closure)
    except Exception as e:
        print(f"      [L-BFGS early stopped: {e}]")

# ---------------------------------------------------------------------------
# 6. Error Evaluation (L2, Linf, H1)
# ---------------------------------------------------------------------------
def evaluate_errors(solver, mms):
    solver.net_y.eval(); solver.net_p.eval()

    x1, x2 = torch.meshgrid(torch.linspace(0, 1, 100),
                             torch.linspace(0, 1, 100), indexing='ij')
    x_test = torch.stack([x1.flatten(), x2.flatten()], dim=-1).to(device)
    x_test.requires_grad_(True)

    # Forward (keep graph for gradient computation)
    raw_y, raw_p = solver.forward_eval(x_test)
    # Unscale to physical space for evaluation
    if solver.system_type == 'scaled':
        y_pred = raw_y * (solver.alpha ** -0.25)
        p_pred = raw_p * (solver.alpha ** 0.25)
    else:
        y_pred, p_pred = raw_y, raw_p

    y_exact = mms.exact_y(x_test)
    p_exact = mms.exact_p(x_test)

    # Predicted gradients via autograd
    grad_y_pred = torch.autograd.grad(
        outputs=y_pred, inputs=x_test,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=False, retain_graph=True
    )[0]
    grad_p_pred = torch.autograd.grad(
        outputs=p_pred, inputs=x_test,
        grad_outputs=torch.ones_like(p_pred),
        create_graph=False, retain_graph=False
    )[0]

    # Exact gradients (analytic)
    dy_dx1_exact, dy_dx2_exact = mms.grad_exact_y(x_test)
    dp_dx1_exact, dp_dx2_exact = mms.grad_exact_p(x_test)
    grad_y_exact = torch.cat([dy_dx1_exact, dy_dx2_exact], dim=1)
    grad_p_exact = torch.cat([dp_dx1_exact, dp_dx2_exact], dim=1)

    # Pointwise errors
    err_y = y_pred - y_exact
    err_p = p_pred - p_exact
    grad_err_y = grad_y_pred - grad_y_exact
    grad_err_p = grad_p_pred - grad_p_exact

    # L2 relative error
    l2_y = torch.sqrt(torch.sum(err_y**2) / torch.sum(y_exact**2)).item()
    l2_p = torch.sqrt(torch.sum(err_p**2) / torch.sum(p_exact**2)).item()

    # Linf absolute error
    linf_y = torch.max(torch.abs(err_y)).item()
    linf_p = torch.max(torch.abs(err_p)).item()

    # H1 relative error
    h1_err_y_sq = torch.sum(err_y**2) + torch.sum(grad_err_y**2)
    h1_exact_y_sq = torch.sum(y_exact**2) + torch.sum(grad_y_exact**2)
    h1_y = torch.sqrt(h1_err_y_sq / h1_exact_y_sq).item()

    h1_err_p_sq = torch.sum(err_p**2) + torch.sum(grad_err_p**2)
    h1_exact_p_sq = torch.sum(p_exact**2) + torch.sum(grad_p_exact**2)
    h1_p = torch.sqrt(h1_err_p_sq / h1_exact_p_sq).item()

    return {
        'l2_y': l2_y, 'l2_p': l2_p,
        'linf_y': linf_y, 'linf_p': linf_p,
        'h1_y': h1_y, 'h1_p': h1_p,
    }

# ---------------------------------------------------------------------------
# 7. Main Loop
# ---------------------------------------------------------------------------
alphas = [1e-2, 1e-3, 1e-4, 1e-5]
results = {
    'unscaled': {'l2_y': [], 'l2_p': [], 'linf_y': [], 'linf_p': [], 'h1_y': [], 'h1_p': []},
    'scaled':   {'l2_y': [], 'l2_p': [], 'linf_y': [], 'linf_p': [], 'h1_y': [], 'h1_p': []},
}

print(f"\n{'Alpha':<10} | {'System':<10} | {'L2_y':<11} | {'L2_p':<11} | "
      f"{'Linf_y':<11} | {'Linf_p':<11} | {'H1_y':<11} | {'H1_p':<11} | {'Time(s)':<8}")
print("-" * 112)

for alpha in alphas:
    mms = ManufacturedSolution(alpha)
    for sys in ['unscaled', 'scaled']:
        torch.manual_seed(42)
        solver = SoftBCSolver(sys, alpha, mms)

        t0 = time.time()
        hybrid_train(solver, adam_epochs=2000, lbfgs_iters=1000)
        t_elap = time.time() - t0

        errs = evaluate_errors(solver, mms)
        for k in errs:
            results[sys][k].append(errs[k])

        print(f"{alpha:<10.0e} | {sys:<10} | {errs['l2_y']:<11.4e} | {errs['l2_p']:<11.4e} | "
              f"{errs['linf_y']:<11.4e} | {errs['linf_p']:<11.4e} | "
              f"{errs['h1_y']:<11.4e} | {errs['h1_p']:<11.4e} | {t_elap:<8.1f}")

# ---------------------------------------------------------------------------
# 8. Plotting: 3x2 (L2, Linf, H1 for y and p)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(13, 14))
fig.suptitle(r'$\alpha$-Sensitivity Analysis (Soft BC, $\omega_{bc}$=1) — $L^2$, $L^\infty$, $H^1$ Norms',
             fontsize=15)

norm_keys = [
    ('l2_y',   'l2_p',   r'Relative $L^2$'),
    ('linf_y', 'linf_p', r'$L^\infty$ (max abs error)'),
    ('h1_y',   'h1_p',   r'Relative $H^1$'),
]
var_labels = [r'$\overline{y}$', r'$\overline{p}$']

for row, (key_y, key_p, norm_name) in enumerate(norm_keys):
    for col, (key, var_label) in enumerate(zip([key_y, key_p], var_labels)):
        ax = axes[row, col]
        ax.loglog(alphas, results['unscaled'][key], 'o-', color='tomato',
                  label='Unscaled (Eq 1.4)', lw=2)
        ax.loglog(alphas, results['scaled'][key], 's-', color='steelblue',
                  label='Scaled (Eq 1.5 Robust-PINN)', lw=2)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(f'{norm_name} Error')
        ax.set_title(f'{norm_name} Error of {var_label}')
        ax.invert_xaxis()
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('alpha_test_softBC_result.png', dpi=150)
print("\nDone! Saved plot to 'alpha_test_softBC_result.png'")
