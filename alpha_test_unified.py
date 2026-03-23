"""
alpha_test_unified.py
=====================
对照实验：单网络双输出 + 单一权重损失
与 alpha_test.py（双网络 + 逐残差归一化）形成对比

改动点：
  1. DualOutputNet 替代两个 SingleOutputNet，共享隐藏层
  2. Loss = mean( concat(r1, r2)² )，仅一个全局权重 w=1
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
# 1. Device Setup
# ---------------------------------------------------------------------------
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Unified-Net Test Script starting on: {device}")

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


# ---------------------------------------------------------------------------
# 3. 单网络双输出 (Requirement 1a)
# ---------------------------------------------------------------------------
class DualOutputNet(nn.Module):
    """共享隐藏层，输出 (y, p) 二维向量"""
    def __init__(self):
        super(DualOutputNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 2)   # 双输出
        )
    def forward(self, x):
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]


# ---------------------------------------------------------------------------
# 4. Solver: 单网络 + 单一权重
# ---------------------------------------------------------------------------
class UnifiedSolver:
    def __init__(self, system_type, alpha, mms):
        self.system_type = system_type
        self.alpha = alpha
        self.mms = mms
        self.net = DualOutputNet().to(device)

    def forward_eval(self, x):
        raw_y, raw_p = self.net(x)
        # 硬约束 (Ansatz)
        x1, x2 = x[:, 0:1], x[:, 1:2]
        D_x = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        raw_y, raw_p = raw_y * D_x, raw_p * D_x

        # 特征缩放（与 alpha_test.py 保持一致）
        if self.system_type == 'scaled':
            y_pred = raw_y * (self.alpha ** 0.25)
            p_pred = raw_p * (self.alpha ** 0.75)
        else:
            y_pred, p_pred = raw_y, raw_p
        return y_pred, p_pred

    def compute_loss(self, x_interior):
        x_interior.requires_grad_(True)
        y_pred, p_pred = self.forward_eval(x_interior)
        lap_y = compute_laplacian(y_pred, x_interior)
        lap_p = compute_laplacian(p_pred, x_interior)

        f = self.mms.source_f(x_interior)
        y_d = self.mms.target_yd(x_interior)
        u_d = self.mms.prior_ud(x_interior)

        if self.system_type == 'unscaled':
            res_pde1 = -lap_y - (f + u_d) + (1.0 / self.alpha) * p_pred
            res_pde2 = -lap_p - y_pred + y_d
        else:
            a_1_2 = self.alpha ** 0.5
            a_3_4 = self.alpha ** 0.75
            a_1_4 = self.alpha ** 0.25
            res_pde1 = -a_1_2 * lap_y + p_pred - a_3_4 * (f + u_d)
            res_pde2 = -a_1_2 * lap_p - y_pred + a_1_4 * y_d

        # Requirement 1b: 拼接为单一向量，单一权重
        # Loss = mean( [r1; r2]² )，等价于 (mean(r1²) + mean(r2²)) / 2
        residual_vec = torch.cat([res_pde1, res_pde2], dim=0)
        loss = torch.mean(residual_vec ** 2)
        return loss


# ---------------------------------------------------------------------------
# 5. Hybrid Training Loop (Adam + L-BFGS)
# ---------------------------------------------------------------------------
def hybrid_train(solver, adam_epochs=1500, lbfgs_epochs=1000):
    optimizer_adam = optim.Adam(solver.net.parameters(), lr=1e-3)
    x_int = torch.rand(2500, 2, device=device)

    solver.net.train()
    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        loss = solver.compute_loss(x_int)
        loss.backward()
        optimizer_adam.step()

    optimizer_lbfgs = optim.LBFGS(
        solver.net.parameters(),
        lr=1.0,
        max_iter=lbfgs_epochs,
        max_eval=int(lbfgs_epochs * 1.25),
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        loss = solver.compute_loss(x_int)
        loss.backward()
        return loss

    try:
        optimizer_lbfgs.step(closure)
    except Exception as e:
        print(f"      [L-BFGS early stopped: {e}]")


# ---------------------------------------------------------------------------
# 6. Comprehensive Error Evaluation (L², L∞, H¹)
# ---------------------------------------------------------------------------
def evaluate_errors(solver, mms):
    solver.net.eval()

    x1, x2 = torch.meshgrid(torch.linspace(0, 1, 100),
                             torch.linspace(0, 1, 100), indexing='ij')
    x_test = torch.stack([x1.flatten(), x2.flatten()], dim=-1).to(device)
    x_test.requires_grad_(True)

    raw_y, raw_p = solver.forward_eval(x_test)
    if solver.system_type == 'scaled':
        y_pred = raw_y * (solver.alpha ** -0.25)
        p_pred = raw_p * (solver.alpha ** 0.25)
    else:
        y_pred, p_pred = raw_y, raw_p

    y_exact = mms.exact_y(x_test)
    p_exact = mms.exact_p(x_test)

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

    dy_dx1_exact, dy_dx2_exact = mms.grad_exact_y(x_test)
    dp_dx1_exact, dp_dx2_exact = mms.grad_exact_p(x_test)
    grad_y_exact = torch.cat([dy_dx1_exact, dy_dx2_exact], dim=1)
    grad_p_exact = torch.cat([dp_dx1_exact, dp_dx2_exact], dim=1)

    err_y = y_pred - y_exact
    err_p = p_pred - p_exact
    grad_err_y = grad_y_pred - grad_y_exact
    grad_err_p = grad_p_pred - grad_p_exact

    l2_y = torch.sqrt(torch.sum(err_y**2) / torch.sum(y_exact**2)).item()
    l2_p = torch.sqrt(torch.sum(err_p**2) / torch.sum(p_exact**2)).item()

    linf_y = torch.max(torch.abs(err_y)).item()
    linf_p = torch.max(torch.abs(err_p)).item()

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
# 7. Execution & Plotting
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
        solver = UnifiedSolver(sys, alpha, mms)

        t0 = time.time()
        hybrid_train(solver, adam_epochs=1500, lbfgs_epochs=1000)
        t_elap = time.time() - t0

        errs = evaluate_errors(solver, mms)
        for k in errs:
            results[sys][k].append(errs[k])

        print(f"{alpha:<10.0e} | {sys:<10} | {errs['l2_y']:<11.4e} | {errs['l2_p']:<11.4e} | "
              f"{errs['linf_y']:<11.4e} | {errs['linf_p']:<11.4e} | "
              f"{errs['h1_y']:<11.4e} | {errs['h1_p']:<11.4e} | {t_elap:<8.1f}")

# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(13, 14))
fig.suptitle(r'$\alpha$-Sensitivity (Unified Net + Single Weight) — $L^2$, $L^\infty$, $H^1$', fontsize=15)

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
plt.savefig('unified_sensitivity_result.png', dpi=150)
print("\nDone! Saved plot to 'unified_sensitivity_result.png'")
