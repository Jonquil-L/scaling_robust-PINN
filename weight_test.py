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
print(f"Weight Sensitivity Test starting on: {device}")

# ---------------------------------------------------------------------------
# 2. Core Math & MMS (Optimized)
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
# 3. Solver (Hard BC, Intrinsic Non-dim intact)
# ---------------------------------------------------------------------------
class FastSolver:
    def __init__(self, system_type, alpha, mms):
        self.system_type = system_type
        self.alpha = alpha
        self.mms = mms
        self.net_y = SingleOutputNet().to(device)
        self.net_p = SingleOutputNet().to(device)

    def forward_eval(self, x):
        raw_y, raw_p = self.net_y(x), self.net_p(x)
        # 硬约束 (Ansatz)
        x1, x2 = x[:, 0:1], x[:, 1:2]
        D_x = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        raw_y, raw_p = raw_y * D_x, raw_p * D_x
        
        # 特征缩放
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

        f, y_d, u_d = self.mms.source_f(x_interior), self.mms.target_yd(x_interior), self.mms.prior_ud(x_interior)

        if self.system_type == 'unscaled':
            res_pde1 = -lap_y - (f + u_d) + (1.0 / self.alpha) * p_pred
            res_pde2 = -lap_p - y_pred + y_d
            loss1 = torch.mean(res_pde1**2)
            loss2 = torch.mean(res_pde2**2)
        else:
            a_1_2, a_3_4, a_1_4 = self.alpha**0.5, self.alpha**0.75, self.alpha**0.25
            res_pde1 = -a_1_2 * lap_y + p_pred - a_3_4 * (f + u_d)
            res_pde2 = -a_1_2 * lap_p - y_pred + a_1_4 * y_d
            # 保留方程自身的归一化，作为基准
            loss1 = torch.mean((res_pde1 / a_3_4)**2)
            loss2 = torch.mean((res_pde2 / a_1_4)**2)
            
        return loss1, loss2

# ---------------------------------------------------------------------------
# 4. Fixed Weight Hybrid Training Loop (The Core Test)
# ---------------------------------------------------------------------------
def fixed_weight_train(solver, gamma, adam_epochs=1500, lbfgs_epochs=1000):
    optimizer_adam = optim.Adam(list(solver.net_y.parameters()) + list(solver.net_p.parameters()), lr=1e-3)
    x_int = torch.rand(2500, 2, device=device) 
    
    solver.net_y.train(); solver.net_p.train()
    
    # --- 阶段 1：Adam ---
    for epoch in range(adam_epochs):
        optimizer_adam.zero_grad()
        l1, l2 = solver.compute_loss(x_int)
        # 人为施加极端的固定偏置权重
        total_loss = gamma * l1 + 1.0 * l2
        total_loss.backward()
        optimizer_adam.step()

    # --- 阶段 2：L-BFGS ---
    optimizer_lbfgs = optim.LBFGS(
        list(solver.net_y.parameters()) + list(solver.net_p.parameters()),
        lr=1.0, max_iter=lbfgs_epochs, max_eval=lbfgs_epochs * 1.25, 
        history_size=50, tolerance_grad=1e-7, tolerance_change=1e-9,
        line_search_fn="strong_wolfe" 
    )

    def closure():
        optimizer_lbfgs.zero_grad()
        l1, l2 = solver.compute_loss(x_int)
        loss = gamma * l1 + 1.0 * l2
        loss.backward()
        return loss

    try:
        optimizer_lbfgs.step(closure)
    except Exception as e:
        pass 

def evaluate_l2(solver, mms):
    solver.net_y.eval(); solver.net_p.eval()
    x1, x2 = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100), indexing='ij')
    x_test = torch.stack([x1.flatten(), x2.flatten()], dim=-1).to(device)
    
    with torch.no_grad():
        raw_y, raw_p = solver.forward_eval(x_test)
        if solver.system_type == 'scaled':
            y_pred = raw_y * (solver.alpha ** -0.25)
            p_pred = raw_p * (solver.alpha ** 0.25)
        else:
            y_pred, p_pred = raw_y, raw_p
            
        y_exact, p_exact = mms.exact_y(x_test), mms.exact_p(x_test)
        err_y = torch.sqrt(torch.sum((y_pred - y_exact)**2) / torch.sum(y_exact**2)).item()
        err_p = torch.sqrt(torch.sum((p_pred - p_exact)**2) / torch.sum(p_exact**2)).item()
    return err_y, err_p

# ---------------------------------------------------------------------------
# 5. Execution & Plotting
# ---------------------------------------------------------------------------
# 固定一个极具挑战性的 alpha
fixed_alpha = 1e-4

# 设置极端的权重比率 (从 1:100 到 100:1)
gammas = [0.01, 0.1, 1.0, 10.0, 100.0]

results = {'unscaled': {'y': [], 'p': []}, 'scaled': {'y': [], 'p': []}}

print(f"Testing Weight Sensitivity at fixed alpha = {fixed_alpha}")
print(f"{'Gamma':<10} | {'System':<10} | {'Err_y':<12} | {'Err_p':<12} | {'Time(s)':<8}")
print("-" * 60)

mms = ManufacturedSolution(fixed_alpha)

for gamma in gammas:
    for sys in ['unscaled', 'scaled']:
        torch.manual_seed(42) 
        solver = FastSolver(sys, fixed_alpha, mms)
        
        t0 = time.time()
        # 传入外部设置的 gamma 进行训练
        fixed_weight_train(solver, gamma=gamma, adam_epochs=1500, lbfgs_epochs=1000)
        t_elap = time.time() - t0
        
        err_y, err_p = evaluate_l2(solver, mms)
        results[sys]['y'].append(err_y)
        results[sys]['p'].append(err_p)
        
        print(f"{gamma:<10.2f} | {sys:<10} | {err_y:<12.4e} | {err_p:<12.4e} | {t_elap:<8.1f}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Weight Sensitivity Analysis ($\\alpha = {fixed_alpha}$)', fontsize=14)

for idx, (ax, var, title) in enumerate(zip(axes, ['y', 'p'], [r'Relative $L^2$ Error of $\overline{y}$', r'Relative $L^2$ Error of $\overline{p}$'])):
    ax.loglog(gammas, results['unscaled'][var], 'o-', color='tomato', label='Unscaled (Eq 1.4)', lw=2)
    ax.loglog(gammas, results['scaled'][var], 's-', color='steelblue', label='Scaled (Eq 1.5 Robust-PINN)', lw=2)
    ax.set_xlabel(r'$\gamma$ (weight on PDE1)')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('weight_sensitivity_result.png', dpi=150)
plt.close(fig)
print("\nDone! Saved plot to 'weight_sensitivity_result.png'")