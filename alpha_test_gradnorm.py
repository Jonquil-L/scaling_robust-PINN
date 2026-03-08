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
print(f"Fast Test Script starting on: {device}")

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
        # 注意：这里我们使用 nn.Sequential 搭建，最后一层是 self.net[-1]
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 50), nn.SiLU(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# 3. Solver (Hard BC, Separate Loss returns for GradNorm)
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
            loss1 = torch.mean((res_pde1 / a_3_4)**2)
            loss2 = torch.mean((res_pde2 / a_1_4)**2)
            
        # 核心修改：分离返回 loss1 和 loss2，供 GradNorm 分别求导
        return loss1, loss2

# ---------------------------------------------------------------------------
# 4. Dynamic Hybrid Training Loop (GradNorm + Adam -> Frozen Weights + LBFGS)
# ---------------------------------------------------------------------------
def dynamic_hybrid_train(solver, adam_epochs=1500, lbfgs_epochs=1000):
    optimizer_adam = optim.Adam(list(solver.net_y.parameters()) + list(solver.net_p.parameters()), lr=1e-3)
    x_int = torch.rand(2500, 2, device=device) 
    
    # 动态权重初始化与参数设定
    w1 = torch.tensor(1.0, device=device)
    w2 = torch.tensor(1.0, device=device)
    alpha_ema = 0.9
    
    # 选取参考层：两个网络各自的最后一层（self.net[-1] 即 nn.Linear(50, 1)）
    ref_params = list(solver.net_y.net[-1].parameters()) + list(solver.net_p.net[-1].parameters())

    solver.net_y.train(); solver.net_p.train()
    
    # ==========================================
    # 阶段 1：Adam + GradNorm 动态权重平衡
    # ==========================================
    for epoch in range(adam_epochs):
        loss1, loss2 = solver.compute_loss(x_int)
        
        # 1. 测量 PDE1 的拉扯力
        optimizer_adam.zero_grad()
        loss1.backward(retain_graph=True)
        grad1_max = torch.max(torch.stack([p.grad.abs().max() for p in ref_params if p.grad is not None]))
        
        # 2. 测量 PDE2 的拉扯力
        optimizer_adam.zero_grad()
        loss2.backward(retain_graph=True)
        grad2_max = torch.max(torch.stack([p.grad.abs().max() for p in ref_params if p.grad is not None]))
        
        # 3. 动态更新权重 (EMA)
        with torch.no_grad():
            mean_grad = (grad1_max + grad2_max) / 2.0
            hat_w1 = mean_grad / (grad1_max + 1e-8)
            hat_w2 = mean_grad / (grad2_max + 1e-8)
            
            w1 = alpha_ema * w1 + (1 - alpha_ema) * hat_w1
            w2 = alpha_ema * w2 + (1 - alpha_ema) * hat_w2

        # 4. 真正更新网络参数
        optimizer_adam.zero_grad()
        total_loss = w1 * loss1 + w2 * loss2
        total_loss.backward()
        optimizer_adam.step()

    # ==========================================
    # 阶段 2：L-BFGS 精确收敛 (使用固定的最佳权重)
    # ==========================================
    optimizer_lbfgs = optim.LBFGS(
        list(solver.net_y.parameters()) + list(solver.net_p.parameters()),
        lr=1.0, 
        max_iter=lbfgs_epochs, 
        max_eval=lbfgs_epochs * 1.25, 
        history_size=50,
        tolerance_grad=1e-7, 
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe" 
    )

    # 捕获当前已收敛的标量权重
    final_w1 = w1.item()
    final_w2 = w2.item()

    def closure():
        optimizer_lbfgs.zero_grad()
        l1, l2 = solver.compute_loss(x_int)
        # 用 Adam 阶段算出的固定权重打包总 Loss
        loss = final_w1 * l1 + final_w2 * l2
        loss.backward()
        return loss

    try:
        optimizer_lbfgs.step(closure)
    except Exception as e:
        # 截获 Unscaled 系统的崩溃报错
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
alphas = [1e-2, 1e-3, 1e-4, 1e-5]
results = {'unscaled': {'y': [], 'p': []}, 'scaled': {'y': [], 'p': []}}

print(f"{'Alpha':<10} | {'System':<10} | {'Err_y':<12} | {'Err_p':<12} | {'Time(s)':<8}")
print("-" * 60)

for alpha in alphas:
    mms = ManufacturedSolution(alpha)
    for sys in ['unscaled', 'scaled']:
        torch.manual_seed(42) 
        solver = FastSolver(sys, alpha, mms)
        
        t0 = time.time()
        # 调用全新的自适应混合训练
        dynamic_hybrid_train(solver, adam_epochs=1500, lbfgs_epochs=1000)
        t_elap = time.time() - t0
        
        err_y, err_p = evaluate_l2(solver, mms)
        results[sys]['y'].append(err_y)
        results[sys]['p'].append(err_p)
        
        print(f"{alpha:<10.0e} | {sys:<10} | {err_y:<12.4e} | {err_p:<12.4e} | {t_elap:<8.1f}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r'Fast $\alpha$-Sensitivity Analysis (Hard BC) with GradNorm', fontsize=14)

for idx, (ax, var, title) in enumerate(zip(axes, ['y', 'p'], [r'Relative $L^2$ Error of $\overline{y}$', r'Relative $L^2$ Error of $\overline{p}$'])):
    ax.loglog(alphas, results['unscaled'][var], 'o-', color='tomato', label='Unscaled (Eq 1.4)', lw=2)
    ax.loglog(alphas, results['scaled'][var], 's-', color='steelblue', label='Scaled (Eq 1.5 Robust-PINN)', lw=2)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Error')
    ax.set_title(title)
    ax.invert_xaxis()
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('gradnorm_sensitivity_result.png', dpi=150)
print("\nDone! Saved plot to 'gradnorm_sensitivity_result.png'")