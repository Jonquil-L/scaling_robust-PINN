import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. 自动适配 Mac MPS 硬件加速
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) for acceleration.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 2. 拉普拉斯算子 (自动微分核心)
def compute_laplacian(u, x):
    """
    计算 u 关于 x 的拉普拉斯算子 (\Delta u)
    要求输入 x 必须设置 requires_grad=True
    """
    # 计算一阶导数 (梯度)
    grad_u = torch.autograd.grad(
        outputs=u, 
        inputs=x, 
        grad_outputs=torch.ones_like(u),
        create_graph=True, # 必须为 True，因为我们需要继续求二阶导
        retain_graph=True
    )[0]
    
    laplacian = torch.zeros_like(u)
    # 对每一个空间维度求二阶导并累加
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

# 3. 解析解生成器
class ManufacturedSolution:
    def __init__(self, alpha, device=device):
        self.alpha = alpha
        self.device = device
        self.pi = math.pi
        
    def exact_y(self, x):
        """真实的未缩放状态变量 \overline{y}"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.sin(self.pi * x1) * torch.sin(self.pi * x2)
        
    def exact_p(self, x):
        """真实的未缩放伴随变量 \overline{p}"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return torch.sin(self.pi * x1) * torch.sin(self.pi * x2)

    def target_yd(self, x):
        """反推的目标状态 y_d"""
        return (1.0 - 2.0 * self.pi**2) * self.exact_y(x)
        
    def source_f(self, x):
        """反推的源项 f (注意这里包含了 alpha^-1，极度依赖 alpha)"""
        return (2.0 * self.pi**2 + 1.0 / self.alpha) * self.exact_y(x)
        
    def prior_ud(self, x):
        """先验控制 u_d"""
        return torch.zeros_like(x[:, 0:1])
    


# 1. 基础多层感知机 (MLP)
class PINN_Net(nn.Module):
    def __init__(self, layers=[2, 50, 50, 50, 2]):
        super(PINN_Net, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        # 使用 Swish (SiLU) 激活函数，因为其二阶导数平滑且非零
        self.activation = nn.SiLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        # 返回两个维度的输出：第一个为状态 y，第二个为伴随变量 p
        return x[:, 0:1], x[:, 1:2]

# 2. 2x2 消融实验的统一求解器
class OptimalControlSolver:
    def __init__(self, system_type, bc_type, alpha, mms, device):
        """
        system_type: 'unscaled' (对应 1.4 式) 或 'scaled' (对应 1.5 式 Robust-PINN)
        bc_type: 'soft' (罚函数法) 或 'hard' (距离函数法)
        alpha: 正则化参数
        """
        self.system_type = system_type
        self.bc_type = bc_type
        self.alpha = alpha
        self.mms = mms
        self.device = device
        
        self.net = PINN_Net().to(device)
        self.omega_bc = 1.0  # 软约束时的边界惩罚权重基准值

    def apply_ansatz(self, x, raw_y, raw_p):
        """应用硬约束距离函数"""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        # 距离函数 D(x) = x1(1-x1)x2(1-x2)
        D_x = x1 * (1.0 - x1) * x2 * (1.0 - x2)
        return raw_y * D_x, raw_p * D_x

    def forward_eval(self, x):
        """获取网络输出并处理边界约束"""
        raw_y, raw_p = self.net(x)
        if self.bc_type == 'hard':
            y_pred, p_pred = self.apply_ansatz(x, raw_y, raw_p)
        else:
            y_pred, p_pred = raw_y, raw_p
        return y_pred, p_pred

    def compute_loss(self, x_interior, x_boundary):
        """核心计算模块：2x2 正交逻辑分支"""
        x_interior.requires_grad_(True)
        
        # --- 获取预测值 ---
        y_pred, p_pred = self.forward_eval(x_interior)
        
        # --- 计算拉普拉斯算子 (调用上一节的函数) ---
        laplace_y = compute_laplacian(y_pred, x_interior)
        laplace_p = compute_laplacian(p_pred, x_interior)
        
        # --- 获取物理场真值 (源项与目标状态) ---
        f = self.mms.source_f(x_interior)
        y_d = self.mms.target_yd(x_interior)
        u_d = self.mms.prior_ud(x_interior)

        # ==========================================
        # 维度 1：方程系统的选择 (System A vs System B)
        # ==========================================
        if self.system_type == 'unscaled':
            # 采用原始 1.4 式 [cite: 22, 23]
            res_pde1 = -laplace_y - (f + u_d) + (1.0 / self.alpha) * p_pred
            res_pde2 = -laplace_p - y_pred + y_d
        elif self.system_type == 'scaled':
            # 采用缩放后的 1.5 式 (Robust-PINN) [cite: 31, 32]
            alpha_pow_1_2 = self.alpha ** 0.5
            alpha_pow_3_4 = self.alpha ** 0.75
            alpha_pow_1_4 = self.alpha ** 0.25
            
            res_pde1 = -alpha_pow_1_2 * laplace_y + p_pred - alpha_pow_3_4 * (f + u_d)
            res_pde2 = -alpha_pow_1_2 * laplace_p - y_pred + alpha_pow_1_4 * y_d
        else:
            raise ValueError("Invalid system_type")

        loss_pde1 = torch.mean(res_pde1 ** 2)
        loss_pde2 = torch.mean(res_pde2 ** 2)

        # ==========================================
        # 维度 2：边界条件处理的选择 (Soft vs Hard)
        # ==========================================
        if self.bc_type == 'soft':
            y_bc_pred, p_bc_pred = self.forward_eval(x_boundary)
            loss_bc1 = torch.mean(y_bc_pred ** 2)
            loss_bc2 = torch.mean(p_bc_pred ** 2)
            loss_bc = loss_bc1 + loss_bc2
        else:
            # 硬约束下，网络结构已经保证了边界绝对为 0，无需加入 Loss
            loss_bc = torch.tensor(0.0, device=self.device)

        # 组合总 Loss
        total_loss = loss_pde1 + loss_pde2 + self.omega_bc * loss_bc
        
        return total_loss, loss_pde1, loss_pde2, loss_bc
    


# 1. 空间采样器
def sample_points(N_interior, N_boundary, device):
    """在 [0, 1] x [0, 1] 区域内采样"""
    # 内部点: (N_interior, 2) 取值在 (0, 1) 之间
    x_interior = torch.rand(N_interior, 2, device=device)
    
    # 边界点: 分配到 4 条边
    N_per_edge = N_boundary // 4
    edge_1 = torch.cat([torch.rand(N_per_edge, 1), torch.zeros(N_per_edge, 1)], dim=1) # y=0
    edge_2 = torch.cat([torch.rand(N_per_edge, 1), torch.ones(N_per_edge, 1)], dim=1)  # y=1
    edge_3 = torch.cat([torch.zeros(N_per_edge, 1), torch.rand(N_per_edge, 1)], dim=1) # x=0
    edge_4 = torch.cat([torch.ones(N_per_edge, 1), torch.rand(N_per_edge, 1)], dim=1)  # x=1
    
    x_boundary = torch.cat([edge_1, edge_2, edge_3, edge_4], dim=0).to(device)
    return x_interior, x_boundary

# 2. 主训练函数
def train_pinn(solver, epochs, N_int=2000, N_bc=400, lr=1e-3):
    optimizer = optim.Adam(solver.net.parameters(), lr=lr)
    
    # 记录 Loss 历史，用于绘制收敛曲线
    history = {'total': [], 'pde1': [], 'pde2': [], 'bc': []}
    
    solver.net.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 每次迭代重新采样，增强泛化能力
        x_int, x_bc = sample_points(N_int, N_bc, solver.device)
        
        total_loss, loss_pde1, loss_pde2, loss_bc = solver.compute_loss(x_int, x_bc)
        total_loss.backward()
        optimizer.step()
        
        # 记录数据
        history['total'].append(total_loss.item())
        history['pde1'].append(loss_pde1.item())
        history['pde2'].append(loss_pde2.item())
        history['bc'].append(loss_bc.item())
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:05d} | Total Loss: {total_loss.item():.4e} | PDE1: {loss_pde1.item():.4e} | PDE2: {loss_pde2.item():.4e} | BC: {loss_bc.item():.4e}")
            
    return history

def evaluate_model(solver, mms, resolution=100):
    solver.net.eval()
    
    # 生成测试网格
    x1 = torch.linspace(0, 1, resolution)
    x2 = torch.linspace(0, 1, resolution)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x_test = torch.stack([X1.flatten(), X2.flatten()], dim=-1).to(solver.device)
    
    with torch.no_grad():
        # 获取网络预测值
        raw_y, raw_p = solver.forward_eval(x_test)
        
        # 极度关键：如果是缩放系统 (1.5式)，必须反向还原为真实物理场
        if solver.system_type == 'scaled':
            y_pred = raw_y * (solver.alpha ** -0.25)
            p_pred = raw_p * (solver.alpha ** 0.25)
        else:
            y_pred = raw_y
            p_pred = raw_p
            
        # 获取解析解真值
        y_exact = mms.exact_y(x_test)
        p_exact = mms.exact_p(x_test)
        
        # 计算相对 L2 误差
        error_y = torch.sqrt(torch.sum((y_pred - y_exact)**2) / torch.sum(y_exact**2))
        error_p = torch.sqrt(torch.sum((p_pred - p_exact)**2) / torch.sum(p_exact**2))
        
    print(f"[{solver.system_type.upper()} + {solver.bc_type.upper()} BC] Relative L2 Error - y: {error_y.item():.4e}, p: {error_p.item():.4e}")
    
    return X1.numpy(), X2.numpy(), y_pred.reshape(resolution, resolution).cpu().numpy(), y_exact.reshape(resolution, resolution).cpu().numpy()

# 初始化参数
alpha_test = 1e-4  # 设定一个较小的 alpha，激发 1.4 式的不稳定性
epochs = 5000
mms = ManufacturedSolution(alpha_test, device)

# 2x2 实验组别配置
experiments = [
    {'system': 'unscaled', 'bc': 'soft'},
    {'system': 'unscaled', 'bc': 'hard'},
    {'system': 'scaled', 'bc': 'soft'},
    {'system': 'scaled', 'bc': 'hard'}
]

results = {}

# 依次运行 4 组实验
for config in experiments:
    sys_type, bc_type = config['system'], config['bc']
    print(f"\n{'='*50}")
    print(f"Running Experiment: System = {sys_type}, BC = {bc_type}")
    print(f"{'='*50}")
    
    # 实例化求解器
    solver = OptimalControlSolver(sys_type, bc_type, alpha_test, mms, device)
    
    # 开始训练
    history = train_pinn(solver, epochs=epochs)
    
    # 评估精度并获取画图数据
    X1, X2, y_pred_grid, y_exact_grid = evaluate_model(solver, mms)
    
    results[f"{sys_type}_{bc_type}"] = {
        'history': history,
        'error_grid': np.abs(y_pred_grid - y_exact_grid)
    }

# 绘制点对点绝对误差热力图 (Point-wise Absolute Error)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Point-wise Absolute Error of $\overline{{y}}$ ($\\alpha={alpha_test}$)', fontsize=16)

titles = ['(A) Unscaled + Soft BC', '(B) Unscaled + Hard BC', 
          '(C) Scaled + Soft BC', '(D) Scaled + Hard BC']
keys = ['unscaled_soft', 'unscaled_hard', 'scaled_soft', 'scaled_hard']

for i, ax in enumerate(axes.flatten()):
    key = keys[i]
    error_grid = results[key]['error_grid']
    
    # 绘制热力图
    im = ax.pcolormesh(X1, X2, error_grid, cmap='jet', shading='auto')
    ax.set_title(titles[i])
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    fig.colorbar(im, ax=ax, format='%.1e')

plt.tight_layout()
plt.show()