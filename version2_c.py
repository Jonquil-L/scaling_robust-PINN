  1 -import torch                                                  
        1 +"""                                                           
        2 +2x2 Orthogonal Ablation Study for Robust-PINN                 
        3 +==============================================                
        4 +Dimension 1: Equation System — Unscaled (1.4) vs Scaled (1.5, 
          + Robust-PINN)                                                 
        5 +Dimension 2: BC Treatment   — Soft (Penalty) vs Hard          
          +(Distance Function)                                           
        6 +                                                              
        7 +Manufactured solution on Ω = [0,1]²:                          
        8 +    y_true(x) = sin(πx₁)sin(πx₂)                              
        9 +    p_true(x) = sin(πx₁)sin(πx₂)                              
       10 +                                                              
       11 +Outputs:                                                      
       12 +    ablation_loss_evolution.png   — 4-panel loss component    
          +curves                                                        
       13 +    ablation_error_heatmaps_y.png — 4-panel pointwise error   
          +for y                                                         
       14 +    ablation_error_heatmaps_p.png — 4-panel pointwise error   
          +for p                                                         
       15 +    alpha_sensitivity.png         — α sensitivity analysis    
       16 +"""                                                           
       17 +                                                              
       18  import math
       19 +import time                                                   
       20 +import torch                                                  
       21  import torch.nn as nn
       22  import torch.optim as optim
       23 -import matplotlib.pyplot as plt                               
       23  import numpy as np
       24 +import matplotlib                                             
       25 +matplotlib.use("Agg")                                         
       26 +import matplotlib.pyplot as plt                               
       27  
       28 -# 1. 自动适配 Mac MPS 硬件加速                                
       29 -if torch.backends.mps.is_available():                         
       28 +# ----------------------------------------------------------- 
          +----------------                                              
       29 +# Device setup: MPS > CUDA > CPU                              
       30 +# ----------------------------------------------------------- 
          +----------------                                              
       31 +if hasattr(torch.backends, "mps") and                         
          +torch.backends.mps.is_available():                            
       32      device = torch.device("mps")
       33 -    print("Using MPS (Metal Performance Shaders) for          
          -acceleration.")                                               
       33  elif torch.cuda.is_available():
       34      device = torch.device("cuda")
       35  else:
       36      device = torch.device("cpu")
       37 +print(f"Using device: {device}")                              
       38  
       39 -# 2. 拉普拉斯算子 (自动微分核心)                              
       39 +                                                              
       40 +# =========================================================== 
          +================                                              
       41 +# Laplacian via Automatic Differentiation                     
       42 +# =========================================================== 
          +================                                              
       43  def compute_laplacian(u, x):
       44      """
       45 -    计算 u 关于 x 的拉普拉斯算子 (\Delta u)                   
       45 +    计算 u 关于 x 的拉普拉斯算子 (Δu)                         
       46      要求输入 x 必须设置 requires_grad=True
       47      """
       48 -    # 计算一阶导数 (梯度)                                     
       48      grad_u = torch.autograd.grad(
       49 -        outputs=u,                                            
       50 -        inputs=x,                                             
       49 +        outputs=u,                                            
       50 +        inputs=x,                                             
       51          grad_outputs=torch.ones_like(u),
       52 -        create_graph=True,                                    
          - # 必须为 True，因为我们需要继续求二阶导                      
       52 +        create_graph=True,                                    
       53          retain_graph=True
       54      )[0]
       55 -                                                              
       55 +                                                              
       56      laplacian = torch.zeros_like(u)
       57 -    # 对每一个空间维度求二阶导并累加                          
       57      for i in range(x.shape[1]):
       58          u_xx_i = torch.autograd.grad(
       59 -            outputs=grad_u[:, i:i+1],                         
       60 -            inputs=x,                                         
       59 +            outputs=grad_u[:, i:i+1],                         
       60 +            inputs=x,                                         
       61              grad_outputs=torch.ones_like(grad_u[:, i:i+1]),
       62              create_graph=True,
       63              retain_graph=True
       64          )[0][:, i:i+1]
       65          laplacian += u_xx_i
       66 -                                                              
       66 +                                                              
       67      return laplacian
       68  
       69 -# 3. 解析解生成器                                             
       69 +                                                              
       70 +# =========================================================== 
          +================                                              
       71 +# Manufactured Solution                                       
       72 +# =========================================================== 
          +================                                              
       73  class ManufacturedSolution:
       74      def __init__(self, alpha, device=device):
       75          self.alpha = alpha
       76          self.device = device
       77          self.pi = math.pi
       78 -                                                              
       78 +                                                              
       79      def exact_y(self, x):
       80 -        """真实的未缩放状态变量 \overline{y}"""               
       80          x1, x2 = x[:, 0:1], x[:, 1:2]
       81          return torch.sin(self.pi * x1) * torch.sin(self.pi *
           x2)
       82 -                                                              
       82 +                                                              
       83      def exact_p(self, x):
       84 -        """真实的未缩放伴随变量 \overline{p}"""               
       84          x1, x2 = x[:, 0:1], x[:, 1:2]
       85          return torch.sin(self.pi * x1) * torch.sin(self.pi *
           x2)
       86  
       87      def target_yd(self, x):
       88 -        """反推的目标状态 y_d"""                              
       88          return (1.0 - 2.0 * self.pi**2) * self.exact_y(x)
       89 -                                                              
       89 +                                                              
       90      def source_f(self, x):
       91 -        """反推的源项 f (注意这里包含了 alpha^-1，极度依赖    
          -alpha)"""                                                     
       91          return (2.0 * self.pi**2 + 1.0 / self.alpha) *
           self.exact_y(x)
       92 -                                                              
       92 +                                                              
       93      def prior_ud(self, x):
       94 -        """先验控制 u_d"""                                    
       94          return torch.zeros_like(x[:, 0:1])
       95 -                                                              
       95  
       96  
       97 -# 1. 基础多层感知机 (MLP)                                     
       97 +# =========================================================== 
          +================                                              
       98 +# MLP Network: [2, 50, 50, 50, 2] + Swish                     
       99 +# =========================================================== 
          +================                                              
      100  class PINN_Net(nn.Module):
      101      def __init__(self, layers=[2, 50, 50, 50, 2]):
      102          super(PINN_Net, self).__init__()
     ...
       82          for i in range(len(layers) - 2):
       83              self.hidden_layers.append(nn.Linear(layers[i],
           layers[i+1]))
       84          self.output_layer = nn.Linear(layers[-2], layers[-1])
       85 -        # 使用 Swish (SiLU)                                   
          -激活函数，因为其二阶导数平滑且非零                            
       85          self.activation = nn.SiLU()
       86  
       87      def forward(self, x):
       88          for layer in self.hidden_layers:
       89              x = self.activation(layer(x))
       90          x = self.output_layer(x)
       91 -        # 返回两个维度的输出：第一个为状态                    
          -y，第二个为伴随变量 p                                         
       91          return x[:, 0:1], x[:, 1:2]
       92  
       93 -# 2. 2x2 消融实验的统一求解器                                 
       93 +                                                              
       94 +# =========================================================== 
          +================                                              
       95 +# Unified Solver for 2x2 Ablation                             
       96 +# =========================================================== 
          +================                                              
       97  class OptimalControlSolver:
       98      def __init__(self, system_type, bc_type, alpha, mms,
           device):
       99          """
      100 -        system_type: 'unscaled' (对应 1.4 式) 或 'scaled' (   
          -对应 1.5 式 Robust-PINN)                                      
      101 -        bc_type: 'soft' (罚函数法) 或 'hard' (距离函数法)     
      102 -        alpha: 正则化参数                                     
      100 +        system_type: 'unscaled' (Eq 1.4) or 'scaled' (Eq 1.5, 
          +Robust-PINN)                                                  
      101 +        bc_type: 'soft' (penalty) or 'hard' (distance function
          +)                                                             
      102          """
      103          self.system_type = system_type
      104          self.bc_type = bc_type
      105          self.alpha = alpha
      106          self.mms = mms
      107          self.device = device
      108 -                                                              
      108          self.net = PINN_Net().to(device)
      109 -        self.omega_bc = 1.0  # 软约束时的边界惩罚权重基准值   
      109 +        self.omega_bc = 1.0                                   
      110  
      111      def apply_ansatz(self, x, raw_y, raw_p):
      112 -        """应用硬约束距离函数"""                              
      112          x1, x2 = x[:, 0:1], x[:, 1:2]
      113 -        # 距离函数 D(x) = x1(1-x1)x2(1-x2)                    
      113          D_x = x1 * (1.0 - x1) * x2 * (1.0 - x2)
      114          return raw_y * D_x, raw_p * D_x
      115  
      116      def forward_eval(self, x):
      117 -        """获取网络输出并处理边界约束"""                      
      117          raw_y, raw_p = self.net(x)
      118          if self.bc_type == 'hard':
      119              y_pred, p_pred = self.apply_ansatz(x, raw_y,
           raw_p)
     ...
      126          return y_pred, p_pred
      127  
      128      def compute_loss(self, x_interior, x_boundary):
      129 -        """核心计算模块：2x2 正交逻辑分支"""                  
      129          x_interior.requires_grad_(True)
      130 -                                                              
      131 -        # --- 获取预测值 ---                                  
      130          y_pred, p_pred = self.forward_eval(x_interior)
      131 -                                                              
      132 -        # --- 计算拉普拉斯算子 (调用上一节的函数) ---         
      131          laplace_y = compute_laplacian(y_pred, x_interior)
      132          laplace_p = compute_laplacian(p_pred, x_interior)
      133 -                                                              
      134 -        # --- 获取物理场真值 (源项与目标状态) ---             
      133 +                                                              
      134          f = self.mms.source_f(x_interior)
      135          y_d = self.mms.target_yd(x_interior)
      136          u_d = self.mms.prior_ud(x_interior)
      137  
      138 -        # ==========================================          
      139 -        # 维度 1：方程系统的选择 (System A vs System B)       
      140 -        # ==========================================          
      138          if self.system_type == 'unscaled':
      139 -            # 采用原始 1.4 式 [cite: 22, 23]                  
      139              res_pde1 = -laplace_y - (f + u_d) + (1.0 /
           self.alpha) * p_pred
      140              res_pde2 = -laplace_p - y_pred + y_d
      141          elif self.system_type == 'scaled':
      142 -            # 采用缩放后的 1.5 式 (Robust-PINN) [cite: 31,    
          -32]                                                           
      142              alpha_pow_1_2 = self.alpha ** 0.5
      143              alpha_pow_3_4 = self.alpha ** 0.75
      144              alpha_pow_1_4 = self.alpha ** 0.25
      145 -                                                              
      145              res_pde1 = -alpha_pow_1_2 * laplace_y + p_pred -
           alpha_pow_3_4 * (f + u_d)
      146              res_pde2 = -alpha_pow_1_2 * laplace_p - y_pred +
           alpha_pow_1_4 * y_d
      147          else:
     ...
      162          loss_pde1 = torch.mean(res_pde1 ** 2)
      163          loss_pde2 = torch.mean(res_pde2 ** 2)
      164  
      165 -        # ==========================================          
      166 -        # 维度 2：边界条件处理的选择 (Soft vs Hard)           
      167 -        # ==========================================          
      165          if self.bc_type == 'soft':
      166              y_bc_pred, p_bc_pred =
           self.forward_eval(x_boundary)
      167 -            loss_bc1 = torch.mean(y_bc_pred ** 2)             
      168 -            loss_bc2 = torch.mean(p_bc_pred ** 2)             
      169 -            loss_bc = loss_bc1 + loss_bc2                     
      167 +            loss_bc = torch.mean(y_bc_pred ** 2)              
          + + torch.mean(p_bc_pred ** 2)                                 
      168          else:
      169 -            # 硬约束下，网络结构已经保证了边界绝对为          
          -0，无需加入 Loss                                              
      169              loss_bc = torch.tensor(0.0, device=self.device)
      170  
      171 -        # 组合总 Loss                                         
      171          total_loss = loss_pde1 + loss_pde2 + self.omega_bc *
           loss_bc
      172 -                                                              
      172          return total_loss, loss_pde1, loss_pde2, loss_bc
      173 -                                                              
      173  
      174  
      175 -# 1. 空间采样器                                               
      175 +# =========================================================== 
          +================                                              
      176 +# Sampling                                                    
      177 +# =========================================================== 
          +================                                              
      178  def sample_points(N_interior, N_boundary, device):
      179 -    """在 [0, 1] x [0, 1] 区域内采样"""                       
      180 -    # 内部点: (N_interior, 2) 取值在 (0, 1) 之间              
      179      x_interior = torch.rand(N_interior, 2, device=device)
      180 -                                                              
      181 -    # 边界点: 分配到 4 条边                                   
      180      N_per_edge = N_boundary // 4
      181 -    edge_1 = torch.cat([torch.rand(N_per_edge, 1),            
          -torch.zeros(N_per_edge, 1)], dim=1) # y=0                     
      182 -    edge_2 = torch.cat([torch.rand(N_per_edge, 1),            
          -torch.ones(N_per_edge, 1)], dim=1)  # y=1                     
      183 -    edge_3 = torch.cat([torch.zeros(N_per_edge, 1),           
          -torch.rand(N_per_edge, 1)], dim=1) # x=0                      
      184 -    edge_4 = torch.cat([torch.ones(N_per_edge, 1),            
          -torch.rand(N_per_edge, 1)], dim=1)  # x=1                     
      185 -                                                              
      181 +    edge_1 = torch.cat([torch.rand(N_per_edge, 1),            
          +torch.zeros(N_per_edge, 1)], dim=1)                           
      182 +    edge_2 = torch.cat([torch.rand(N_per_edge, 1),            
          +torch.ones(N_per_edge, 1)], dim=1)                            
      183 +    edge_3 = torch.cat([torch.zeros(N_per_edge, 1),           
          +torch.rand(N_per_edge, 1)], dim=1)                            
      184 +    edge_4 = torch.cat([torch.ones(N_per_edge, 1),            
          +torch.rand(N_per_edge, 1)], dim=1)                            
      185      x_boundary = torch.cat([edge_1, edge_2, edge_3, edge_4],
           dim=0).to(device)
      186      return x_interior, x_boundary
      187  
      188 -# 2. 主训练函数                                               
      188 +                                                              
      189 +# =========================================================== 
          +================                                              
      190 +# Training                                                    
      191 +# =========================================================== 
          +================                                              
      192  def train_pinn(solver, epochs, N_int=2000, N_bc=400,
           lr=1e-3):
      193      optimizer = optim.Adam(solver.net.parameters(), lr=lr)
      194 -                                                              
      195 -    # 记录 Loss 历史，用于绘制收敛曲线                        
      194      history = {'total': [], 'pde1': [], 'pde2': [], 'bc': []}
      195 -                                                              
      195 +                                                              
      196      solver.net.train()
      197      for epoch in range(epochs):
      198          optimizer.zero_grad()
      199 -                                                              
      200 -        # 每次迭代重新采样，增强泛化能力                      
      199          x_int, x_bc = sample_points(N_int, N_bc,
           solver.device)
      200 -                                                              
      200          total_loss, loss_pde1, loss_pde2, loss_bc =
           solver.compute_loss(x_int, x_bc)
      201          total_loss.backward()
      202          optimizer.step()
      203 -                                                              
      204 -        # 记录数据                                            
      203 +                                                              
      204          history['total'].append(total_loss.item())
      205          history['pde1'].append(loss_pde1.item())
      206          history['pde2'].append(loss_pde2.item())
      207          history['bc'].append(loss_bc.item())
      208 -                                                              
      208 +                                                              
      209          if epoch % 1000 == 0:
      210 -            print(f"Epoch {epoch:05d} | Total Loss            
          -: {total_loss.item():.4e}                                     
          -| PDE1: {loss_pde1.item():.4e} | PDE2: {loss_pde2.item():.4e} 
          -| BC: {loss_bc.item():.4e}")                                  
      211 -                                                              
      210 +            print(f"  Epoch {epoch:05d} | Total               
          +: {total_loss.item():.4e} "                                   
      211 +                  f"| PDE1: {loss_pde1.item():.4e} | PDE2:    
          +{loss_pde2.item():.4e} "                                      
      212 +                  f"| BC: {loss_bc.item():.4e}")              
      213 +                                                              
      214      return history
      215  
      216 +                                                              
      217 +# =========================================================== 
          +================                                              
      218 +# Enhanced Evaluation                                         
      219 +# =========================================================== 
          +================                                              
      220  def evaluate_model(solver, mms, resolution=100):
      221 +    """Return predictions, exact solutions, pointwise errors, 
          + and L2 errors for both y and p."""                           
      222      solver.net.eval()
      223 -                                                              
      224 -    # 生成测试网格                                            
      223 +                                                              
      224      x1 = torch.linspace(0, 1, resolution)
      225      x2 = torch.linspace(0, 1, resolution)
      226      X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
      227      x_test = torch.stack([X1.flatten(), X2.flatten()],
           dim=-1).to(solver.device)
      228 -                                                              
      228 +                                                              
      229      with torch.no_grad():
      230 -        # 获取网络预测值                                      
      230          raw_y, raw_p = solver.forward_eval(x_test)
      231 -                                                              
      232 -        # 极度关键：如果是缩放系统                            
          -(1.5式)，必须反向还原为真实物理场                             
      231 +                                                              
      232 +        # Unscale if using scaled system                      
      233          if solver.system_type == 'scaled':
      234              y_pred = raw_y * (solver.alpha ** -0.25)
      235              p_pred = raw_p * (solver.alpha ** 0.25)
      236          else:
      237              y_pred = raw_y
      238              p_pred = raw_p
      239 -                                                              
      240 -        # 获取解析解真值                                      
      239 +                                                              
      240          y_exact = mms.exact_y(x_test)
      241          p_exact = mms.exact_p(x_test)
      242 -                                                              
      243 -        # 计算相对 L2 误差                                    
      244 -        error_y = torch.sqrt(torch.sum((y_pred - y_exact)**2) 
          - / torch.sum(y_exact**2))                                     
      245 -        error_p = torch.sqrt(torch.sum((p_pred - p_exact)**2) 
          - / torch.sum(p_exact**2))                                     
      246 -                                                              
      247 -    print(f"[{solver.system_type.upper()} +                   
          -{solver.bc_type.upper()} BC] Relative L2 Error - y:           
          -{error_y.item():.4e}, p: {error_p.item():.4e}")               
      248 -                                                              
      249 -    return X1.numpy(), X2.numpy(), y_pred.reshape(resolution, 
          - resolution).cpu().numpy(), y_exact.reshape(resolution,       
          -resolution).cpu().numpy()                                     
      242  
      243 -# 初始化参数                                                  
      244 -alpha_test = 1e-4  # 设定一个较小的 alpha，激发 1.4           
          -式的不稳定性                                                  
      245 -epochs = 5000                                                 
      246 -mms = ManufacturedSolution(alpha_test, device)                
      243 +        error_y_l2 = torch.sqrt(torch.sum((y_pred -           
          +y_exact)**2) / torch.sum(y_exact**2)).item()                  
      244 +        error_p_l2 = torch.sqrt(torch.sum((p_pred -           
          +p_exact)**2) / torch.sum(p_exact**2)).item()                  
      245  
      246 -# 2x2 实验组别配置                                            
      247 -experiments = [                                               
      248 -    {'system': 'unscaled', 'bc': 'soft'},                     
      249 -    {'system': 'unscaled', 'bc': 'hard'},                     
      250 -    {'system': 'scaled', 'bc': 'soft'},                       
      251 -    {'system': 'scaled', 'bc': 'hard'}                        
      246 +    label = f"{solver.system_type.upper()} +                  
          +{solver.bc_type.upper()} BC"                                  
      247 +    print(f"  [{label}] Relative L2 Error — y:                
          +{error_y_l2:.4e}, p: {error_p_l2:.4e}")                       
      248 +                                                              
      249 +    res = resolution                                          
      250 +    return {                                                  
      251 +        'X1': X1.numpy(),                                     
      252 +        'X2': X2.numpy(),                                     
      253 +        'y_pred': y_pred.reshape(res, res).cpu().numpy(),     
      254 +        'p_pred': p_pred.reshape(res, res).cpu().numpy(),     
      255 +        'y_exact': y_exact.reshape(res, res).cpu().numpy(),   
      256 +        'p_exact': p_exact.reshape(res, res).cpu().numpy(),   
      257 +        'error_y_grid': np.abs(y_pred.reshape(res,            
          +res).cpu().numpy()                                            
      258 +                               - y_exact.reshape(res,         
          +res).cpu().numpy()),                                          
      259 +        'error_p_grid': np.abs(p_pred.reshape(res,            
          +res).cpu().numpy()                                            
      260 +                               - p_exact.reshape(res,         
          +res).cpu().numpy()),                                          
      261 +        'error_y_l2': error_y_l2,                             
      262 +        'error_p_l2': error_p_l2,                             
      263 +    }                                                         
      264 +                                                              
      265 +                                                              
      266 +# =========================================================== 
          +================                                              
      267 +# Visualization: Loss Evolution (2x2 subplot)                 
      268 +# =========================================================== 
          +================                                              
      269 +EXPERIMENT_KEYS = ['unscaled_soft', 'unscaled_hard',          
          +'scaled_soft', 'scaled_hard']                                 
      270 +EXPERIMENT_TITLES = [                                         
      271 +    '(A) Unscaled + Soft BC', '(B) Unscaled + Hard BC',       
      272 +    '(C) Scaled + Soft BC',   '(D) Scaled + Hard BC',         
      273  ]
      274  
      275 -results = {}                                                  
      275  
      276 -# 依次运行 4 组实验                                           
      277 -for config in experiments:                                    
      278 -    sys_type, bc_type = config['system'], config['bc']        
      279 -    print(f"\n{'='*50}")                                      
      280 -    print(f"Running Experiment: System = {sys_type}, BC =     
          -{bc_type}")                                                   
      281 -    print(f"{'='*50}")                                        
      282 -                                                              
      283 -    # 实例化求解器                                            
      284 -    solver = OptimalControlSolver(sys_type, bc_type,          
          -alpha_test, mms, device)                                      
      285 -                                                              
      286 -    # 开始训练                                                
      287 -    history = train_pinn(solver, epochs=epochs)               
      288 -                                                              
      289 -    # 评估精度并获取画图数据                                  
      290 -    X1, X2, y_pred_grid, y_exact_grid =                       
          -evaluate_model(solver, mms)                                   
      291 -                                                              
      292 -    results[f"{sys_type}_{bc_type}"] = {                      
      293 -        'history': history,                                   
      294 -        'error_grid': np.abs(y_pred_grid - y_exact_grid)      
      276 +def plot_loss_evolution(results, alpha,                       
          +filename='ablation_loss_evolution.png'):                      
      277 +    """2x2 subplot: each panel shows total/pde1/pde2/bc loss  
          +curves (semilogy)."""                                         
      278 +    fig, axes = plt.subplots(2, 2, figsize=(14, 10))          
      279 +    fig.suptitle(f'Loss Component Evolution                   
          +($\\alpha={alpha}$)', fontsize=16)                            
      280 +                                                              
      281 +    for idx, ax in enumerate(axes.flatten()):                 
      282 +        key = EXPERIMENT_KEYS[idx]                            
      283 +        hist = results[key]['history']                        
      284 +        epochs_arr = np.arange(1, len(hist['total']) + 1)     
      285 +                                                              
      286 +        ax.semilogy(epochs_arr, hist['total'], label='Total', 
          + color='black', lw=1.5)                                       
      287 +        ax.semilogy(epochs_arr, hist['pde1'],  label='PDE1    
          +(state)',   color='steelblue', alpha=0.8)                     
      288 +        ax.semilogy(epochs_arr, hist['pde2'],  label='PDE2    
          +(adjoint)', color='darkorange', alpha=0.8)                    
      289 +        if max(hist['bc']) > 0:                               
      290 +            ax.semilogy(epochs_arr, hist['bc'], label='BC',   
          +color='forestgreen', alpha=0.8)                               
      291 +                                                              
      292 +        ax.set_title(EXPERIMENT_TITLES[idx])                  
      293 +        ax.set_xlabel('Epoch')                                
      294 +        ax.set_ylabel('Loss')                                 
      295 +        ax.legend(fontsize=8)                                 
      296 +        ax.grid(True, which='both', alpha=0.3)                
      297 +                                                              
      298 +    plt.tight_layout()                                        
      299 +    plt.savefig(filename, dpi=150)                            
      300 +    plt.close()                                               
      301 +    print(f"Saved: {filename}")                               
      302 +                                                              
      303 +                                                              
      304 +# =========================================================== 
          +================                                              
      305 +# Visualization: Error Heatmaps (2x2 subplot, one figure per  
          +variable)                                                     
      306 +# =========================================================== 
          +================                                              
      307 +def plot_error_heatmaps(results, alpha, variable='y',         
      308 +                                                              
          +filename='ablation_error_heatmaps_y.png'):                    
      309 +    """2x2 subplot of pointwise absolute error heatmaps for y 
          + or p."""                                                     
      310 +    fig, axes = plt.subplots(2, 2, figsize=(12, 10))          
      311 +    var_label = r'$\overline{y}$' if variable == 'y' else     
          +r'$\overline{p}$'                                             
      312 +    fig.suptitle(f'Point-wise Absolute Error of {var_label}   
          +($\\alpha={alpha}$)', fontsize=16)                            
      313 +                                                              
      314 +    error_key = f'error_{variable}_grid'                      
      315 +                                                              
      316 +    for idx, ax in enumerate(axes.flatten()):                 
      317 +        key = EXPERIMENT_KEYS[idx]                            
      318 +        eval_data = results[key]['eval']                      
      319 +        X1, X2 = eval_data['X1'], eval_data['X2']             
      320 +        error_grid = eval_data[error_key]                     
      321 +                                                              
      322 +        im = ax.pcolormesh(X1, X2, error_grid, cmap='jet',    
          +shading='auto')                                               
      323 +        ax.set_title(EXPERIMENT_TITLES[idx])                  
      324 +        ax.set_xlabel('$x_1$')                                
      325 +        ax.set_ylabel('$x_2$')                                
      326 +        ax.set_aspect('equal')                                
      327 +        fig.colorbar(im, ax=ax, format='%.1e')                
      328 +                                                              
      329 +    plt.tight_layout()                                        
      330 +    plt.savefig(filename, dpi=150)                            
      331 +    plt.close()                                               
      332 +    print(f"Saved: {filename}")                               
      333 +                                                              
      334 +                                                              
      335 +# =========================================================== 
          +================                                              
      336 +# Visualization: Alpha Sensitivity (dual panel)               
      337 +# =========================================================== 
          +================                                              
      338 +def plot_alpha_sensitivity(sweep_results,                     
          +filename='alpha_sensitivity.png'):                            
      339 +    """Dual-panel log-log plot: relative L2 error vs alpha    
          +for all 4 groups."""                                          
      340 +    fig, axes = plt.subplots(1, 2, figsize=(14, 5))           
      341 +    fig.suptitle(r'$\alpha$-Sensitivity Analysis: Relative    
          +$L^2$ Error', fontsize=14)                                    
      342 +                                                              
      343 +    colors = {                                                
      344 +        'unscaled_soft': 'tomato',                            
      345 +        'unscaled_hard': 'salmon',                            
      346 +        'scaled_soft':   'steelblue',                         
      347 +        'scaled_hard':   'navy',                              
      348      }
      349 +    labels = {                                                
      350 +        'unscaled_soft': 'Unscaled + Soft BC',                
      351 +        'unscaled_hard': 'Unscaled + Hard BC',                
      352 +        'scaled_soft':   'Scaled + Soft BC',                  
      353 +        'scaled_hard':   'Scaled + Hard BC',                  
      354 +    }                                                         
      355  
      356 -# 绘制点对点绝对误差热力图 (Point-wise Absolute Error)        
      357 -fig, axes = plt.subplots(2, 2, figsize=(12, 10))              
      358 -fig.suptitle(f'Point-wise Absolute Error of $\overline{{y}}$  
          -($\\alpha={alpha_test}$)', fontsize=16)                       
      356 +    alphas = sorted(sweep_results.keys())                     
      357  
      358 -titles = ['(A) Unscaled + Soft BC', '(B) Unscaled + Hard BC', 
          -                                                              
      359 -          '(C) Scaled + Soft BC', '(D) Scaled + Hard BC']     
      360 -keys = ['unscaled_soft', 'unscaled_hard', 'scaled_soft',      
          -'scaled_hard']                                                
      358 +    for var_idx, (ax, var_name, var_label) in enumerate(zip(  
      359 +            axes,                                             
      360 +            ['error_y_l2', 'error_p_l2'],                     
      361 +            [r'$\overline{y}$', r'$\overline{p}$'])):         
      362  
      363 -for i, ax in enumerate(axes.flatten()):                       
      364 -    key = keys[i]                                             
      365 -    error_grid = results[key]['error_grid']                   
      366 -                                                              
      367 -    # 绘制热力图                                              
      368 -    im = ax.pcolormesh(X1, X2, error_grid, cmap='jet',        
          -shading='auto')                                               
      369 -    ax.set_title(titles[i])                                   
      370 -    ax.set_xlabel('$x_1$')                                    
      371 -    ax.set_ylabel('$x_2$')                                    
      372 -    fig.colorbar(im, ax=ax, format='%.1e')                    
      363 +        for key in EXPERIMENT_KEYS:                           
      364 +            errors = [sweep_results[a][key]['eval'][var_name] 
          + for a in alphas]                                             
      365 +            ax.loglog(alphas, errors, 'o-',                   
          +color=colors[key], label=labels[key], lw=1.5)                 
      366  
      367 -plt.tight_layout()                                            
      368 -plt.show()                                                    
      367   No newline at end of file
      368 +        ax.set_xlabel(r'$\alpha$')                            
      369 +        ax.set_ylabel(f'Relative $L^2$ Error of {var_label}') 
      370 +        ax.set_title(f'Error of {var_label}')                 
      371 +        ax.legend(fontsize=8)                                 
      372 +        ax.grid(True, which='both', alpha=0.3)                
      373 +        ax.invert_xaxis()  # smaller alpha on the right       
      374 +                                                              
      375 +    plt.tight_layout()                                        
      376 +    plt.savefig(filename, dpi=150)                            
      377 +    plt.close()                                               
      378 +    print(f"Saved: {filename}")                               
      379 +                                                              
      380 +                                                              
      381 +# =========================================================== 
          +================                                              
      382 +# Orchestration: run 2x2 ablation for a single alpha          
      383 +# =========================================================== 
          +================                                              
      384 +def run_2x2_ablation(alpha, epochs, seed=42):                 
      385 +    """Run all 4 experiment groups for a given alpha. Returns 
          + results dict."""                                             
      386 +    mms = ManufacturedSolution(alpha, device)                 
      387 +    experiments = [                                           
      388 +        ('unscaled', 'soft'),                                 
      389 +        ('unscaled', 'hard'),                                 
      390 +        ('scaled',   'soft'),                                 
      391 +        ('scaled',   'hard'),                                 
      392 +    ]                                                         
      393 +    results = {}                                              
      394 +                                                              
      395 +    for sys_type, bc_type in experiments:                     
      396 +        key = f"{sys_type}_{bc_type}"                         
      397 +        print(f"\n{'='*60}")                                  
      398 +        print(f"  α={alpha:.0e} | System={sys_type} |         
          +BC={bc_type}")                                                
      399 +        print(f"{'='*60}")                                    
      400 +                                                              
      401 +        # Fixed seed for reproducibility                      
      402 +        torch.manual_seed(seed)                               
      403 +        np.random.seed(seed)                                  
      404 +                                                              
      405 +        solver = OptimalControlSolver(sys_type, bc_type,      
          +alpha, mms, device)                                           
      406 +        t0 = time.time()                                      
      407 +        history = train_pinn(solver, epochs=epochs)           
      408 +        elapsed = time.time() - t0                            
      409 +        print(f"  Training time: {elapsed:.1f}s")             
      410 +                                                              
      411 +        eval_data = evaluate_model(solver, mms)               
      412 +        results[key] = {                                      
      413 +            'history': history,                               
      414 +            'eval': eval_data,                                
      415 +            'time': elapsed,                                  
      416 +        }                                                     
      417 +                                                              
      418 +    return results                                            
      419 +                                                              
      420 +                                                              
      421 +# =========================================================== 
          +================                                              
      422 +# Orchestration: alpha sweep                                  
      423 +# =========================================================== 
          +================                                              
      424 +def run_alpha_sweep(alpha_list, epochs, seed=42):             
      425 +    """Run 2x2 ablation for each alpha. Returns {alpha:       
          +results}."""                                                  
      426 +    sweep = {}                                                
      427 +    for alpha in alpha_list:                                  
      428 +        print(f"\n{'#'*70}")                                  
      429 +        print(f"#  Alpha Sweep: α = {alpha:.0e}")             
      430 +        print(f"{'#'*70}")                                    
      431 +        sweep[alpha] = run_2x2_ablation(alpha, epochs,        
          +seed=seed)                                                    
      432 +    return sweep                                              
      433 +                                                              
      434 +                                                              
      435 +# =========================================================== 
          +================                                              
      436 +# Main                                                        
      437 +# =========================================================== 
          +================                                              
      438 +def main():                                                   
      439 +    # ------------------------------------------------------- 
          +---------                                                     
      440 +    # Part 1: 2x2 Ablation at α = 1e-4                        
      441 +    # ------------------------------------------------------- 
          +---------                                                     
      442 +    alpha_main = 1e-4                                         
      443 +    epochs_main = 5000                                        
      444 +                                                              
      445 +    print("\n" + "=" * 70)                                    
      446 +    print("  PART 1: 2x2 Ablation Study (α = {:.0e}, {}       
          +epochs)".format(alpha_main, epochs_main))                     
      447 +    print("=" * 70)                                           
      448 +                                                              
      449 +    results = run_2x2_ablation(alpha_main, epochs_main)       
      450 +                                                              
      451 +    # Generate loss evolution and error heatmap figures       
      452 +    plot_loss_evolution(results, alpha_main)                  
      453 +    plot_error_heatmaps(results, alpha_main, variable='y',    
      454 +                                                              
          +filename='ablation_error_heatmaps_y.png')                     
      455 +    plot_error_heatmaps(results, alpha_main, variable='p',    
      456 +                                                              
          +filename='ablation_error_heatmaps_p.png')                     
      457 +                                                              
      458 +    # Summary table for Part 1                                
      459 +    print("\n" + "=" * 70)                                    
      460 +    print("  PART 1 SUMMARY (α = {:.0e})".format(alpha_main)) 
      461 +    print("=" * 70)                                           
      462 +    print(f"{'Config':<25s} {'err_y (L2)':>12s} {'err_p       
          +(L2)':>12s} {'Time (s)':>10s}")                               
      463 +    print("-" * 60)                                           
      464 +    for key in EXPERIMENT_KEYS:                               
      465 +        r = results[key]                                      
      466 +        print(f"  {key:<23s} {r['eval']['error_y_l2']:>12.4e} 
          + "                                                            
      467 +              f"{r['eval']['error_p_l2']:>12.4e}              
          +{r['time']:>10.1f}")                                          
      468 +                                                              
      469 +    # ------------------------------------------------------- 
          +---------                                                     
      470 +    # Part 2: Alpha Sensitivity Sweep                         
      471 +    # ------------------------------------------------------- 
          +---------                                                     
      472 +    alpha_list = [1e-2, 1e-3, 1e-4, 1e-5]                     
      473 +    epochs_sweep = 5000                                       
      474 +                                                              
      475 +    print("\n" + "=" * 70)                                    
      476 +    print("  PART 2: α-Sensitivity Sweep")                    
      477 +    print("=" * 70)                                           
      478 +                                                              
      479 +    sweep = run_alpha_sweep(alpha_list, epochs_sweep)         
      480 +                                                              
      481 +    # Reuse Part 1 results for α=1e-4 to avoid redundant      
          +training                                                      
      482 +    sweep[alpha_main] = results                               
      483 +                                                              
      484 +    plot_alpha_sensitivity(sweep)                             
      485 +                                                              
      486 +    # Summary table for Part 2                                
      487 +    print("\n" + "=" * 70)                                    
      488 +    print("  PART 2 SUMMARY: α-Sensitivity")                  
      489 +    print("=" * 70)                                           
      490 +    for alpha in sorted(alpha_list):                          
      491 +        print(f"\n  α = {alpha:.0e}")                         
      492 +        print(f"  {'Config':<25s} {'err_y':>12s}              
          +{'err_p':>12s}")                                              
      493 +        print(f"  {'-'*50}")                                  
      494 +        for key in EXPERIMENT_KEYS:                           
      495 +            ey = sweep[alpha][key]['eval']['error_y_l2']      
      496 +            ep = sweep[alpha][key]['eval']['error_p_l2']      
      497 +            print(f"    {key:<23s} {ey:>12.4e} {ep:>12.4e}")  
      498 +                                                              
      499 +    print("\n\nDone. Generated files:")                       
      500 +    print("  ablation_loss_evolution.png")                    
      501 +    print("  ablation_error_heatmaps_y.png")                  
      502 +    print("  ablation_error_heatmaps_p.png")                  
      503 +    print("  alpha_sensitivity.png")                          
      504 +                                                              
      505 +                                                              
      506 +if __name__ == "__main__":                                    
      507 +    main()           
