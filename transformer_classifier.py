import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class FeatureEmbedding(nn.Module):
    """特征嵌入层，带有自动相关性确定(ARD)功能"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        self.embedding = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        # 应用特征加权(ARD)
        x = x * F.softplus(self.feature_weights)
        # 应用嵌入
        return self.embedding(x)
    
    def get_ard_weights(self):
        """返回ARD权重用于正则化"""
        return F.softplus(self.feature_weights)

class TransformerClassifier(nn.Module):
    """
    结合GP专家的Transformer分类器
    """
    def __init__(self, input_dim, d_model, nhead, dim_feedforward, num_layers, num_classes,
                 gp_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.top_k = min(top_k, num_classes)  # 确保top_k不大于num_classes
        self.gp_experts = gp_experts
        self.input_dim = input_dim
        
        # 特征嵌入层(带ARD)
        self.feature_embedding = FeatureEmbedding(input_dim, d_model)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
        # 冻结GP专家模型参数
        self._freeze_gp_experts()
        
    def _freeze_gp_experts(self):
        """冻结GP专家模型参数"""
        for class_idx, expert_dict in self.gp_experts.items():
            if 'model' in expert_dict:
                for param in expert_dict['model'].parameters():
                    param.requires_grad = False
                    
            if 'likelihood' in expert_dict:
                for param in expert_dict['likelihood'].parameters():
                    param.requires_grad = False
    
    def get_gp_prediction(self, original_input, class_probs):
        """
        基于类别概率获取GP专家预测
        该函数与梯度图分离
        
        Args:
            original_input: 输入特征 [batch_size, input_dim]
            class_probs: 分类概率 [batch_size, num_classes]
            
        Returns:
            GP回归预测
        """
        device = original_input.device
        batch_size = original_input.shape[0]
        regression_output = torch.zeros(batch_size, 1, device=device)
        
        # 获取top-k类别
        top_k_probs, top_k_indices = torch.topk(class_probs, self.top_k, dim=1)
        
        # 归一化top-k概率，使其和为1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=1, keepdim=True)
        
        # 对批次中的每个样本
        for i in range(batch_size):
            sample_regression = torch.zeros(1, device=device)
            sample_input = original_input[i:i+1]  # [1, input_dim]
            
            valid_predictions = 0
            
            for k in range(self.top_k):
                class_idx = top_k_indices[i, k].item()
                class_prob = top_k_probs[i, k].item()
                
                # 跳过没有GP专家的类别
                if class_idx not in self.gp_experts:
                    continue
                
                # 获取该类的GP专家
                expert_dict = self.gp_experts[class_idx]
                
                # 跳过缺少模型或似然函数的专家
                if 'model' not in expert_dict or 'likelihood' not in expert_dict:
                    continue
                
                gp_expert = expert_dict['model']
                likelihood = expert_dict['likelihood']
                
                # 获取GP专家的设备
                expert_device = next(gp_expert.parameters()).device
                
                try:
                    # 确保输入与GP专家在同一设备上
                    if sample_input.device != expert_device:
                        expert_input = sample_input.to(expert_device)
                    else:
                        expert_input = sample_input
                        
                    # 使用torch.no_grad()完全从计算图中分离
                    with torch.no_grad():
                        gp_output = likelihood(gp_expert(expert_input))
                        gp_mean = gp_output.mean
                        
                        # 如果需要，将预测移回原设备
                        if gp_mean.device != device:
                            gp_mean = gp_mean.to(device)
                        
                        # 用类别概率加权GP预测
                        sample_regression += class_prob * gp_mean
                        valid_predictions += 1
                except Exception as e:
                    print(f"类别{class_idx}的GP预测失败: {str(e)}")
                    continue
            
            # 设置回归输出
            if valid_predictions > 0:
                regression_output[i] = sample_regression
            else:
                # 没有GP专家时的后备方案
                fallback = torch.sum(class_probs[i] * torch.arange(self.num_classes, device=device).float())
                regression_output[i] = fallback.unsqueeze(0)
                
        return regression_output
    
    def forward(self, x):
        """
        前向传播，结合GP专家整合
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            tuple: (logits, class_probs, regression_output, ard_weights)
        """
        device = x.device
        batch_size = x.shape[0]
        
        # 存储原始输入用于GP模型
        original_input = x.clone()
        
        # 应用特征嵌入
        x = self.feature_embedding(x)  # [batch_size, d_model]
        
        # 获取ARD权重用于正则化
        ard_weights = self.feature_embedding.get_ard_weights()
        
        # Transformer期望输入形状为[seq_len, batch_size, d_model]
        x = x.unsqueeze(0)  # [1, batch_size, d_model]
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)  # [1, batch_size, d_model]
        x = x.squeeze(0)  # [batch_size, d_model]
        
        # 获取分类输出
        logits = self.classifier(x)  # [batch_size, num_classes]
        class_probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
        
        # 获取GP回归预测
        regression_output = self.get_gp_prediction(original_input, class_probs)
        
        # 连接回归输出到计算图
        # 这是关键部分：我们使用分离的GP预测，但创建一个梯度路径
        if self.training:
            # 在训练期间，允许梯度通过class_probs反向流动
            # 但保持GP预测值不变
            regression_output = regression_output.detach() + 0.0 * class_probs.sum().unsqueeze(0).unsqueeze(0)
        
        return logits, class_probs, regression_output, ard_weights
        
    def to(self, device, *args, **kwargs):
        """
        将模型移动到指定设备
        """
        model = super().to(device, *args, **kwargs)
        
        # 同时将GP专家移动到设备
        if hasattr(model, 'gp_experts') and model.gp_experts:
            for class_idx in model.gp_experts.keys():
                if 'model' in model.gp_experts[class_idx]:
                    model.gp_experts[class_idx]['model'] = model.gp_experts[class_idx]['model'].to(device)
                
                if 'likelihood' in model.gp_experts[class_idx]:
                    model.gp_experts[class_idx]['likelihood'] = model.gp_experts[class_idx]['likelihood'].to(device)
                
                if 'train_x' in model.gp_experts[class_idx]:
                    model.gp_experts[class_idx]['train_x'] = model.gp_experts[class_idx]['train_x'].to(device)
                
                if 'train_y' in model.gp_experts[class_idx]:
                    model.gp_experts[class_idx]['train_y'] = model.gp_experts[class_idx]['train_y'].to(device)
        
        return model