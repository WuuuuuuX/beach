import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
import time
from PIL import Image
import GPUtil
from torch.cuda.amp import autocast, GradScaler  # 添加混合精度支持

import warnings

warnings.filterwarnings("ignore")


# ===== DINOv2 特定模块 =====
def DINOv2Transform(size=224):
    """DINOv2专用的图像预处理流程"""
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_dinov2_feature_extractor(model_name='dinov2_vits14', device='cuda'):
    """获取DINOv2特征提取器"""
    model = torch.hub.load('facebookresearch/dinov2', model_name).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ===== 内存监控工具 =====
def print_gpu_usage(msg):
    """打印GPU内存使用情况"""
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print(f"{msg} - GPU {i}: {gpu.memoryUsed:.1f}/{gpu.memoryTotal:.1f} MB ({gpu.memoryUtil * 100:.1f}%)")
    if torch.cuda.is_available():
        print(f"{msg} - Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB, "
              f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")


# ===== 修改后的网络结构 =====
class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vits14', device='cuda'):
        super(DINOv2FeatureExtractor, self).__init__()
        self.device = device
        self.model = get_dinov2_feature_extractor(model_name, device)
        self.feature_dim = self.model.embed_dim  # DINOv2的特征维度

    def forward(self, x):
        """提取DINOv2特征，自动处理梯度控制"""
        # 使用 torch.no_grad() 替代 torch.inference_mode() 解决反向传播问题
        with torch.no_grad():
            features = self.model(x)
        # 确保返回float32（与后续层兼容）
        return features.float()


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=512,
                 device='cuda', model_name='dinov2_vits14'):
        super(ResCNNEncoder, self).__init__()

        # DINOv2特征提取器
        self.feature_extractor = DINOv2FeatureExtractor(model_name, device)
        self.embed_dim = self.feature_extractor.feature_dim

        # 简化后的特征转换层
        self.fc1 = nn.Linear(self.embed_dim, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, CNN_embed_dim)
        self.bn2 = nn.BatchNorm1d(CNN_embed_dim, momentum=0.01)
        self.drop_p = drop_p

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # 提取DINOv2特征
            raw_features = self.feature_extractor(x_3d[:, t])

            # 特征转换
            x = self.bn1(self.fc1(raw_features))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.bn2(self.fc2(x))
            cnn_embed_seq.append(x)

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose(0, 1)
        return cnn_embed_seq


# DecoderRNN保持不变
class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=512, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()
        self.LSTM = nn.LSTM(input_size=CNN_embed_dim,
                            hidden_size=h_RNN,
                            num_layers=h_RNN_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(h_RNN, h_FC_dim)
        self.fc2 = nn.Linear(h_FC_dim, num_classes)
        self.drop_p = drop_p

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, _ = self.LSTM(x_RNN, None)
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


# ===== 训练流程优化 =====
def train(log_interval, model, device, train_loader, optimizer, epoch, class_weights, scaler):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0
    total_batches = len(train_loader)

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device).view(-1)

        # 梯度清零放在此处减少显存占用
        optimizer.zero_grad()

        # 使用混合精度训练
        with autocast():
            # 特征提取和序列处理
            features = cnn_encoder(X)
            output = rnn_decoder(features)

            # 计算损失
            loss = F.cross_entropy(output, y, weight=class_weights.to(device))

        # 反向传播（使用梯度缩放）
        scaler.scale(loss).backward()

        # 取消缩放梯度用于裁剪
        scaler.unscale_(optimizer)

        # 梯度裁剪防止爆炸
        torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(rnn_decoder.parameters(), 5.0)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()

        # 记录损失
        losses.append(loss.item())

        # 计算准确率（不使用混合精度）
        with torch.no_grad():
            y_pred = torch.max(output, 1)[1]
            step_score = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
            scores.append(step_score)

        N_count += X.size(0)

        # 打印进度和资源使用
        if (batch_idx + 1) % log_interval == 0:
            print(f'Epoch: {epoch + 1} [{N_count}/{len(train_loader.dataset)} '
                  f'({100. * (batch_idx + 1) / total_batches:.0f}%)]\tLoss: {loss.item():.6f}, Accu: {100 * step_score:.2f}%')
            print_gpu_usage("After batch")

    return losses, scores


# 验证函数保持不变，但添加资源监控
def validation(model, device, test_loader, class_weights):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []

    # 使用梯度关闭减少显存使用
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1)

            # 特征提取和序列处理
            features = cnn_encoder(X)
            output = rnn_decoder(features)

            # 计算损失和预测
            loss = F.cross_entropy(output, y, weight=class_weights.to(device), reduction='sum')
            test_loss += loss.item()
            y_pred = output.max(1, keepdim=True)[1]

            # 收集结果
            all_y.extend(y.cpu())
            all_y_pred.extend(y_pred.squeeze().cpu())

    # 计算平均损失和准确率
    test_loss /= len(test_loader.dataset)
    all_y = torch.tensor(all_y)
    all_y_pred = torch.tensor(all_y_pred)
    test_score = (all_y == all_y_pred).float().mean().item()

    print(f'\nValid set ({len(all_y)} samples): Avg loss: {test_loss:.4f}, Acc: {100 * test_score:.2f}%')
    print_gpu_usage("After validation")

    return test_loss, test_score


# 模型保存/加载函数保持不变
def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(cnn_encoder, rnn_decoder, optimizer, scheduler, scaler, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    cnn_encoder.load_state_dict(checkpoint['cnn_encoder_state_dict'])
    rnn_decoder.load_state_dict(checkpoint['rnn_decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_valid_score = checkpoint['best_valid_score']
    no_improve_count = checkpoint.get('no_improve_count', 0)
    return start_epoch, best_valid_score, no_improve_count


# ===== 主训练流程 =====
if __name__ == "__main__":
    random_seeds = [96, 312]
    data_path = "D:/Mypython/Data/NC_WB_9000/WB_Set B_jpg_test1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    for seed in random_seeds:
        print(f"\n=== 开始训练，随机种子: {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # 设置为True提高效率

        save_path = f"./TGRS_result_baseline1/output_dinov2_seed{seed}"
        save_model_path = os.path.join(save_path, "models")
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_model_path, exist_ok=True)

        # 配置参数
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
        CNN_embed_dim = 512
        res_size = 224
        dropout_p = 0.3
        RNN_hidden_layers = 3
        RNN_hidden_nodes = 512
        RNN_FC_dim = 256
        k = 3
        epochs = 100
        batch_size = 50  # 进一步减少批大小以适应DINOv2内存需求
        learning_rate = 1e-4  # 降低学习率
        log_interval = 10
        patience = 20
        begin_frame, end_frame, skip_frame = 1, 60, 1
        window_size = 5

        use_cuda = torch.cuda.is_available()
        params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {
            'batch_size': batch_size, 'shuffle': True}

        action_names = ['Plunging', 'Spilling', 'Surging']
        le = LabelEncoder()
        le.fit(action_names)

        fnames = os.listdir(data_path)
        actions, all_names = [], []
        for f in fnames:
            loc = f.find('_')
            if loc == -1:
                print(f"Unexpected file format: {f}")
                continue
            actions.append(f[:loc])
            all_names.append(f)

        all_X_list = all_names
        all_y_list = labels2cat(le, actions)

        # 数据集划分
        indices = list(range(len(all_X_list)))
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices = indices[:split]
        valid_indices = indices[split:]

        # 类别权重处理
        train_labels = [all_y_list[i] for i in train_indices]
        train_labels_np = np.array(train_labels)
        class_sample_counts = np.array([(train_labels_np == i).sum() for i in range(k)])
        class_weights = 1. / (class_sample_counts + 1e-6)
        class_weights = class_weights * k / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        # DINOv2专用预处理
        transform = DINOv2Transform(res_size)
        selected_frames = list(range(begin_frame, end_frame, skip_frame))

        # 数据集
        train_set = Dataset_CRNN(data_path, [all_X_list[i] for i in train_indices],
                                 [all_y_list[i] for i in train_indices], selected_frames, transform=transform)
        valid_set = Dataset_CRNN(data_path, [all_X_list[i] for i in valid_indices],
                                 [all_y_list[i] for i in valid_indices], selected_frames, transform=transform)

        # 数据加载器
        train_loader = DataLoader(train_set, **params)
        valid_loader = DataLoader(valid_set, **params)

        # 创建模型
        print("创建模型中...")
        cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                                    drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim,
                                    device=device).to(device)
        rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers,
                                 h_RNN=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim,
                                 drop_p=dropout_p, num_classes=k).to(device)

        # 模型并行
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} GPU!")
            cnn_encoder = nn.DataParallel(cnn_encoder)
            rnn_decoder = nn.DataParallel(rnn_decoder)
            crnn_params = list(cnn_encoder.module.parameters()) + list(rnn_decoder.parameters())
        else:
            print("使用单GPU!")
            crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())

        # 优化器配置
        optimizer = torch.optim.Adam(crnn_params, lr=learning_rate, weight_decay=1e-5)  # 添加权重衰减
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)

        # 创建梯度缩放器用于混合精度训练
        scaler = GradScaler()

        checkpoint_path = os.path.join(save_model_path, 'checkpoint.pth')

        train_losses, train_scores, valid_losses, valid_scores = [], [], [], []

        if os.path.exists(checkpoint_path):
            print(f"加载检查点: {checkpoint_path}")
            start_epoch, best_valid_score, no_improve_count = load_checkpoint(
                cnn_encoder, rnn_decoder, optimizer, scheduler, scaler, checkpoint_path, device)

            # 恢复指标
            for key, lst in zip(['train_losses', 'train_scores', 'valid_losses', 'valid_scores'],
                                [train_losses, train_scores, valid_losses, valid_scores]):
                fpath = os.path.join(save_path, f"{key}_seed{seed}.npy")
                if os.path.exists(fpath):
                    arr = np.load(fpath).tolist()
                    lst.extend(arr if isinstance(arr, list) else [arr])
        else:
            print("没有找到检查点，开始新的训练")
            start_epoch = 0
            best_valid_score = -float('inf')
            no_improve_count = 0

        # 训练循环
        for epoch in range(start_epoch, epochs):
            print_gpu_usage(f"Epoch {epoch + 1} 开始前")
            epoch_start_time = time.time()

            print(f"\n训练 Epoch {epoch + 1}/{epochs}")
            train_loss, train_score = train(log_interval, [cnn_encoder, rnn_decoder], device,
                                            train_loader, optimizer, epoch, class_weights, scaler)

            print(f"验证 Epoch {epoch + 1}/{epochs}")
            valid_loss, valid_score = validation([cnn_encoder, rnn_decoder], device, valid_loader, class_weights)

            # 更新学习率
            scheduler.step(valid_score)

            # 记录结果
            train_losses.append(np.mean(train_loss))
            train_scores.append(np.mean(train_score))
            valid_losses.append(valid_loss)
            valid_scores.append(valid_score)

            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'cnn_encoder_state_dict': cnn_encoder.state_dict(),
                'rnn_decoder_state_dict': rnn_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_valid_score': best_valid_score,
                'no_improve_count': no_improve_count
            }
            save_checkpoint(checkpoint, checkpoint_path)

            # 保存指标
            np.save(os.path.join(save_path, f'train_losses_seed{seed}.npy'), np.array(train_losses))
            np.save(os.path.join(save_path, f'train_scores_seed{seed}.npy'), np.array(train_scores))
            np.save(os.path.join(save_path, f'valid_losses_seed{seed}.npy'), np.array(valid_losses))
            np.save(os.path.join(save_path, f'valid_scores_seed{seed}.npy'), np.array(valid_scores))

            # 检查早停
            if epoch >= window_size - 1:
                window_avg = np.mean(valid_scores[-window_size:])
                if window_avg > best_valid_score:
                    print(f"验证准确率提升 ({best_valid_score:.4f} → {window_avg:.4f})")
                    best_valid_score = window_avg
                    no_improve_count = 0
                    # 保存最佳模型
                    torch.save(cnn_encoder.state_dict(),
                               os.path.join(save_model_path, f'best_cnn_encoder_seed{seed}.pth'))
                    torch.save(rnn_decoder.state_dict(),
                               os.path.join(save_model_path, f'best_rnn_decoder_seed{seed}.pth'))
                else:
                    no_improve_count += 1
                    print(f"无提升 {no_improve_count}/{patience}")

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1} 完成，耗时 {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s")

            if no_improve_count >= patience:
                print(f"连续{patience}轮无性能提升，提前结束训练")
                break

        # 最终测试
        print(f"\n=== 随机种子 {seed} 的最终测试 ===")
        cnn_encoder.load_state_dict(torch.load(os.path.join(save_model_path, f'best_cnn_encoder_seed{seed}.pth')))
        rnn_decoder.load_state_dict(torch.load(os.path.join(save_model_path, f'best_rnn_decoder_seed{seed}.pth')))

        test_loss, test_score = validation([cnn_encoder, rnn_decoder], device, valid_loader, class_weights)
        print(f"测试准确率: {test_score:.4f}")
        np.save(os.path.join(save_path, f'test_score_seed{seed}.npy'), np.array([test_score]))