#!/usr/bin/env python3
"""
LeRobot 학습 + TensorBoard 실시간 모니터링
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime

print("=== LeRobot 학습 (TensorBoard) ===\n")

DATA_ROOT = Path("~/lerobot_data").expanduser()
CHECKPOINT_DIR = Path("~/handkerchief_checkpoints").expanduser()
CHECKPOINT_DIR.mkdir(exist_ok=True)

# TensorBoard 로그 디렉토리
LOG_DIR = Path("~/handkerchief_logs").expanduser()
LOG_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 730
LEARNING_RATE = 5e-5
HIDDEN_DIM = 256


class HandkerchiefDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.episodes = sorted([d for d in self.data_root.glob("episode_*") if d.is_dir()])
        
        print(f"[Dataset] {len(self.episodes)}개 에피소드")
        
        self.episode_lengths = []
        for ep_dir in self.episodes:
            qpos = np.load(ep_dir / "qpos.npy")
            self.episode_lengths.append(len(qpos))
        
        self.total_frames = sum(self.episode_lengths)
        print(f"[Dataset] 총 {self.total_frames}개 프레임\n")
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        ep_idx = 0
        frame_idx = idx
        
        for i, ep_len in enumerate(self.episode_lengths):
            if frame_idx < ep_len:
                ep_idx = i
                break
            frame_idx -= ep_len
        
        ep_dir = self.episodes[ep_idx]
        
        qpos = np.load(ep_dir / "qpos.npy")
        qvel = np.load(ep_dir / "qvel.npy")
        
        state = np.concatenate([qpos[frame_idx], qvel[frame_idx]])
        
        if frame_idx + 1 < len(qpos):
            action = qpos[frame_idx + 1]
        else:
            action = qpos[frame_idx]
        
        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action),
        }


class ImprovedPolicy(nn.Module):
    def __init__(self, state_dim=24, action_dim=12, hidden_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        encoded = self.encoder(state)
        action = self.decoder(encoded)
        return action


def train():
    dataset = HandkerchiefDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}\n")
    
    model = ImprovedPolicy(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.SmoothL1Loss()
    
    # TensorBoard Writer
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(LOG_DIR / f"run_{timestamp}")
    
    print(f"TensorBoard 로그: {LOG_DIR / f'run_{timestamp}'}")
    print(f"실행 명령: tensorboard --logdir={LOG_DIR}\n")
    
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        batch_losses = []
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress):
            state = batch['state'].to(device)
            action_gt = batch['action'].to(device)
            
            action_pred = model(state)
            loss = criterion(action_pred, action_gt)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # 배치별 Loss 기록
            writer.add_scalar('Loss/batch', batch_loss, global_step)
            global_step += 1
            
            progress.set_postfix({'loss': f'{batch_loss:.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Epoch별 메트릭 기록
        writer.add_scalar('Loss/epoch', avg_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 예측 vs 정답 샘플 기록 (처음 배치)
        with torch.no_grad():
            sample_batch = next(iter(dataloader))
            sample_state = sample_batch['state'][:4].to(device)
            sample_action_gt = sample_batch['action'][:4].to(device)
            sample_action_pred = model(sample_state)
            
            for joint_idx in range(12):
                writer.add_scalars(f'Joint_{joint_idx}', {
                    'predicted': sample_action_pred[0][joint_idx].item(),
                    'ground_truth': sample_action_gt[0][joint_idx].item(),
                }, epoch)
        
        # 체크포인트 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, CHECKPOINT_DIR / "best_model.pth")
            
            writer.add_scalar('Best_Loss', best_loss, epoch)
        
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch+1}: Loss = {avg_loss:.4f} (Best: {best_loss:.4f})")
    
    writer.close()
    
    print(f"\n✓ 학습 완료! 최고 Loss: {best_loss:.4f}")
    print(f"  체크포인트: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"  TensorBoard 로그: {LOG_DIR / f'run_{timestamp}'}\n")


if __name__ == "__main__":
    train()
