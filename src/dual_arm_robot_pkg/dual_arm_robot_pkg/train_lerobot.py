#!/usr/bin/env python3
"""
LeRobot ACT 모델 학습
"""

import torch
import numpy as np
from pathlib import Path
import json
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("=== LeRobot ACT 학습 ===\n")

# 설정
DATA_ROOT = Path("~/lerobot_data").expanduser()
CHECKPOINT_DIR = Path("~/handkerchief_checkpoints").expanduser()
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
NUM_QUERIES = 100
HIDDEN_DIM = 512
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 7


class HandkerchiefDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.episodes = sorted([d for d in self.data_root.glob("episode_*") if d.is_dir()])
        
        print(f"[Dataset] {len(self.episodes)}개 에피소드 로드")
        
        # 각 에피소드의 프레임 수 저장
        self.episode_lengths = []
        for ep_dir in self.episodes:
            qpos = np.load(ep_dir / "qpos.npy")
            self.episode_lengths.append(len(qpos))
        
        self.total_frames = sum(self.episode_lengths)
        print(f"[Dataset] 총 {self.total_frames}개 프레임")
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        # idx를 에피소드 + 프레임 인덱스로 변환
        ep_idx = 0
        frame_idx = idx
        
        for i, ep_len in enumerate(self.episode_lengths):
            if frame_idx < ep_len:
                ep_idx = i
                break
            frame_idx -= ep_len
        
        ep_dir = self.episodes[ep_idx]
        
        # 데이터 로드
        qpos = np.load(ep_dir / "qpos.npy")
        qvel = np.load(ep_dir / "qvel.npy")
        
        # 현재 상태
        state = np.concatenate([qpos[frame_idx], qvel[frame_idx]])
        
        # 목표 동작 (다음 프레임)
        if frame_idx + 1 < len(qpos):
            action = qpos[frame_idx + 1]
        else:
            action = qpos[frame_idx]  # 마지막 프레임
        
        return {
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action),
        }


class ACTPolicy(torch.nn.Module):
    """간단한 ACT 모델"""
    def __init__(self, state_dim=24, action_dim=12, hidden_dim=512):
        super().__init__()
        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, state):
        encoded = self.encoder(state)
        action = self.decoder(encoded)
        return action


def train():
    print("\n[학습 시작]\n")
    
    # 데이터셋
    dataset = HandkerchiefDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 모델
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}\n")
    
    model = ACTPolicy(state_dim=24, action_dim=12, hidden_dim=HIDDEN_DIM).to(device)
    
    # 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    
    # 학습
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in progress:
            state = batch['state'].to(device)
            action_gt = batch['action'].to(device)
            
            # Forward
            action_pred = model(state)
            loss = criterion(action_pred, action_gt)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # 체크포인트 저장
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"  ✓ 체크포인트 저장: {checkpoint_path}")
        
        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
    
    print(f"\n{'='*60}")
    print(f"✓ 학습 완료!")
    print(f"  최고 손실: {best_loss:.4f}")
    print(f"  체크포인트: {CHECKPOINT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
