#!/usr/bin/env python3
"""
이미지 기반 모방학습 (CUDA 강제)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

print("=== Vision Imitation Learning ===\n")

DATA_ROOT = Path("~/lerobot_vision_data").expanduser()
CHECKPOINT_DIR = Path("~/handkerchief_checkpoints").expanduser()
CHECKPOINT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224


# CUDA 체크
print("[CUDA 확인]")
print(f"  PyTorch 버전: {torch.__version__}")
print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA 버전: {torch.version.cuda}")
    print(f"  GPU 개수: {torch.cuda.device_count()}")
    print(f"  GPU 이름: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
    print(f"  ✓ 디바이스: cuda\n")
else:
    print("  ✗ CUDA를 사용할 수 없습니다!")
    print("  해결방법:")
    print("    1. nvidia-smi 실행 확인")
    print("    2. pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("    3. CUDA 드라이버 설치 확인\n")
    
    choice = input("CPU로 계속하시겠습니까? (y/n): ").strip().lower()
    if choice != 'y':
        print("종료합니다.")
        sys.exit(1)
    
    device = torch.device("cpu")
    print("  ⚠ 디바이스: cpu (느립니다!)\n")


class VisionDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.episodes = sorted([d for d in self.data_root.glob("episode_*") if d.is_dir()])
        
        print(f"[Dataset] {len(self.episodes)}개 에피소드")
        
        self.samples = []
        
        for ep_dir in self.episodes:
            qpos = np.load(ep_dir / "qpos.npy")
            qvel = np.load(ep_dir / "qvel.npy")
            frames_dir = ep_dir / "frames"
            
            frame_files = sorted(frames_dir.glob("*.jpg"))
            
            for i in range(len(qpos) - 1):
                if i < len(frame_files):
                    self.samples.append({
                        'frame_path': frame_files[i],
                        'qpos': qpos[i],
                        'qvel': qvel[i],
                        'action': qpos[i + 1],
                    })
        
        self.total_frames = len(self.samples)
        print(f"[Dataset] 총 {self.total_frames}개 샘플\n")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드 및 전처리
        img = cv2.imread(str(sample['frame_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        # 상태
        state = np.concatenate([sample['qpos'], sample['qvel']])
        
        return {
            'image': torch.from_numpy(img),
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(sample['action']),
        }


class VisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN for image
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
        )
        
        self.img_feat_size = 128 * 7 * 7
        
        # MLP for joint state
        self.state_encoder = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
        # Combined decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.img_feat_size + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 12),
        )
    
    def forward(self, image, state):
        img_feat = self.cnn(image)
        state_feat = self.state_encoder(state)
        
        combined = torch.cat([img_feat, state_feat], dim=1)
        action = self.decoder(combined)
        
        return action


def train():
    dataset = VisionDataset(DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = VisionPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.SmoothL1Loss()
    
    print(f"[학습 시작]")
    print(f"  배치 크기: {BATCH_SIZE}")
    print(f"  에폭: {NUM_EPOCHS}")
    print(f"  학습률: {LEARNING_RATE}")
    print(f"  디바이스: {device}\n")
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch in progress:
            image = batch['image'].to(device, non_blocking=True)
            state = batch['state'].to(device, non_blocking=True)
            action_gt = batch['action'].to(device, non_blocking=True)
            
            action_pred = model(image, state)
            loss = criterion(action_pred, action_gt)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, CHECKPOINT_DIR / "vision_best_model.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: Loss = {avg_loss:.4f} (Best: {best_loss:.4f})")
    
    print(f"\n{'='*60}")
    print(f"✓ 학습 완료!")
    print(f"  최고 Loss: {best_loss:.4f}")
    print(f"  체크포인트: {CHECKPOINT_DIR / 'vision_best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
