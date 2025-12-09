#!/usr/bin/env python3
"""
LeRobot ACT 모델 훈련
"""

import torch
from pathlib import Path
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def train():
    print("\n" + "="*60)
    print("LeRobot ACT 모델 훈련")
    print("="*60)
    
    # 데이터셋 경로
    dataset_path = Path("~/handkerchief_robot_dual_lerobot_dataset").expanduser()
    output_dir = Path("~/handkerchief_dual_training").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n데이터셋: {dataset_path}")
    print(f"출력: {output_dir}\n")
    
    # 데이터셋 로드
    print("데이터셋 로드 중...")
    dataset = LeRobotDataset(str(dataset_path))
    print(f"  ✓ {len(dataset)} 샘플")
    
    # 모델 생성
    print("\n모델 생성 중...")
    policy = ACTPolicy(
        state_dim=12,
        action_dim=12,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
    )
    
    if torch.cuda.is_available():
        policy = policy.cuda()
        print(f"  ✓ GPU 사용")
    
    # 훈련
    print("\n훈련 시작...")
    policy.train(
        dataset=dataset,
        num_epochs=100,
        batch_size=8,
        learning_rate=1e-4,
        output_dir=str(output_dir),
    )
    
    print(f"\n{'='*60}")
    print(f"✓ 훈련 완료!")
    print(f"{'='*60}")
    print(f"\n모델 저장 위치: {output_dir}")
    print(f"\n다음 단계: python autonomous.py\n")


if __name__ == "__main__":
    train()
