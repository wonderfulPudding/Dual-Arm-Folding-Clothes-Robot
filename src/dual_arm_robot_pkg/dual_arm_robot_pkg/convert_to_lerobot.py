#!/usr/bin/env python3
"""
LeRobot 형식으로 변환
"""

import numpy as np
import json
import cv2
from pathlib import Path
import os

def convert_to_lerobot(data_root="~/handkerchief_robot_dual_data"):
    data_root = Path(data_root).expanduser()
    lerobot_root = data_root.parent / "lerobot_data"
    lerobot_root.mkdir(exist_ok=True)
    
    print("\n[LeRobot 변환]")
    print(f"입력: {data_root}")
    print(f"출력: {lerobot_root}")
    
    # 에피소드 디렉토리 찾기
    episodes = sorted([d for d in data_root.glob("episode_*") if d.is_dir()])
    
    if not episodes:
        print("✗ 에피소드가 없습니다!")
        return
    
    print(f"\n변환할 에피소드: {len(episodes)}개\n")
    
    for ep_idx, episode_dir in enumerate(episodes):
        obs_dir = episode_dir / "observations"
        
        print(f"[{ep_idx+1}/{len(episodes)}] {episode_dir.name} 변환 중...")
        
        # 데이터 로드
        follower1_pos = np.load(obs_dir / "follower1_positions.npy")
        follower2_pos = np.load(obs_dir / "follower2_positions.npy")
        
        # 결합 (12개 DOF)
        qpos = np.concatenate([follower1_pos, follower2_pos], axis=1)  # (N, 12)
        
        print(f"  위치 데이터: {qpos.shape}")
        
        # qvel 계산 (속도 = 다음 위치 - 현재 위치)
        qvel = np.diff(qpos, axis=0, prepend=qpos[0:1])
        
        # LeRobot 형식으로 저장
        ep_save_dir = lerobot_root / f"episode_{ep_idx:06d}"
        ep_save_dir.mkdir(exist_ok=True)
        
        # 1. 위치 데이터
        np.save(ep_save_dir / "qpos.npy", qpos.astype(np.float32))
        
        # 2. 속도 데이터
        np.save(ep_save_dir / "qvel.npy", qvel.astype(np.float32))
        
        # 3. 카메라 이미지 추출 (mp4에서)
        video_path = obs_dir / "camera.mp4"
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            frames_dir = ep_save_dir / "images"
            frames_dir.mkdir(exist_ok=True)
            
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB로 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 저장
                frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), frame_rgb)
                frame_idx += 1
            
            cap.release()
            print(f"  이미지: {frame_idx}개 추출")
        
        # 4. 메타데이터
        metadata = {
            'episode_index': ep_idx,
            'num_frames': len(qpos),
            'num_joints': 12,
            'fps': 30,
            'joint_names': [
                'follower1_j1', 'follower1_j2', 'follower1_j3',
                'follower1_j4', 'follower1_j5', 'follower1_j6',
                'follower2_j1', 'follower2_j2', 'follower2_j3',
                'follower2_j4', 'follower2_j5', 'follower2_j6',
            ],
        }
        
        with open(ep_save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ 저장 완료\n")
    
    print(f"{'='*60}")
    print(f"✓ 변환 완료!")
    print(f"  저장 위치: {lerobot_root}")
    print(f"  에피소드: {len(episodes)}개")
    print(f"{'='*60}\n")
    
    return lerobot_root


if __name__ == "__main__":
    convert_to_lerobot()
