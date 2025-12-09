#!/usr/bin/env python3
"""
LeRobot 형식 변환 (이미지 포함)
"""

import numpy as np
from pathlib import Path
import json
import shutil

print("=== LeRobot 변환 (Vision) ===\n")

SOURCE_ROOT = Path("~/handkerchief_robot_dual_data").expanduser()
TARGET_ROOT = Path("~/lerobot_vision_data").expanduser()
TARGET_ROOT.mkdir(parents=True, exist_ok=True)

def convert():
    episodes = sorted(SOURCE_ROOT.glob("episode_*"))
    
    if not episodes:
        print("✗ 소스 데이터가 없습니다!")
        return
    
    print(f"발견된 에피소드: {len(episodes)}개\n")
    
    converted = 0
    skipped = 0
    
    for ep_dir in episodes:
        ep_num = int(ep_dir.name.split("_")[1])
        
        print(f"[{ep_dir.name}]", end=" ")
        
        obs_dir = ep_dir / "observations"
        frames_dir = obs_dir / "frames"
        
        # 필수 파일 체크
        required_files = [
            obs_dir / "follower1_positions.npy",
            obs_dir / "follower2_positions.npy",
            obs_dir / "timestamps.npy",
        ]
        
        missing = []
        for f in required_files:
            if not f.exists():
                missing.append(f.name)
        
        if missing or not frames_dir.exists():
            print(f"✗ 스킵 (누락: {', '.join(missing + (['frames'] if not frames_dir.exists() else []))})")
            skipped += 1
            continue
        
        try:
            # 데이터 로드
            follower1_pos = np.load(obs_dir / "follower1_positions.npy")
            follower2_pos = np.load(obs_dir / "follower2_positions.npy")
            timestamps = np.load(obs_dir / "timestamps.npy")
            
            # 프레임 수 확인
            frame_files = sorted(frames_dir.glob("*.jpg"))
            
            if len(follower1_pos) != len(follower2_pos) or len(follower1_pos) != len(timestamps):
                print(f"✗ 스킵 (길이 불일치)")
                skipped += 1
                continue
            
            if len(frame_files) != len(follower1_pos):
                print(f"✗ 스킵 (프레임 수 불일치: {len(frame_files)} vs {len(follower1_pos)})")
                skipped += 1
                continue
            
            # qpos, qvel 계산
            qpos = np.concatenate([follower1_pos, follower2_pos], axis=1)
            qvel = np.zeros_like(qpos)
            qvel[1:] = qpos[1:] - qpos[:-1]
            
            # 타겟 디렉토리
            target_dir = TARGET_ROOT / f"episode_{ep_num:06d}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 저장
            np.save(target_dir / "qpos.npy", qpos)
            np.save(target_dir / "qvel.npy", qvel)
            np.save(target_dir / "timestamps.npy", timestamps)
            
            # 프레임 복사
            target_frames = target_dir / "frames"
            target_frames.mkdir(exist_ok=True)
            
            for frame_file in frame_files:
                shutil.copy(frame_file, target_frames / frame_file.name)
            
            # 메타데이터
            metadata = {
                'episode_index': ep_num,
                'task': 'fold_handkerchief',
                'num_frames': len(qpos),
                'dof': 12,
                'has_vision': True,
            }
            
            with open(target_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ ({len(qpos)}프레임, {len(frame_files)}이미지)")
            converted += 1
        
        except Exception as e:
            print(f"✗ 에러: {e}")
            skipped += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"변환 완료!")
    print(f"  성공: {converted}개")
    print(f"  스킵: {skipped}개")
    print(f"  출력: {TARGET_ROOT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    convert()
