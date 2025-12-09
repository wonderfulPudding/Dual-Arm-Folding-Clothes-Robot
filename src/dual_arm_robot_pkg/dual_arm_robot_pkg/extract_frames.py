#!/usr/bin/env python3
"""
MP4에서 프레임 추출
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("=== 프레임 추출 ===\n")

SOURCE_ROOT = Path("~/handkerchief_robot_dual_data").expanduser()

def extract_frames():
    episodes = sorted(SOURCE_ROOT.glob("episode_*"))
    
    if not episodes:
        print("✗ 에피소드가 없습니다!")
        return
    
    print(f"발견된 에피소드: {len(episodes)}개\n")
    
    for ep_dir in tqdm(episodes, desc="프레임 추출"):
        video_path = ep_dir / "observations" / "camera.mp4"
        frames_dir = ep_dir / "observations" / "frames"
        
        if not video_path.exists():
            print(f"  ✗ {ep_dir.name}: camera.mp4 없음")
            continue
        
        frames_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_path = frames_dir / f"{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
        
        cap.release()
        print(f"  ✓ {ep_dir.name}: {frame_count}프레임")
    
    print("\n✓ 프레임 추출 완료!")


if __name__ == "__main__":
    extract_frames()
