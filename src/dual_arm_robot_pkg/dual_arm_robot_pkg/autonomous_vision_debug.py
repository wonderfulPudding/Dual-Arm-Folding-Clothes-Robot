#!/usr/bin/env python3
"""
이미지 기반 자동제어 (디버그 버전)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import glob
from pathlib import Path
from dynamixel_sdk import *

print("=== Vision-based Autonomous Control (DEBUG) ===\n")

CHECKPOINT_PATH = Path("~/handkerchief_checkpoints/vision_best_model.pth").expanduser()
PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000

FOLLOWER1_IDS = list(range(1, 7))
FOLLOWER2_IDS = list(range(11, 17))

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
ADDR_MOVING_SPEED = 32

MOTOR_SPEED = 600
CAMERA_INDEX = 4
IMAGE_SIZE = 224


def detect_ports():
    usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
    config = {}
    if len(usb_ports) >= 2:
        config['follower1'] = usb_ports[0]
        config['follower2'] = usb_ports[1]
    return config


class VisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        
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
        
        self.state_encoder = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
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


class VisionController:
    def __init__(self):
        self.ports = detect_ports()
        print(f"포트 설정:")
        print(f"  팔로워1: {self.ports['follower1']}")
        print(f"  팔로워2: {self.ports['follower2']}\n")
        
        self.ph_follower1 = None
        self.ph_follower2 = None
        self.pkt_handler = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"디바이스: {self.device}\n")
        
        self.model = VisionPolicy().to(self.device)
        
        if CHECKPOINT_PATH.exists():
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 모델 로드: epoch {checkpoint.get('epoch', 'N/A')}, loss {checkpoint.get('loss', 'N/A'):.4f}\n")
            self.model.eval()
        else:
            print(f"✗ 체크포인트 없음: {CHECKPOINT_PATH}\n")
            exit(1)
        
        self.prev_qpos = None
        
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print(f"✗ 카메라 열기 실패!")
            exit(1)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print(f"✓ 카메라: /dev/video{CAMERA_INDEX}\n")
        
        cv2.namedWindow('Vision Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vision Control', 960, 540)
    
    def connect_robot(self):
        print("[로봇 연결]")
        self.pkt_handler = PacketHandler(PROTOCOL_VERSION)
        
        self.ph_follower1 = PortHandler(self.ports['follower1'])
        if not self.ph_follower1.openPort():
            raise Exception(f"팔로워1 포트 열기 실패!")
        self.ph_follower1.setBaudRate(BAUDRATE)
        print(f"  ✓ 팔로워1 연결")
        
        self.ph_follower2 = PortHandler(self.ports['follower2'])
        if not self.ph_follower2.openPort():
            raise Exception(f"팔로워2 포트 열기 실패!")
        self.ph_follower2.setBaudRate(BAUDRATE)
        print(f"  ✓ 팔로워2 연결\n")
        
        print("[모터 토크 활성화]")
        
        for mid in FOLLOWER1_IDS:
            self.pkt_handler.write1ByteTxRx(self.ph_follower1, mid, ADDR_TORQUE_ENABLE, 1)
            self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_MOVING_SPEED, MOTOR_SPEED)
        
        for mid in FOLLOWER2_IDS:
            self.pkt_handler.write1ByteTxRx(self.ph_follower2, mid, ADDR_TORQUE_ENABLE, 1)
            self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_MOVING_SPEED, MOTOR_SPEED)
        
        print("  ✓ 토크 활성화\n")
    
    def disconnect_robot(self):
        if self.ph_follower1:
            self.ph_follower1.closePort()
        if self.ph_follower2:
            self.ph_follower2.closePort()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def get_current_state(self):
        f1_pos = []
        for mid in FOLLOWER1_IDS:
            pos, _, _ = self.pkt_handler.read2ByteTxRx(self.ph_follower1, mid, ADDR_PRESENT_POSITION)
            f1_pos.append(pos if pos else 512)
        
        f2_pos = []
        for mid in FOLLOWER2_IDS:
            pos, _, _ = self.pkt_handler.read2ByteTxRx(self.ph_follower2, mid, ADDR_PRESENT_POSITION)
            f2_pos.append(pos if pos else 512)
        
        qpos = np.array(f1_pos + f2_pos, dtype=np.float32)
        
        if self.prev_qpos is None:
            qvel = np.zeros(12, dtype=np.float32)
        else:
            qvel = qpos - self.prev_qpos
        
        self.prev_qpos = qpos.copy()
        state = np.concatenate([qpos, qvel])
        
        return state, qpos
    
    def preprocess_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)
    
    def execute_action(self, action):
        action_np = action.cpu().numpy()
        
        for i, mid in enumerate(FOLLOWER1_IDS):
            pos = int(np.clip(action_np[i], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_GOAL_POSITION, pos)
        
        for i, mid in enumerate(FOLLOWER2_IDS):
            pos = int(np.clip(action_np[i+6], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_GOAL_POSITION, pos)
        
        return action_np
    
    def run(self, duration=30.0):
        print("[이미지 기반 자동제어 시작 - 디버그 모드]")
        print(f"  시간: {duration}초\n")
        
        start_time = time.time()
        step = 0
        
        try:
            while (time.time() - start_time) < duration:
                ret, frame = self.camera.read()
                if not ret:
                    print("  ✗ 프레임 읽기 실패!")
                    break
                
                state, current_qpos = self.get_current_state()
                
                img_tensor = self.preprocess_image(frame)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_tensor = self.model(img_tensor, state_tensor).squeeze(0)
                
                action_np = self.execute_action(action_tensor)
                
                # 🔍 디버그 출력
                if step % 10 == 0:
                    print(f"\n[스텝 {step}]")
                    print(f"  현재 위치 (qpos): {current_qpos[:6].astype(int)}")
                    print(f"  예측 행동 (action): {action_np[:6].astype(int)}")
                    print(f"  차이 (delta): {(action_np[:6] - current_qpos[:6]).astype(int)}")
                
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Step: {step}", (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Action: {action_np[0]:.0f}", (20, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Vision Control', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n✗ 중단")
                    break
                
                step += 1
                time.sleep(0.03)
        
        except KeyboardInterrupt:
            print("\n✗ Ctrl+C")
        
        print(f"\n✓ 완료 ({step}스텝)\n")


def main():
    print("="*60)
    print("이미지 기반 자동제어 (디버그)")
    print("="*60 + "\n")
    
    controller = VisionController()
    
    try:
        controller.connect_robot()
        input("\n준비되면 Enter...")
        controller.run(duration=30.0)
        print("="*60)
        print("✓ 완료!")
        print("="*60)
    
    except Exception as e:
        print(f"\n✗ 에러: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        controller.disconnect_robot()


if __name__ == "__main__":
    main()
