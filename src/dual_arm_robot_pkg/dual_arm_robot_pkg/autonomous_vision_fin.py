#!/usr/bin/env python3
"""
이미지 기반 자동제어 (토크 활성화 강화 + 정지 기능)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import glob
import threading
from pathlib import Path
from dynamixel_sdk import *


print("=== Vision-based Autonomous Control ===\n")


CHECKPOINT_PATH = Path("~/handkerchief_checkpoints/vision_best_model.pth").expanduser()
PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000


FOLLOWER1_IDS = list([1,2,3,4,5,6])
FOLLOWER2_IDS = list([11,12,13,14,15,16])


ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
ADDR_MOVING_SPEED = 100


MOTOR_SPEED = 200
CAMERA_INDEX = 4
IMAGE_SIZE = 224



def detect_ports():
    """포트 고정 지정"""
    config = {
        'follower1': '/dev/ttyUSB0',
        'follower2': '/dev/ttyUSB1',
    }
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
        
        # 카메라
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print(f"✗ 카메라 열기 실패!")
            exit(1)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print(f"✓ 카메라: /dev/video{CAMERA_INDEX}\n")
        
        cv2.namedWindow('Vision Control', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Vision Control', 960, 540)
        
        # 정지 플래그
        self.stop_flag = False
        self.stop_lock = threading.Lock()
    
    def set_stop_flag(self):
        """정지 플래그 설정"""
        with self.stop_lock:
            self.stop_flag = True
    
    def is_stopped(self):
        """정지 여부 확인"""
        with self.stop_lock:
            return self.stop_flag
    
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
        
        # 팔로워1 토크
        success1 = 0
        for mid in FOLLOWER1_IDS:
            for attempt in range(3):
                result, error = self.pkt_handler.write1ByteTxRx(
                    self.ph_follower1, mid, ADDR_TORQUE_ENABLE, 1
                )
                
                if result == COMM_SUCCESS and error == 0:
                    time.sleep(0.05)  # 토크 쓰기 후 대기
                    
                    # 속도 설정
                    self.pkt_handler.write2ByteTxRx(
                        self.ph_follower1, mid, ADDR_MOVING_SPEED, MOTOR_SPEED
                    )
                    time.sleep(0.02)
                    
                    # 토크 상태 확인
                    torque, _, _ = self.pkt_handler.read1ByteTxRx(
                        self.ph_follower1, mid, ADDR_TORQUE_ENABLE
                    )
                    
                    if torque == 1:
                        print(f"    팔로워1 ID {mid}: ✓")
                        success1 += 1
                        break
                    else:
                        print(f"    팔로워1 ID {mid}: 토크 확인 실패 (재시도 중...)")
                        time.sleep(0.05)
                else:
                    if attempt < 2:
                        time.sleep(0.05)
            else:
                print(f"    팔로워1 ID {mid}: ✗ (재시도 3회 실패)")
        
        print(f"  → 팔로워1: {success1}/{len(FOLLOWER1_IDS)} 성공")
        
        # 팔로워2 토크
        success2 = 0
        for mid in FOLLOWER2_IDS:
            for attempt in range(3):
                result, error = self.pkt_handler.write1ByteTxRx(
                    self.ph_follower2, mid, ADDR_TORQUE_ENABLE, 1
                )
                
                if result == COMM_SUCCESS and error == 0:
                    time.sleep(0.05)
                    
                    # 속도 설정
                    self.pkt_handler.write2ByteTxRx(
                        self.ph_follower2, mid, ADDR_MOVING_SPEED, MOTOR_SPEED
                    )
                    time.sleep(0.02)
                    
                    # 토크 상태 확인
                    torque, _, _ = self.pkt_handler.read1ByteTxRx(
                        self.ph_follower2, mid, ADDR_TORQUE_ENABLE
                    )
                    
                    if torque == 1:
                        print(f"    팔로워2 ID {mid}: ✓")
                        success2 += 1
                        break
                    else:
                        print(f"    팔로워2 ID {mid}: 토크 확인 실패 (재시도 중...)")
                        time.sleep(0.05)
                else:
                    if attempt < 2:
                        time.sleep(0.05)
            else:
                print(f"    팔로워2 ID {mid}: ✗ (재시도 3회 실패)")
        
        print(f"  → 팔로워2: {success2}/{len(FOLLOWER2_IDS)} 성공\n")
        
        if success1 == 0 or success2 == 0:
            print("\n⚠️  경고: 일부 모터 토크가 활성화되지 않았습니다!")
            print("  - USB 포트 연결 확인")
            print("  - 로봇 전원 ON 확인")
            print("  - 모터 ID 번호 확인\n")
            choice = input("계속하시겠습니까? (y/n): ").strip().lower()
            if choice != 'y':
                raise Exception("사용자 중단")
    
    def disconnect_robot(self):
        """로봇 연결 해제 (모든 모터 토크 끄기)"""
        print("\n[모터 토크 비활성화]")
        
        if self.pkt_handler and self.ph_follower1:
            for mid in FOLLOWER1_IDS:
                self.pkt_handler.write1ByteTxRx(self.ph_follower1, mid, ADDR_TORQUE_ENABLE, 0)
                time.sleep(0.01)
            print("  ✓ 팔로워1 토크 비활성화")
        
        if self.pkt_handler and self.ph_follower2:
            for mid in FOLLOWER2_IDS:
                self.pkt_handler.write1ByteTxRx(self.ph_follower2, mid, ADDR_TORQUE_ENABLE, 0)
                time.sleep(0.01)
            print("  ✓ 팔로워2 토크 비활성화")
        
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
        
        return state
    
    def preprocess_image(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0).to(self.device)
    


    '''
    def execute_action(self, action):
        action = action.cpu().numpy()
        
        for i, mid in enumerate(FOLLOWER1_IDS):
            pos = int(np.clip(action[i], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_GOAL_POSITION, pos)
        
        for i, mid in enumerate(FOLLOWER2_IDS):
            pos = int(np.clip(action[i+6], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_GOAL_POSITION, pos)
    '''
            
    def execute_action(self, action):
        action = action.cpu().numpy()
    
        # 액션 값 정규화 (최소~최대를 0~1로 변환)
        action_min = action.min()
        action_max = action.max()
        action_range = action_max - action_min
    
        if action_range < 1e-6:  # 모든 값이 거의 같으면
            action_normalized = np.ones_like(action) * 0.5
        else:
            action_normalized = (action - action_min) / action_range
    
        # 0~1023 범위로 스케일링
        action_scaled = action_normalized * 1023
    
        for i, mid in enumerate(FOLLOWER1_IDS):
            pos = int(np.clip(action_scaled[i], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_GOAL_POSITION, pos)
    
        for i, mid in enumerate(FOLLOWER2_IDS):
            pos = int(np.clip(action_scaled[i+6], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_GOAL_POSITION, pos)


    def run(self, duration=30.0):
        print("[이미지 기반 자동제어 시작]")
        print(f"  시간: {duration}초")
        print(f"  정지: 'Q' 또는 'ESC' 키 또는 Ctrl+C\n")
        
        start_time = time.time()
        step = 0
        
        try:
            while (time.time() - start_time) < duration:
                # 정지 플래그 확인
                if self.is_stopped():
                    print("\n✗ 사용자가 정지 요청")
                    break
                
                # 카메라 읽기
                ret, frame = self.camera.read()
                if not ret:
                    print("  ✗ 프레임 읽기 실패!")
                    break
                
                # 상태 읽기
                state = self.get_current_state()
                
                # 이미지 전처리
                img_tensor = self.preprocess_image(frame)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # 행동 예측
                with torch.no_grad():
                    action_tensor = self.model(img_tensor, state_tensor).squeeze(0)
                
                # 행동 실행
                self.execute_action(action_tensor)
                
                # 프리뷰
                display_frame = frame.copy()
                cv2.putText(display_frame, f"Step: {step}", (20, 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "VISION CONTROL", (20, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                elapsed = time.time() - start_time
                cv2.putText(display_frame, f"Time: {elapsed:.1f}s", (20, display_frame.shape[0] - 50),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 정지 키 안내
                cv2.putText(display_frame, "Press 'Q' or 'ESC' to STOP", (20, display_frame.shape[0] - 20),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Vision Control', display_frame)
                
                # 키 입력 확인
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    self.set_stop_flag()
                    break
                
                step += 1
                
                if step % 30 == 0:
                    print(f"  ▌ {step}스텝")
                
                time.sleep(0.03)
        
        except KeyboardInterrupt:
            print("\n✗ Ctrl+C - 중단됨")
        
        print(f"\n✓ 완료 ({step}스텝)\n")



def main():
    print("="*60)
    print("이미지 기반 자동제어 (정지 기능 포함)")
    print("="*60 + "\n")
    
    controller = VisionController()
    
    try:
        controller.connect_robot()
        input("\n준비되면 Enter...")
        controller.run(duration=300.0)
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
