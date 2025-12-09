#!/usr/bin/env python3
"""
자율 실행 (손수건 감지 포함)
"""

import torch
import torch.nn as nn
import numpy as np
import time
import glob
from pathlib import Path
from dynamixel_sdk import *
import cv2

print("=== 자율 실행 (손수건 감지) ===\n")

CHECKPOINT_PATH = Path("~/handkerchief_checkpoints/best_model.pth").expanduser()
PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000

FOLLOWER1_IDS = list(range(1, 7))
FOLLOWER2_IDS = list(range(11, 17))

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
ADDR_MOVING_SPEED = 32

MOTOR_SPEED = 200
CAMERA_INDEX = 4

# 손수건 색상 범위 (HSV) - 흰색 손수건 기준
# 다른 색상이면 조정 필요
HANDKERCHIEF_COLOR_LOWER = np.array([0, 0, 180])    # 흰색 하한
HANDKERCHIEF_COLOR_UPPER = np.array([180, 30, 255])  # 흰색 상한


def detect_ports():
    usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
    config = {}
    if len(usb_ports) >= 2:
        config['follower1'] = usb_ports[0]
        config['follower2'] = usb_ports[1]
    return config


def detect_handkerchief(frame):
    """
    손수건 감지 (색상 기반)
    
    Returns:
        detected (bool): 손수건 감지 여부
        bbox (tuple): (x, y, w, h) 바운딩 박스
        area (int): 감지된 영역 크기
    """
    # HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 색상 마스크
    mask = cv2.inRange(hsv, HANDKERCHIEF_COLOR_LOWER, HANDKERCHIEF_COLOR_UPPER)
    
    # 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, None, 0
    
    # 가장 큰 컨투어 찾기
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    
    # 최소 크기 필터 (너무 작은 건 무시)
    MIN_AREA = 5000
    if area < MIN_AREA:
        return False, None, 0
    
    # 바운딩 박스
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return True, (x, y, w, h), area


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


class AutonomousController:
    def __init__(self):
        self.ports = detect_ports()
        print(f"포트 설정:")
        print(f"  팔로워1 (ID 1~6): {self.ports['follower1']}")
        print(f"  팔로워2 (ID 11~16): {self.ports['follower2']}\n")
        
        self.ph_follower1 = None
        self.ph_follower2 = None
        self.pkt_handler = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"디바이스: {self.device}\n")
        
        self.model = ImprovedPolicy(hidden_dim=256).to(self.device)
        
        if CHECKPOINT_PATH.exists():
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                loss = checkpoint.get('loss', 'N/A')
                epoch = checkpoint.get('epoch', 'N/A')
                print(f"✓ 모델 로드: epoch {epoch}, loss {loss:.4f}\n")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✓ 모델 로드: {CHECKPOINT_PATH}\n")
            
            self.model.eval()
        else:
            print(f"✗ 체크포인트 없음: {CHECKPOINT_PATH}\n")
            exit(1)
        
        self.prev_qpos = None
        
        # 카메라 초기화
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print(f"✗ 카메라 {CAMERA_INDEX}를 열 수 없습니다!")
            self.camera = None
        else:
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(f"✓ 카메라: /dev/video{CAMERA_INDEX}\n")
            
            # 프리뷰 윈도우
            cv2.namedWindow('Handkerchief Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Handkerchief Detection', 960, 540)
    
    def connect_robot(self):
        print("[로봇 연결]")
        self.pkt_handler = PacketHandler(PROTOCOL_VERSION)
        
        self.ph_follower1 = PortHandler(self.ports['follower1'])
        if not self.ph_follower1.openPort():
            raise Exception(f"팔로워1 연결 실패: {self.ports['follower1']}")
        self.ph_follower1.setBaudRate(BAUDRATE)
        print(f"  ✓ 팔로워1: {self.ports['follower1']} (ID 1~6)")
        
        self.ph_follower2 = PortHandler(self.ports['follower2'])
        if not self.ph_follower2.openPort():
            raise Exception(f"팔로워2 연결 실패: {self.ports['follower2']}")
        self.ph_follower2.setBaudRate(BAUDRATE)
        print(f"  ✓ 팔로워2: {self.ports['follower2']} (ID 11~16)")
        
        print("\n[모터 토크 설정]")
        
        success1 = 0
        for mid in FOLLOWER1_IDS:
            result, error = self.pkt_handler.write1ByteTxRx(self.ph_follower1, mid, ADDR_TORQUE_ENABLE, 1)
            
            if result == COMM_SUCCESS and error == 0:
                self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_MOVING_SPEED, MOTOR_SPEED)
                
                torque, _, _ = self.pkt_handler.read1ByteTxRx(self.ph_follower1, mid, ADDR_TORQUE_ENABLE)
                if torque == 1:
                    success1 += 1
        
        print(f"  → 팔로워1: {success1}/{len(FOLLOWER1_IDS)} 성공")
        
        success2 = 0
        for mid in FOLLOWER2_IDS:
            result, error = self.pkt_handler.write1ByteTxRx(self.ph_follower2, mid, ADDR_TORQUE_ENABLE, 1)
            
            if result == COMM_SUCCESS and error == 0:
                self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_MOVING_SPEED, MOTOR_SPEED)
                
                torque, _, _ = self.pkt_handler.read1ByteTxRx(self.ph_follower2, mid, ADDR_TORQUE_ENABLE)
                if torque == 1:
                    success2 += 1
        
        print(f"  → 팔로워2: {success2}/{len(FOLLOWER2_IDS)} 성공\n")
        
        if success1 == 0 and success2 == 0:
            raise Exception("모든 모터 토크 활성화 실패!")
    
    def disconnect_robot(self):
        try:
            for ph in [self.ph_follower1, self.ph_follower2]:
                if ph:
                    ph.closePort()
        except:
            pass
        
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
    
    def execute_action(self, action):
        action = action.cpu().numpy()
        
        for i, mid in enumerate(FOLLOWER1_IDS):
            pos = int(np.clip(action[i], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_GOAL_POSITION, pos)
        
        for i, mid in enumerate(FOLLOWER2_IDS):
            pos = int(np.clip(action[i+6], 0, 1023))
            self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_GOAL_POSITION, pos)
    
    def run(self, duration=30.0):
        print("[자율 실행 시작]")
        print(f"  시간: {duration}초")
        print("  Ctrl+C로 중단\n")
        
        start_time = time.time()
        step = 0
        handkerchief_detected = False
        
        try:
            while (time.time() - start_time) < duration:
                # 상태 읽기
                state, current_qpos = self.get_current_state()
                
                # 행동 예측
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action_tensor = self.model(state_tensor).squeeze(0)
                
                # 행동 실행
                self.execute_action(action_tensor)
                
                # 카메라로 손수건 감지
                if self.camera:
                    ret, frame = self.camera.read()
                    
                    if ret:
                        detected, bbox, area = detect_handkerchief(frame)
                        
                        # 프레임에 표시
                        display_frame = frame.copy()
                        
                        if detected:
                            if not handkerchief_detected:
                                print(f"\n  🟢 손수건 감지됨! (면적: {area})")
                                handkerchief_detected = True
                            
                            x, y, w, h = bbox
                            
                            # 초록색 박스
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                            
                            # 텍스트
                            cv2.putText(display_frame, "HANDKERCHIEF DETECTED", (x, y-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Area: {area}", (x, y+h+30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            if handkerchief_detected:
                                print(f"\n  🔴 손수건 감지 해제")
                                handkerchief_detected = False
                            
                            # 빨간색 텍스트
                            cv2.putText(display_frame, "Searching...", (20, 50),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # 스텝 정보
                        cv2.putText(display_frame, f"Step: {step}", (20, display_frame.shape[0] - 50),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        elapsed = time.time() - start_time
                        cv2.putText(display_frame, f"Time: {elapsed:.1f}s", (20, display_frame.shape[0] - 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow('Handkerchief Detection', display_frame)
                        
                        # 키 입력
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            print("\n✗ 사용자 중단")
                            break
                
                step += 1
                
                if step % 30 == 0:
                    elapsed = time.time() - start_time
                    status = "🟢 감지됨" if handkerchief_detected else "🔴 미감지"
                    print(f"  ▌ {step}스텝 ({elapsed:.1f}초) - {status}")
                
                time.sleep(0.03)
        
        except KeyboardInterrupt:
            print("\n✗ 중단됨")
        
        print(f"\n✓ 자율 실행 완료 ({step}스텝)\n")


def main():
    print("="*60)
    print("자율 실행 모드 (손수건 감지)")
    print("="*60 + "\n")
    
    controller = AutonomousController()
    
    try:
        controller.connect_robot()
        
        input("\n준비되면 Enter를 눌러 시작...")
        
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
