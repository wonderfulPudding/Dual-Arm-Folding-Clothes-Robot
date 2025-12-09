#!/usr/bin/env python3
"""
듀얼 암 로봇 (자동 포트 모니터링 + 재연결)
"""

import numpy as np
import time
from dynamixel_sdk import *
import glob

PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000

LEADER1_IDS = list(range(21, 27))
LEADER2_IDS = list(range(31, 37))
FOLLOWER1_IDS = list(range(1, 7))
FOLLOWER2_IDS = list(range(11, 17))

ADDR_AX_TORQUE_ENABLE = 24
ADDR_AX_GOAL_POSITION = 30
ADDR_AX_PRESENT_POSITION = 36
ADDR_AX_MOVING_SPEED = 32


def detect_ports():
    """자동 포트 감지"""
    acm_ports = sorted(glob.glob("/dev/ttyACM*"))
    usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
    
    config = {}
    
    if len(acm_ports) >= 2:
        config['leader1_port'] = acm_ports[0]
        config['leader2_port'] = acm_ports[1]
    else:
        raise Exception(f"ACM 포트 부족 (필요: 2, 발견: {len(acm_ports)})")
    
    if len(usb_ports) >= 2:
        config['follower1_port'] = usb_ports[0]
        config['follower2_port'] = usb_ports[1]
    else:
        raise Exception(f"USB 포트 부족 (필요: 2, 발견: {len(usb_ports)})")
    
    return config


class DualArmRobot:
    def __init__(self):
        self.port_handler_leader1 = None
        self.port_handler_leader2 = None
        self.port_handler_follower1 = None
        self.port_handler_follower2 = None
        self.packet_handler = None
        self.is_connected = False
        
        self.leader1_port = None
        self.leader2_port = None
        self.follower1_port = None
        self.follower2_port = None
        
        print("[DualArmRobot] 초기화 완료")
    
    def update_ports(self):
        """포트 업데이트"""
        try:
            config = detect_ports()
            return config
        except:
            return None
    
    def connect(self):
        print("\n[DualArmRobot] 포트 감지 중...")
        ports_config = self.update_ports()
        if not ports_config:
            raise Exception("포트를 찾을 수 없음")
        
        self.leader1_port = ports_config['leader1_port']
        self.leader2_port = ports_config['leader2_port']
        self.follower1_port = ports_config['follower1_port']
        self.follower2_port = ports_config['follower2_port']
        
        print(f"  리더 1: {self.leader1_port}")
        print(f"  리더 2: {self.leader2_port}")
        print(f"  팔로워 1: {self.follower1_port}")
        print(f"  팔로워 2: {self.follower2_port}\n")
        
        print("[DualArmRobot] 연결 시작...")
        try:
            self.packet_handler = PacketHandler(PROTOCOL_VERSION)
            
            self.port_handler_leader1 = PortHandler(self.leader1_port)
            if not self.port_handler_leader1.openPort():
                raise Exception("리더 1 포트 열기 실패")
            if not self.port_handler_leader1.setBaudRate(BAUDRATE):
                raise Exception("리더 1 보드레이트 설정 실패")
            print("    ✓ 리더 1 연결")
            
            self.port_handler_leader2 = PortHandler(self.leader2_port)
            if not self.port_handler_leader2.openPort():
                raise Exception("리더 2 포트 열기 실패")
            if not self.port_handler_leader2.setBaudRate(BAUDRATE):
                raise Exception("리더 2 보드레이트 설정 실패")
            print("    ✓ 리더 2 연결")
            
            self.port_handler_follower1 = PortHandler(self.follower1_port)
            if not self.port_handler_follower1.openPort():
                raise Exception("팔로워 1 포트 열기 실패")
            if not self.port_handler_follower1.setBaudRate(BAUDRATE):
                raise Exception("팔로워 1 보드레이트 설정 실패")
            print("    ✓ 팔로워 1 연결")
            
            self.port_handler_follower2 = PortHandler(self.follower2_port)
            if not self.port_handler_follower2.openPort():
                raise Exception("팔로워 2 포트 열기 실패")
            if not self.port_handler_follower2.setBaudRate(BAUDRATE):
                raise Exception("팔로워 2 보드레이트 설정 실패")
            print("    ✓ 팔로워 2 연결")
            
            self._setup_motors()
            self.is_connected = True
            print("\n[DualArmRobot] ✓ 연결 완료!\n")
        except Exception as e:
            print(f"\n[DualArmRobot] ✗ 연결 실패: {e}")
            self.disconnect()
            raise
    
    def _setup_motors(self):
        print("  모터 설정 중...")
        for motor_id in LEADER1_IDS:
            self.packet_handler.write1ByteTxRx(self.port_handler_leader1, motor_id, ADDR_AX_TORQUE_ENABLE, 0)
        for motor_id in LEADER2_IDS:
            self.packet_handler.write1ByteTxRx(self.port_handler_leader2, motor_id, ADDR_AX_TORQUE_ENABLE, 0)
        print("    ✓ 리더 모터: 토크 OFF")
        
        for motor_id in FOLLOWER1_IDS:
            self.packet_handler.write1ByteTxRx(self.port_handler_follower1, motor_id, ADDR_AX_TORQUE_ENABLE, 1)
            self.packet_handler.write2ByteTxRx(self.port_handler_follower1, motor_id, ADDR_AX_MOVING_SPEED, 1023)
        for motor_id in FOLLOWER2_IDS:
            self.packet_handler.write1ByteTxRx(self.port_handler_follower2, motor_id, ADDR_AX_TORQUE_ENABLE, 1)
            self.packet_handler.write2ByteTxRx(self.port_handler_follower2, motor_id, ADDR_AX_MOVING_SPEED, 1023)
        print("    ✓ 팔로워 모터: 토크 ON + 속도 1023")
    
    def disconnect(self):
        print("\n[DualArmRobot] 연결 해제 중...")
        if self.port_handler_leader1:
            try:
                self.port_handler_leader1.closePort()
            except:
                pass
        if self.port_handler_leader2:
            try:
                self.port_handler_leader2.closePort()
            except:
                pass
        if self.port_handler_follower1:
            try:
                self.port_handler_follower1.closePort()
            except:
                pass
        if self.port_handler_follower2:
            try:
                self.port_handler_follower2.closePort()
            except:
                pass
        self.is_connected = False
        print("[DualArmRobot] 연결 해제 완료")
    
    def get_leader_state(self):
        if not self.is_connected:
            raise RuntimeError("로봇이 연결되지 않음")
        timestamp = time.time()
        leader1_pos = []
        for motor_id in LEADER1_IDS:
            try:
                pos, _, _ = self.packet_handler.read2ByteTxRx(self.port_handler_leader1, motor_id, ADDR_AX_PRESENT_POSITION)
                leader1_pos.append(pos)
            except:
                leader1_pos.append(512)
        leader1 = np.array(leader1_pos, dtype=np.float32)
        
        leader2_pos = []
        for motor_id in LEADER2_IDS:
            try:
                pos, _, _ = self.packet_handler.read2ByteTxRx(self.port_handler_leader2, motor_id, ADDR_AX_PRESENT_POSITION)
                leader2_pos.append(pos)
            except:
                leader2_pos.append(512)
        leader2 = np.array(leader2_pos, dtype=np.float32)
        combined = np.concatenate([leader1, leader2])
        return {'combined': combined, 'timestamp': timestamp}
    
    def send_follower_action(self, action):
        if not self.is_connected:
            raise RuntimeError("로봇이 연결되지 않음")
        if action.shape != (12,):
            raise ValueError(f"Expected (12,), got {action.shape}")
        
        for i, goal_pos in enumerate(action[0:6]):
            motor_id = FOLLOWER1_IDS[i]
            try:
                self.packet_handler.write2ByteTxRx(self.port_handler_follower1, motor_id, ADDR_AX_GOAL_POSITION, int(goal_pos))
            except:
                pass
        
        time.sleep(0.002)
        
        for i, goal_pos in enumerate(action[6:12]):
            motor_id = FOLLOWER2_IDS[i]
            try:
                self.packet_handler.write2ByteTxRx(self.port_handler_follower2, motor_id, ADDR_AX_GOAL_POSITION, int(goal_pos))
            except:
                pass
    
    def teleoperation_step(self):
        leader_state = self.get_leader_state()
        self.send_follower_action(leader_state['combined'])
        return leader_state['combined']
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    print("\n" + "="*60)
    print("듀얼 암 로봇 (자동 포트 모니터링)")
    print("="*60)
    
    robot = DualArmRobot()
    
    while True:
        try:
            robot.connect()
            
            print("텔레오퍼레이션 시작! (Ctrl+C로 종료)\n")
            
            error_count = 0
            while error_count < 5:
                try:
                    robot.teleoperation_step()
                    time.sleep(0.005)
                except Exception as e:
                    error_count += 1
                    if error_count >= 5:
                        print(f"\n⚠ 포트 오류 감지! 재연결 중...\n")
                        robot.disconnect()
                        time.sleep(2)
                        break
        
        except KeyboardInterrupt:
            print("\n\n[DualArmRobot] Ctrl+C로 종료합니다...\n")
            robot.disconnect()
            break
        
        except Exception as e:
            print(f"\n✗ 에러: {e}")
            robot.disconnect()
            print("5초 후 재연결 시도...\n")
            time.sleep(5)


if __name__ == "__main__":
    main()
