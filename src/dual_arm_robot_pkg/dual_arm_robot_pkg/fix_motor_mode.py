#!/usr/bin/env python3
"""
모터 위치 제어 모드 강제 설정
"""

from dynamixel_sdk import *
import time

PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000

FOLLOWER1_IDS = list(range(1, 7))
FOLLOWER2_IDS = list(range(11, 17))

ADDR_CW_ANGLE_LIMIT = 6
ADDR_CCW_ANGLE_LIMIT = 8
ADDR_TORQUE_ENABLE = 24
ADDR_MOVING_SPEED = 32

ports = ['/dev/ttyUSB0', '/dev/ttyUSB1']
motor_ids = [FOLLOWER1_IDS, FOLLOWER2_IDS]

print("=== 모터 위치 제어 모드 설정 ===\n")

for idx, port_name in enumerate(ports):
    print(f"[{port_name}]")
    
    ph = PortHandler(port_name)
    if not ph.openPort():
        print(f"  ✗ 포트 열기 실패!")
        continue
    
    ph.setBaudRate(BAUDRATE)
    pkt = PacketHandler(PROTOCOL_VERSION)
    
    for mid in motor_ids[idx]:
        print(f"  ID {mid}:", end=" ")
        
        # 1. 토크 비활성화 (설정 변경 전 필수)
        pkt.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 0)
        time.sleep(0.01)
        
        # 2. CW/CCW 각도 제한 설정 (0이 아닌 값으로)
        # 0~1023 범위로 설정 (위치 제어 모드)
        pkt.write2ByteTxRx(ph, mid, ADDR_CW_ANGLE_LIMIT, 0)
        time.sleep(0.01)
        pkt.write2ByteTxRx(ph, mid, ADDR_CCW_ANGLE_LIMIT, 1023)
        time.sleep(0.01)
        
        # 3. 토크 재활성화
        pkt.write1ByteTxRx(ph, mid, ADDR_TORQUE_ENABLE, 1)
        time.sleep(0.01)
        
        # 4. 속도 설정
        pkt.write2ByteTxRx(ph, mid, ADDR_MOVING_SPEED, 300)
        time.sleep(0.01)
        
        # 5. 확인
        cw, _, _ = pkt.read2ByteTxRx(ph, mid, ADDR_CW_ANGLE_LIMIT)
        ccw, _, _ = pkt.read2ByteTxRx(ph, mid, ADDR_CCW_ANGLE_LIMIT)
        
        if cw == 0 and ccw == 1023:
            print(f"✓ 위치 제어 모드 (0~1023)")
        else:
            print(f"✗ 설정 실패 (CW:{cw}, CCW:{ccw})")
    
    ph.closePort()
    print()

print("✓ 모든 모터 설정 완료!\n")
print("이제 autonomous_vision_debug.py를 다시 실행하세요.")
