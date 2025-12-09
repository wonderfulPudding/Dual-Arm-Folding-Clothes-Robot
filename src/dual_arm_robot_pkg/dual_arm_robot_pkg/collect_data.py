#!/usr/bin/env python3
"""
데이터 수집 (프리뷰 + 녹화)
"""
print("=== collect_data.py ===")

import numpy as np
import cv2
import json
import time
from pathlib import Path
import glob
from dynamixel_sdk import *
import threading
import sys

PROTOCOL_VERSION = 1.0
BAUDRATE = 1000000

LEADER1_IDS = list(range(21, 27))
LEADER2_IDS = list(range(31, 37))
FOLLOWER1_IDS = list(range(1, 7))
FOLLOWER2_IDS = list(range(11, 17))

ADDR_TORQUE_ENABLE = 24
ADDR_GOAL_POSITION = 30
ADDR_PRESENT_POSITION = 36
ADDR_MOVING_SPEED = 32

MOTOR_SPEED = 200
TARGET_FPS = 60
DEFAULT_CAMERA = 4


def check_gui_support():
    """OpenCV GUI 지원 확인"""
    try:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test')
        return True
    except:
        return False


def detect_ports():
    acm_ports = sorted(glob.glob("/dev/ttyACM*"))
    usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
    
    config = {}
    if len(acm_ports) >= 2:
        config['leader1'] = acm_ports[0]
        config['leader2'] = acm_ports[1]
    if len(usb_ports) >= 2:
        config['follower1'] = usb_ports[0]
        config['follower2'] = usb_ports[1]
    
    return config


class DataCollector:
    def __init__(self, data_root="~/handkerchief_robot_dual_data", target_fps=TARGET_FPS, camera_index=DEFAULT_CAMERA):
        self.data_root = Path(data_root).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.camera_index = camera_index
        
        print(f"\n[설정]")
        print(f"  카메라: /dev/video{self.camera_index}")
        print(f"  목표 FPS: {target_fps}")
        
        self.gui_available = check_gui_support()
        if self.gui_available:
            print(f"  OpenCV GUI: 지원됨 ✓")
        else:
            print(f"  OpenCV GUI: 지원 안 됨 (프리뷰 비활성화)")
        
        self.current_episode = None
        self.episode_num = self._get_next_episode_num()
        self.camera = None
        
        self.recording = False
        self.should_stop = False
        self.show_preview = self.gui_available
        
        ports = detect_ports()
        self.ph_leader1 = None
        self.ph_leader2 = None
        self.ph_follower1 = None
        self.ph_follower2 = None
        self.pkt_handler = None
        self.ports = ports
        
        print(f"\n[DataCollector] 초기화 완료")
    
    def _get_next_episode_num(self):
        episodes = list(self.data_root.glob("episode_*"))
        if not episodes:
            return 0
        nums = [int(e.name.split("_")[1]) for e in episodes]
        return max(nums) + 1
    
    def connect_robot(self):
        print("\n[로봇 연결]")
        self.pkt_handler = PacketHandler(PROTOCOL_VERSION)
        
        self.ph_leader1 = PortHandler(self.ports['leader1'])
        if not self.ph_leader1.openPort():
            raise Exception(f"리더1 연결 실패: {self.ports['leader1']}")
        self.ph_leader1.setBaudRate(BAUDRATE)
        print(f"  ✓ 리더1: {self.ports['leader1']}")
        
        self.ph_leader2 = PortHandler(self.ports['leader2'])
        if not self.ph_leader2.openPort():
            raise Exception(f"리더2 연결 실패: {self.ports['leader2']}")
        self.ph_leader2.setBaudRate(BAUDRATE)
        print(f"  ✓ 리더2: {self.ports['leader2']}")
        
        self.ph_follower1 = PortHandler(self.ports['follower1'])
        if not self.ph_follower1.openPort():
            raise Exception(f"팔로워1 연결 실패: {self.ports['follower1']}")
        self.ph_follower1.setBaudRate(BAUDRATE)
        print(f"  ✓ 팔로워1: {self.ports['follower1']}")
        
        self.ph_follower2 = PortHandler(self.ports['follower2'])
        if not self.ph_follower2.openPort():
            raise Exception(f"팔로워2 연결 실패: {self.ports['follower2']}")
        self.ph_follower2.setBaudRate(BAUDRATE)
        print(f"  ✓ 팔로워2: {self.ports['follower2']}")
        
        print("\n[모터 설정]")
        
        for mid in LEADER1_IDS:
            self.pkt_handler.write1ByteTxRx(self.ph_leader1, mid, ADDR_TORQUE_ENABLE, 0)
        for mid in LEADER2_IDS:
            self.pkt_handler.write1ByteTxRx(self.ph_leader2, mid, ADDR_TORQUE_ENABLE, 0)
        print("  ✓ 리더: 토크 OFF")
        
        success1 = 0
        for mid in FOLLOWER1_IDS:
            result, error = self.pkt_handler.write1ByteTxRx(self.ph_follower1, mid, ADDR_TORQUE_ENABLE, 1)
            if result == COMM_SUCCESS and error == 0:
                self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_MOVING_SPEED, MOTOR_SPEED)
                success1 += 1
        
        success2 = 0
        for mid in FOLLOWER2_IDS:
            result, error = self.pkt_handler.write1ByteTxRx(self.ph_follower2, mid, ADDR_TORQUE_ENABLE, 1)
            if result == COMM_SUCCESS and error == 0:
                self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_MOVING_SPEED, MOTOR_SPEED)
                success2 += 1
        
        print(f"  ✓ 팔로워1: {success1}/{len(FOLLOWER1_IDS)}")
        print(f"  ✓ 팔로워2: {success2}/{len(FOLLOWER2_IDS)}")
    
    def disconnect_robot(self):
        try:
            for ph in [self.ph_leader1, self.ph_leader2, self.ph_follower1, self.ph_follower2]:
                if ph:
                    ph.closePort()
        except:
            pass
    
    def _init_camera(self):
        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not self.camera.isOpened():
            raise Exception(f"카메라 {self.camera_index}를 열 수 없습니다!")
        
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n  카메라 설정: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        
        if self.gui_available and self.show_preview:
            cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Camera Preview', 1280, 720)
    
    def _close_camera(self):
        if self.camera:
            self.camera.release()
        if self.gui_available:
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def start_episode(self):
        episode_dir = self.data_root / f"episode_{self.episode_num:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        obs_dir = episode_dir / "observations"
        obs_dir.mkdir(exist_ok=True)
        
        self.current_episode = {
            'episode_dir': episode_dir,
            'obs_dir': obs_dir,
            'frames': [],
            'follower1_positions': [],
            'follower2_positions': [],
            'timestamps': [],
        }
        
        self._init_camera()
        self.recording = False
        self.should_stop = False
        
        print(f"\n[에피소드 {self.episode_num}] 준비")
    
    def _wait_for_stop(self):
        input()
        self.should_stop = True
    
    def record_step(self):
        print(f"\n녹화 시작! (목표: {self.target_fps} FPS)")
        if self.gui_available:
            print("  [Enter] 또는 [q]/[ESC]로 종료\n")
        else:
            print("  [Enter]로 종료\n")
        
        stop_thread = threading.Thread(target=self._wait_for_stop, daemon=True)
        stop_thread.start()
        
        start_time = time.time()
        frame_count = 0
        last_print = 0
        
        self.recording = True
        
        try:
            while not self.should_stop:
                loop_start = time.time()
                
                # 로봇 제어
                l1 = [self.pkt_handler.read2ByteTxRx(self.ph_leader1, mid, ADDR_PRESENT_POSITION)[0] for mid in LEADER1_IDS]
                l2 = [self.pkt_handler.read2ByteTxRx(self.ph_leader2, mid, ADDR_PRESENT_POSITION)[0] for mid in LEADER2_IDS]
                
                for i, mid in enumerate(FOLLOWER1_IDS):
                    self.pkt_handler.write2ByteTxRx(self.ph_follower1, mid, ADDR_GOAL_POSITION, int(l1[i]))
                
                for i, mid in enumerate(FOLLOWER2_IDS):
                    self.pkt_handler.write2ByteTxRx(self.ph_follower2, mid, ADDR_GOAL_POSITION, int(l2[i]))
                
                f1 = [self.pkt_handler.read2ByteTxRx(self.ph_follower1, mid, ADDR_PRESENT_POSITION)[0] for mid in FOLLOWER1_IDS]
                f2 = [self.pkt_handler.read2ByteTxRx(self.ph_follower2, mid, ADDR_PRESENT_POSITION)[0] for mid in FOLLOWER2_IDS]
                
                # 카메라 읽기
                ret, frame = self.camera.read()
                if ret:
                    # 데이터 저장
                    self.current_episode['follower1_positions'].append(np.array(f1, dtype=np.float32))
                    self.current_episode['follower2_positions'].append(np.array(f2, dtype=np.float32))
                    self.current_episode['frames'].append(frame.copy())
                    self.current_episode['timestamps'].append(time.time())
                    frame_count += 1
                    
                    # 프리뷰 표시
                    if self.gui_available and self.show_preview:
                        display_frame = frame.copy()
                        
                        # 녹화 표시
                        cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)
                        cv2.putText(display_frame, "REC", (80, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        # 정보 표시
                        elapsed = time.time() - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        cv2.putText(display_frame, f"Cam: video{self.camera_index}", (20, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"Frame: {frame_count}", (20, 130), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"FPS: {current_fps:.1f} / {self.target_fps}", (20, 160), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow('Camera Preview', display_frame)
                        
                        # 키보드 입력
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:
                            print("\n카메라 창에서 종료")
                            self.should_stop = True
                
                # FPS 제어
                elapsed_loop = time.time() - loop_start
                sleep_time = self.frame_interval - elapsed_loop
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # 진행 상황 출력
                now = time.time()
                if now - last_print >= 1.0:
                    elapsed = now - start_time
                    current_fps = frame_count / elapsed
                    print(f"  ▌ {frame_count}프레임 @ {current_fps:.1f} FPS | {elapsed:.0f}초")
                    last_print = now
        
        except KeyboardInterrupt:
            print("\n✗ Ctrl+C")
            self.should_stop = True
        
        self.recording = False
        
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n녹화 종료!")
        print(f"  프레임: {frame_count}")
        print(f"  평균 FPS: {final_fps:.1f}")
        
        return frame_count
    
    def save_episode(self):
        episode_dir = self.current_episode['episode_dir']
        obs_dir = self.current_episode['obs_dir']
        
        print(f"\n저장 중...")
        
        np.save(obs_dir / "follower1_positions.npy", np.array(self.current_episode['follower1_positions']))
        np.save(obs_dir / "follower2_positions.npy", np.array(self.current_episode['follower2_positions']))
        np.save(obs_dir / "timestamps.npy", np.array(self.current_episode['timestamps']))
        
        frames = self.current_episode['frames']
        if frames:
            video_path = obs_dir / "camera.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, self.target_fps, (frames[0].shape[1], frames[0].shape[0]))
            for frame in frames:
                out.write(frame)
            out.release()
        
        with open(episode_dir / "metadata.json", "w") as f:
            json.dump({
                'episode_index': self.episode_num,
                'task': 'fold_handkerchief',
                'num_frames': len(frames),
                'fps': self.target_fps,
                'camera': f'/dev/video{self.camera_index}',
                'dof': 12,
            }, f, indent=2)
        
        print(f"✓ 에피소드 {self.episode_num} 저장!\n")
        
        self._close_camera()
        self.current_episode = None
        self.episode_num += 1
    
    def reset_episode(self):
        print("\n⟲ 리셋")
        
        self._close_camera()
        
        if self.current_episode:
            episode_dir = self.current_episode['episode_dir']
            if episode_dir.exists():
                import shutil
                shutil.rmtree(episode_dir)
                print(f"  ✓ {episode_dir.name} 삭제")
        
        self.current_episode = None
        print("  ✓ 리셋 완료!\n")


def main():
    print("\n" + "="*60)
    print(f"데이터 수집")
    print("="*60)
    
    # 명령줄 인자로 카메라 지정
    camera_idx = DEFAULT_CAMERA
    if len(sys.argv) > 1:
        try:
            camera_idx = int(sys.argv[1])
        except ValueError:
            print(f"\n✗ 잘못된 카메라 번호: {sys.argv[1]}")
            print(f"기본값 사용: {DEFAULT_CAMERA}")
    
    collector = DataCollector(target_fps=TARGET_FPS, camera_index=camera_idx)
    
    try:
        collector.connect_robot()
        
        ep = 0
        while ep < 15:
            print(f"\n{'='*60}")
            print(f"에피소드 {ep+1}/15")
            print(f"{'='*60}")
            
            input(f"\n준비되면 Enter...")
            
            collector.start_episode()
            frame_count = collector.record_step()
            
            while True:
                choice = input(f"\n저장? (y/n/r): ").strip().lower()
                
                if choice == 'y':
                    collector.save_episode()
                    ep += 1
                    break
                elif choice == 'n':
                    collector.reset_episode()
                    ep += 1
                    break
                elif choice == 'r':
                    collector.reset_episode()
                    print("\n다시 녹화!")
                    input("Enter...")
                    collector.start_episode()
                    frame_count = collector.record_step()
                else:
                    print("y, n, r 중 하나!")
        
        print(f"\n{'='*60}")
        print(f"✓ 완료!")
        print(f"{'='*60}\n")
    
    except Exception as e:
        print(f"\n✗ 에러: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        collector.disconnect_robot()
        if collector.gui_available:
            try:
                cv2.destroyAllWindows()
            except:
                pass


if __name__ == "__main__":
    main()
