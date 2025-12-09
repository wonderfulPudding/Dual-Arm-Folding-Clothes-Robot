#!/usr/bin/env python3
"""
카메라 프리뷰 (단독 실행)
"""
print("=== 카메라 프리뷰 ===")

import cv2
import sys
import time

DEFAULT_CAMERA = 4


def main():
    # 카메라 번호 받기
    camera_idx = DEFAULT_CAMERA
    if len(sys.argv) > 1:
        try:
            camera_idx = int(sys.argv[1])
        except ValueError:
            print(f"✗ 잘못된 카메라 번호: {sys.argv[1]}")
            print(f"기본값 사용: {DEFAULT_CAMERA}")
    
    print(f"\n카메라: /dev/video{camera_idx}")
    print("종료: [q] 또는 [ESC]\n")
    
    # 카메라 열기
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"✗ 카메라 {camera_idx}를 열 수 없습니다!")
        return
    
    # 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"설정: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    
    # 윈도우 생성
    cv2.namedWindow('Camera Preview', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Preview', 1280, 720)
    
    frame_count = 0
    start_time = time.time()
    last_print = 0
    
    print("\n프리뷰 시작!\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ 프레임 읽기 실패!")
                break
            
            frame_count += 1
            
            # 정보 표시
            display_frame = frame.copy()
            
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(display_frame, f"Camera: video{camera_idx}", (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (20, 80), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (20, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Time: {elapsed:.1f}s", (20, 160), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 조작 안내
            cv2.putText(display_frame, "Press [q] or [ESC] to quit", 
                      (20, display_frame.shape[0] - 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Camera Preview', display_frame)
            
            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n종료됨")
                break
            
            # 1초마다 터미널 출력
            now = time.time()
            if now - last_print >= 1.0:
                print(f"  ▌ {frame_count}프레임 @ {current_fps:.1f} FPS")
                last_print = now
    
    except KeyboardInterrupt:
        print("\n✗ Ctrl+C")
    
    finally:
        total_time = time.time() - start_time
        final_fps = frame_count / total_time if total_time > 0 else 0
        
        print(f"\n통계:")
        print(f"  총 프레임: {frame_count}")
        print(f"  총 시간: {total_time:.1f}초")
        print(f"  평균 FPS: {final_fps:.1f}")
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
