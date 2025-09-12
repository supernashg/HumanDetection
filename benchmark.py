"""
性能基准测试脚本
对比YOLO和MediaPipe的检测速度
"""
import cv2
import time
import numpy as np
from detectors import create_detector


def benchmark_detector(detector_type, num_frames=30):
    """测试检测器性能"""
    print(f"\n=== {detector_type.upper()} 性能测试 ===")
    
    try:
        # 创建检测器
        if detector_type == 'yolo':
            detector = create_detector('yolo', model_path='yolov8n-pose.pt')
        else:
            detector = create_detector('mediapipe')
        
        print(f"模型能力: {detector.get_capabilities()}")
        
        # 创建测试图片 (模拟摄像头帧)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 预热
        for _ in range(3):
            detector.detect(test_frame)
        
        # 性能测试
        times = []
        for i in range(num_frames):
            start_time = time.time()
            results = detector.detect(test_frame)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            times.append(inference_time)
            
            if i % 10 == 0:
                print(f"  帧 {i+1}: {inference_time:.2f}ms")
        
        # 统计结果
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / avg_time
        
        print(f"\n📊 性能统计:")
        print(f"  平均时间: {avg_time:.2f}ms")
        print(f"  最快时间: {min_time:.2f}ms")  
        print(f"  最慢时间: {max_time:.2f}ms")
        print(f"  理论FPS: {fps:.1f}")
        
        return avg_time, fps
        
    except Exception as e:
        print(f"❌ {detector_type} 测试失败: {e}")
        return None, None


def main():
    """主函数"""
    print("🚀 人体检测性能基准测试")
    print("测试环境: Apple Silicon (M2), Python 3.11.13")
    
    # 测试YOLO
    yolo_time, yolo_fps = benchmark_detector('yolo')
    
    # 测试MediaPipe
    mp_time, mp_fps = benchmark_detector('mediapipe')
    
    # 对比结果
    if yolo_time and mp_time:
        print(f"\n🏆 性能对比结果:")
        print(f"{'模型':<12} {'平均时间':<10} {'FPS':<8} {'相对速度'}")
        print("-" * 40)
        print(f"{'YOLO':<12} {yolo_time:<10.2f} {yolo_fps:<8.1f} 1.0x")
        
        speedup = yolo_time / mp_time
        print(f"{'MediaPipe':<12} {mp_time:<10.2f} {mp_fps:<8.1f} {speedup:.1f}x")
        
        if mp_time < yolo_time:
            print(f"\n✨ MediaPipe 比 YOLO 快 {speedup:.1f} 倍!")
        else:
            print(f"\n✨ YOLO 比 MediaPipe 快 {1/speedup:.1f} 倍!")


if __name__ == "__main__":
    main()