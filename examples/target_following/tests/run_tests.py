"""
模型测试套件入口
Model Test Suite Entry

选择要运行的测试:
1. 人脸检测测试 - 测试 SCRFD 检测效果
2. 人脸识别测试 - 测试 MobileFaceNet 相似度阈值
3. 人体检测测试 - 测试 MediaPipe Pose 效果
4. 手势检测测试 - 测试 MediaPipe Hands 效果
5. 性能基准测试 - 测试所有模块性能
"""

import sys
import os


def main():
    print("=" * 60)
    print("    模型测试套件 (Model Test Suite)")
    print("=" * 60)
    print()
    print("  可用测试:")
    print()
    print("  1. 人脸检测测试 (SCRFD)")
    print("     - 测试检测效果")
    print("     - 调整置信度阈值")
    print()
    print("  2. 人脸识别测试 (MobileFaceNet)")
    print("     - 保存目标人脸")
    print("     - 测试相似度阈值")
    print("     - 验证是否会误识别")
    print()
    print("  3. 人体检测测试 (MediaPipe Pose)")
    print("     - 测试骨架检测")
    print("     - 查看关键点")
    print()
    print("  4. 手势检测测试 (MediaPipe Hands)")
    print("     - 测试手势识别")
    print("     - 验证各种手势")
    print()
    print("  5. 性能基准测试")
    print("     - 测试所有模块速度")
    print("     - 生成性能报告")
    print()
    print("=" * 60)
    print()
    
    try:
        choice = input("请选择测试 (1-5), 或 'q' 退出: ").strip()
    except EOFError:
        choice = '1'
    
    if choice == 'q':
        print("退出")
        return
    
    # 切换到正确目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    if choice == '1':
        print("\n启动人脸检测测试...\n")
        import test_face_detection
        test_face_detection.main()
    elif choice == '2':
        print("\n启动人脸识别测试...\n")
        import test_face_recognition
        test_face_recognition.main()
    elif choice == '3':
        print("\n启动人体检测测试...\n")
        import test_person_detection
        test_person_detection.main()
    elif choice == '4':
        print("\n启动手势检测测试...\n")
        import test_gesture_detection
        test_gesture_detection.main()
    elif choice == '5':
        print("\n启动性能基准测试...\n")
        import benchmark
        benchmark.main()
    else:
        print(f"无效选择: {choice}")


if __name__ == "__main__":
    main()
