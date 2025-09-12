# 多模型人体检测指南

## 概述

项目现已支持两种检测模型：**YOLO** 和 **MediaPipe**，用户可以根据需求选择最适合的检测方法。

## 模型对比

| 特性 | YOLO | MediaPipe |
|------|------|-----------|
| **检测速度** | 中等 (~110-200ms) | **极快** (~5-20ms) |
| **模型大小** | 较大 (6-50MB) | **小** (几MB) |
| **准确度** | 高 | 中等偏高 |
| **关键点数量** | 17个 (COCO格式) | **33个** (更详细) |
| **多人检测** | ✅ 优秀 | ❌ 主要单人 |
| **分割掩码** | ✅ (仅-seg模型) | ✅ |
| **边界框** | ✅ | ✅ |
| **手部检测** | ❌ | ✅ |
| **面部检测** | ❌ | ✅ |

## 使用方法

### 基本用法

```bash
# 使用YOLO检测器（默认）
uv run python main_multi.py

# 使用MediaPipe检测器（需要安装mediapipe）
uv run python main_multi.py --detector mediapipe

# 使用特定YOLO模型
uv run python main_multi.py --detector yolo --model yolov8s-pose.pt
```

### 安装MediaPipe（可选）

```bash
# 安装MediaPipe支持
pip install mediapipe

# 或者添加到项目依赖
uv add mediapipe
```

### 高级选项

```bash
# 调整置信度阈值
uv run python main_multi.py --detector mediapipe --confidence 0.7

# 禁用某些可视化功能
uv run python main_multi.py --no-mask --no-help

# 使用不同摄像头
uv run python main_multi.py --camera 1
```

## 实时控制

运行时可以使用以下按键：

- **'b'** - 切换边界框显示
- **'m'** - 切换分割掩码显示  
- **'p'** - 切换姿态关键点显示
- **'h'** - 切换帮助文本显示
- **'q'** - 退出程序

## 模型能力检测

系统会自动检测每个模型的能力：

```python
# YOLO姿态模型能力
{
    'bbox': True,
    'pose': True, 
    'segmentation': False,
    'multi_person': True,
    'keypoints_count': 17,
    'model_type': 'YOLO'
}

# MediaPipe能力
{
    'bbox': True,
    'pose': True,
    'segmentation': True, 
    'multi_person': False,
    'keypoints_count': 33,
    'model_type': 'MediaPipe'
}
```

## 架构设计

### 统一接口

所有检测器都实现 `BaseDetector` 抽象类：

```python
class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]
    
    @abstractmethod 
    def get_capabilities(self) -> dict
```

### 检测结果格式

统一的 `DetectionResult` 数据结构：

```python
@dataclass
class DetectionResult:
    bbox: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Optional[np.ndarray]  # (N, 3) [x, y, confidence] 
    mask: Optional[np.ndarray]
    person_id: int
```

### 关键点格式转换

MediaPipe的33个关键点会自动转换为COCO的17个关键点格式，以保持可视化的一致性。

## 推荐使用场景

### YOLO 适用于：
- **多人场景**：需要同时检测多个人
- **高精度要求**：需要更准确的检测结果  
- **分割掩码**：需要精确的人体轮廓
- **工业应用**：对准确度要求高于速度

### MediaPipe 适用于：
- **实时应用**：需要极低延迟
- **单人场景**：主要检测一个人
- **移动设备**：资源受限环境
- **详细姿态**：需要更多关键点信息
- **手部面部**：需要检测手部和面部细节

## 性能优化建议

1. **速度优先**：选择MediaPipe + 较低分辨率
2. **精度优先**：选择YOLO大模型 (yolov8l-pose.pt)
3. **平衡选择**：YOLO nano模型 (yolov8n-pose.pt)
4. **多人检测**：只能使用YOLO
5. **资源受限**：优先考虑MediaPipe

## 故障排除

### MediaPipe导入错误
```bash
# 安装MediaPipe
pip install mediapipe==0.10.9

# 或降级版本
pip install mediapipe==0.9.3.0
```

### 摄像头权限问题
1. 系统设置 > 安全性与隐私 > 摄像头
2. 允许终端应用访问摄像头
3. 重新运行程序

### 性能问题
- 降低摄像头分辨率
- 使用更快的模型
- 调整置信度阈值

## 扩展开发

添加新的检测器只需：

1. 继承 `BaseDetector` 类
2. 实现 `detect()` 和 `get_capabilities()` 方法  
3. 在 `create_detector()` 函数中注册
4. 返回统一的 `DetectionResult` 格式

这种设计让添加新模型变得非常简单！