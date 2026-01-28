import pygame
import os
import sys
import platform

# --- 系统设置 ---
DESIGN_WIDTH = 2752
DESIGN_HEIGHT = 1356
FPS = 30


# --- [核心修改] 智能用户数据路径 ---
def get_user_data_dir():
    """
    获取跨平台的用户数据存储目录 (例如 Windows 的 My Documents)
    用于解决打包后无法写入或数据丢失的问题
    """
    app_name = "NeuroTasting_Data"  # 您的程序数据文件夹名

    if platform.system() == "Windows":
        # 获取 Windows 文档路径
        base_path = os.path.join(os.path.expanduser("~"), "Documents")
    else:
        # Mac / Linux
        base_path = os.path.join(os.path.expanduser("~"), "Documents")

    # 完整数据路径
    data_path = os.path.join(base_path, app_name)

    # 如果文件夹不存在，自动创建
    if not os.path.exists(data_path):
        try:
            os.makedirs(data_path)
            print(f"Created data directory: {data_path}")
        except Exception as e:
            print(f"Error creating data dir: {e}")
            return "."  # 回退到当前目录

    return data_path


# --- [核心修改] 资源路径辅助 ---
def resource_path(relative_path):
    """ 
    获取资源绝对路径 (适配 PyInstaller 打包) 
    PyInstaller 会将资源解压到 sys._MEIPASS 临时目录
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# --- 颜色定义 (Cyberpunk Palette) ---
COLOR_BG = (25, 35, 60)  # 低饱和度深蓝色背景
COLOR_PANEL_BG = (16, 20, 25, 200)  # 半透明黑
COLOR_TEXT = (200, 220, 255)
COLOR_TEXT_DIM = (100, 120, 140)

# 指示灯颜色
COLOR_RED = (255, 50, 50)
COLOR_GREEN = (50, 255, 100)
COLOR_YELLOW = (255, 200, 50)

# 视觉元素颜色
COLOR_FP1 = (255, 230, 50)
COLOR_FP2 = (50, 255, 255)
COLOR_GOLD = (255, 215, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_FRAME = (40, 50, 60)
COLOR_FRAME_LIGHT = (80, 100, 120)
COLOR_GRID = (30, 40, 50)


# 字体加载辅助
def get_font(name, size, bold=False):
    try:
        font = pygame.font.SysFont(name, size, bold=bold)
    except:
        font = pygame.font.Font(None, size)
    return font