# main_v16.py
# 升级版 v19：
# 1. 新增 PSD 频谱图 X 轴对数/线性切换功能 (按 'L' 键切换)。
#    - Linear: 0-60Hz (默认)
#    - Log: 1-100Hz (刻度 1, 10, 100)
# 2. 保持 v18 的仪表盘多边形渲染优化。
# 3. [新增] 2D 情感状态空间 (Valence-Arousal) 可视化。
# 4. [新增] 情感状态趋势记录与可视化 (集成 part.py 组件)。

import sys
import os
import time
import math
import numpy as np
import pygame
import threading
import queue
import csv
import datetime

# 添加路径以导入同级模块
sys.path.append(".")


def resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# --- 导入核心模块 ---
try:
    from bluetooth_core import EEGBluetoothReceiver
except ImportError:
    print("❌ 错误: 找不到 bluetooth_core.py，请确保该文件在同一目录下")
    sys.exit(1)

from eeg_engine import EEGAnalyzer
from part import ScatterTrendVisualizer  # [新增] 导入趋势可视化组件

# --- [关键配置] 设计分辨率 ---
DESIGN_WIDTH = 2752
DESIGN_HEIGHT = 1356

FPS = 30

# 颜色定义 (Cyberpunk Palette)
COLOR_BG = (10, 10, 15)
COLOR_TEXT = (200, 220, 255)
COLOR_FP1 = (255, 230, 50)
COLOR_FP2 = (50, 255, 255)
COLOR_RED = (255, 50, 50)
COLOR_GREEN = (50, 255, 100)
COLOR_GOLD = (50, 255, 255)
COLOR_CYAN = (255, 215, 0)
COLOR_FRAME = (40, 50, 60)
COLOR_FRAME_LIGHT = (80, 100, 120)
COLOR_GRID = (30, 40, 50)


class AsyncProcessor(threading.Thread):
    def __init__(self, receiver_queue, output_queue):
        super().__init__()
        self.receiver_queue = receiver_queue
        self.output_queue = output_queue
        self.engine = EEGAnalyzer(fs=1000)
        self.running = True
        self.daemon = True

    def run(self):
        print("⚙️ 计算线程已启动...")
        while self.running:
            try:
                data_tuple = self.receiver_queue.get(timeout=0.1)

                if isinstance(data_tuple, tuple):
                    raw_chunk, reliability = data_tuple
                else:
                    raw_chunk = data_tuple
                    reliability = None

                result = self.engine.process(raw_chunk, reliability_mask=reliability)

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 计算线程出错: {e}")

    def stop(self):
        self.running = False


class EEGVisualizer:
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()

        self.info = pygame.display.Info()
        self.screen_w = self.info.current_w
        self.screen_h = self.info.current_h

        self.is_fullscreen = False
        self.set_window_mode(fullscreen=False)
        self.canvas = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))

        try:
            self.font_large = pygame.font.SysFont("SimHei", 48, bold=True)
            self.font_small = pygame.font.SysFont("SimSun", 24)
            self.font_tiny = pygame.font.SysFont("SimSun", 18)
            self.font_num_large = pygame.font.SysFont("SimHei", 32, bold=True)
        except:
            self.font_large = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 14)
            self.font_tiny = pygame.font.Font(None, 10)
            self.font_num_large = pygame.font.Font(None, 32)

        self.line_config = {
            'enable': True,
            'pos': (DESIGN_WIDTH // 2 - 270, DESIGN_HEIGHT // 2 + 100),
            'alpha': 180,
            'scale': 1.0
        }

        print("🚀 初始化多线程架构...")
        self.bt_receiver = EEGBluetoothReceiver()
        self.ui_queue = queue.Queue(maxsize=10)
        self.processor = AsyncProcessor(self.bt_receiver.get_gui_queue(), self.ui_queue)
        self.processor.start()

        self.wave_len = 1000
        self.wave_fp1 = np.zeros(self.wave_len)
        self.wave_fp2 = np.zeros(self.wave_len)

        self.n_bars = 60
        self.bars_fp1 = np.zeros(self.n_bars)
        self.bars_fp2 = np.zeros(self.n_bars)

        self.psd_data = {
            'freqs': np.linspace(1, 60, 30).tolist(),
            'ch1': np.zeros(30).tolist(),
            'ch2': np.zeros(30).tolist()
        }

        self.current_metrics = {
            'metrics': {
                'sensory_attention': 0,
                'emotional_valence': 0,
                'emotional_arousal': 0,
                'neural_relaxation': 0,
                'cortical_arousal': 0,
                'cognitive_depth': 0,
                'immersion_ratio': 0,
                'slope': -1.0,
            },
            'powers': {'left': {}, 'right': {}},
            'signals': {'fp1': {'chunk_clean': []}, 'fp2': {'chunk_clean': []}},
            'psd_info': self.psd_data
        }

        self.smooth_metrics = self.current_metrics['metrics'].copy()
        self.load_assets()
        self.spiral_layout = self.precompute_spiral()
        self.rainbow_bar = self.create_rainbow_gradient(300, 10)

        self.fixed_scale_uv = 100.0
        self.fixed_psd_max = 1e11

        # [新增] PSD 对数坐标模式开关
        self.psd_log_mode = False

        # [新增] CSV 记录与趋势可视化
        self.record_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_records")
        os.makedirs(self.record_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.csv_filename = os.path.join(self.record_dir, f"{timestamp}.csv")
        
        print(f"📁 Recording data to: {self.csv_filename}")
        self.csv_file = open(self.csv_filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp", "valence", "arousal"])
        self.record_start_time = time.time()
        
        # 放置在底部中间
        viz_w = 800
        viz_h = 180
        viz_x = (DESIGN_WIDTH - viz_w) // 2
        viz_y = DESIGN_HEIGHT - viz_h - 10
        self.trend_viz = ScatterTrendVisualizer(viz_x, viz_y, viz_w, viz_h, "Affective State Trend", 300, fps=FPS)
        
        self.current_affect = (0.5, 0.5) # Default center

    def toggle_psd_log_mode(self):
        """切换 PSD 图表的 X 轴坐标模式 (线性/对数)"""
        self.psd_log_mode = not self.psd_log_mode
        mode_str = "Logarithmic (1-100Hz)" if self.psd_log_mode else "Linear (0-60Hz)"
        print(f"🔀 PSD X-Axis Mode Switched to: {mode_str}")

    def set_window_mode(self, fullscreen=False):
        self.is_fullscreen = fullscreen
        if fullscreen:
            target_w = self.screen_w
            target_h = self.screen_h
            flags = pygame.FULLSCREEN | pygame.DOUBLEBUF
        else:
            target_w = self.screen_w * 0.9
            target_h = self.screen_h * 0.9
            flags = pygame.RESIZABLE | pygame.DOUBLEBUF

        scale_w = target_w / DESIGN_WIDTH
        scale_h = target_h / DESIGN_HEIGHT
        self.scale_factor = min(scale_w, scale_h)

        self.window_w = int(DESIGN_WIDTH * self.scale_factor)
        self.window_h = int(DESIGN_HEIGHT * self.scale_factor)

        if fullscreen:
            self.real_screen = pygame.display.set_mode((self.screen_w, self.screen_h), flags)
        else:
            self.real_screen = pygame.display.set_mode((self.window_w, self.window_h), flags)

        pygame.display.set_caption(
            f"Neuro-Tasting System (Scale: {self.scale_factor:.2f}x) - {'Fullscreen' if fullscreen else 'Windowed'}")

    def create_rainbow_gradient(self, width, height):
        surf = pygame.Surface((width, height))
        return surf

    def load_smart(self, base_name, size=None, alpha=None, auto_fix=False):
        path = resource_path(os.path.join(r'imgs', base_name + ".png"))
        img = None
        if os.path.exists(path):
            try:
                img = pygame.image.load(path).convert_alpha()
            except:
                pass
        if img is None:
            surf = pygame.Surface(size if size else (64, 64))
            surf.fill((200, 200, 200))
            if base_name == "line": pygame.draw.line(surf, (255, 255, 255), (0, 32), (64, 32), 4)
            img = surf
        if size: img = pygame.transform.smoothscale(img, size)
        if alpha is not None: img.set_alpha(alpha)
        return img

    def load_assets(self):
        print("Loading assets...")
        self.img_bg = self.load_smart("bg", (DESIGN_WIDTH, DESIGN_HEIGHT), alpha=255)
        self.img_bg_light = self.load_smart("bg_light", (DESIGN_WIDTH, DESIGN_HEIGHT))
        self.img_center_raw = self.load_smart("center", (800, 650))
        self.img_brain_light = self.load_smart("brain_light", (800, 650))
        self.img_bar_raw = self.load_smart("bar", (180, 330))
        self.img_line = self.load_smart("line")
        steps = 50

        gradient_stops = [
            (0.00, (80, 80, 200)), (0.05, (90, 110, 210)), (0.10, (100, 140, 230)),
            (0.15, (100, 180, 240)), (0.20, (100, 210, 250)), (0.25, (100, 230, 240)),
            (0.30, (100, 240, 230)), (0.35, (100, 240, 200)), (0.40, (120, 240, 180)),
            (0.45, (150, 240, 150)), (0.50, (180, 240, 120)), (0.55, (210, 240, 100)),
            (0.60, (240, 240, 80)), (0.65, (250, 220, 70)), (0.70, (255, 200, 60)),
            (0.75, (255, 180, 50)), (0.80, (255, 160, 40)), (0.85, (255, 140, 50)),
            (0.90, (255, 120, 60)), (0.95, (255, 100, 70)), (1.00, (255, 80, 80))
        ]
        self.bar_slices_heatmap = self.generate_heatmap_slices(self.img_bar_raw, gradient_stops, steps)

    def tint_image(self, image, color):
        image = image.copy()
        colored_surf = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        colored_surf.fill(color)
        image.blit(colored_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return image

    def slice_bar_image(self, img):
        w, h = img.get_size()
        h_top, h_bottom = int(h * 0.3), int(h * 0.5)
        h_mid = h - h_top - h_bottom
        return {
            'top': img.subsurface((0, 0, w, h_top)).copy(),
            'mid': img.subsurface((0, h_top, w, h_mid)).copy(),
            'bottom': img.subsurface((0, h_top + h_mid, w, h_bottom)).copy(),
            'orig_size': (w, h), 'h_ratios': (h_top, h_mid, h_bottom)
        }

    def get_gradient_color(self, t, stops):
        for i in range(len(stops) - 1):
            t1, c1 = stops[i]
            t2, c2 = stops[i + 1]
            if t1 <= t <= t2:
                ratio = (t - t1) / (t2 - t1)
                r = int(c1[0] + (c2[0] - c1[0]) * ratio)
                g = int(c1[1] + (c2[1] - c1[1]) * ratio)
                b = int(c1[2] + (c2[2] - c1[2]) * ratio)
                return (r, g, b)
        return stops[-1][1]

    def generate_heatmap_slices(self, base_img, stops, steps=50):
        sliced_list = []
        for i in range(steps):
            t = i / (steps - 1)
            color = self.get_gradient_color(t, stops)
            tinted = self.tint_image(base_img, color)
            sliced_list.append(self.slice_bar_image(tinted))
        return sliced_list

    def precompute_spiral(self):
        layout = []
        center_x, center_y = 1376, 840
        rot_offset = math.pi * 0.65
        r_start, r_end = 600, 750
        for i in range(self.n_bars):
            prog = i / (self.n_bars - 1)
            radius = r_start + (r_end - r_start) * prog
            theta = np.linspace(0, -2 * np.pi, self.n_bars, endpoint=False)[i] + rot_offset
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta) * 0.55
            scale = 0.98 + (y - center_y) / 2000
            layout.append({'y_sort': y, 'x': x, 'y': y, 'scale': scale, 'type': 'fp1', 'idx': i})
        r_start, r_end = 840, 1050
        for i in range(self.n_bars):
            prog = i / (self.n_bars - 1)
            radius = r_start + (r_end - r_start) * prog
            theta = np.linspace(0, -2 * np.pi, self.n_bars, endpoint=False)[i] + rot_offset
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta) * 0.55
            scale = 0.98 + (y - center_y) / 2000
            layout.append({'y_sort': y, 'x': x, 'y': y, 'scale': scale, 'type': 'fp2', 'idx': i})
        layout.append({'y_sort': center_y, 'x': center_x, 'y': center_y, 'scale': 1.0, 'type': 'center', 'idx': -1})
        layout.sort(key=lambda item: item['y_sort'])
        return layout

    def catmull_rom_interpolate(self, points, num_points=20, closed=True):
        if len(points) < 3: return points
        if closed:
            ctrl_pts = [points[-1]] + points + [points[0], points[1]]
        else:
            ctrl_pts = [points[0]] + points + [points[-1]]
        smooth_points = []
        for i in range(len(points)):
            p0 = ctrl_pts[i];
            p1 = ctrl_pts[i + 1];
            p2 = ctrl_pts[i + 2];
            p3 = ctrl_pts[i + 3]
            for t in np.linspace(0, 1, num_points, endpoint=False):
                t2 = t * t
                t3 = t2 * t
                x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
                           (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                           (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
                y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
                           (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                           (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
                smooth_points.append((x, y))
        if closed: smooth_points.append(smooth_points[0])
        return smooth_points

    def calculate_affect_coords(self):
        """计算情感坐标 (Valence, Arousal)"""
        m = self.smooth_metrics
        
        # Y轴: Cortical Arousal (Left Gauge)
        a_org = m.get('cortical_arousal', 0)
        a_new = a_org / 100 * 2 - 1
        low_a, high_a = 0.3, 0.8
        a_norm = (a_new - low_a) / (high_a - low_a)
        val_y = 1 / (1 + np.exp(-10 * (np.clip(a_norm, -500, 500)-0.5)))  # 0.0 ~ 1.0

        # X轴: Emotional Valence (Right Gauge)
        b_org = m.get('emotional_valence', 0)
        b_new = b_org / 100 * 2 - 1
        low_b, high_b = 0.0, 0.75
        b_norm = (b_new - low_b) / (high_b - low_b)
        val_x = 1 / (1 +  np.exp(-10 * (np.clip(b_norm, -500, 500)-0.5)))  # 0.0 ~ 1.0
        
        return val_x, val_y

    def update_state(self):
        has_new_data = False
        while not self.ui_queue.empty():
            try:
                packet = self.ui_queue.get_nowait()
                self.current_metrics = packet
                if 'psd_info' in packet and packet['psd_info'].get('freqs'):
                    self.psd_data = packet['psd_info']
                new_chunk_fp1 = packet['signals']['fp1'].get('chunk_clean', [])
                new_chunk_fp2 = packet['signals']['fp2'].get('chunk_clean', [])
                if len(new_chunk_fp1) > 0:
                    n = len(new_chunk_fp1)
                    if n > self.wave_len:
                        n = self.wave_len
                        new_chunk_fp1 = new_chunk_fp1[-n:]
                        new_chunk_fp2 = new_chunk_fp2[-n:]
                    self.wave_fp1 = np.roll(self.wave_fp1, -n)
                    self.wave_fp1[-n:] = new_chunk_fp1
                    self.wave_fp2 = np.roll(self.wave_fp2, -n)
                    self.wave_fp2[-n:] = new_chunk_fp2
                    has_new_data = True
            except queue.Empty:
                break

        if has_new_data:
            step = 4
            req_len = self.n_bars * step
            if len(self.wave_fp1) >= req_len:
                raw_slice_1 = self.wave_fp1[-req_len::step]
                raw_slice_2 = self.wave_fp2[-req_len::step]
                OFFSET  = 60
                SCALE = OFFSET * 2
                self.bars_fp1 = np.clip((raw_slice_1 + OFFSET) / SCALE, 0.0, 1.0)[::-1]
                self.bars_fp2 = np.clip((raw_slice_2 + OFFSET) / SCALE, 0.0, 1.0)[::-1]

        alpha = 0.1
        for key in self.smooth_metrics:
            target = self.current_metrics['metrics'].get(key, 0)
            current = self.smooth_metrics.get(key, 0)
            self.smooth_metrics[key] = current + (target - current) * alpha

        # [新增] 更新情感坐标并记录
        val_x, val_y = self.calculate_affect_coords()
        self.current_affect = (val_x, val_y)
        
        # 映射到 [-1, 1] 用于 Visualizer
        v_mapped = val_x * 2 - 1
        a_mapped = val_y * 2 - 1
        
        self.trend_viz.add_point(v_mapped, a_mapped)
        
        # 写入 CSV
        elapsed = time.time() - self.record_start_time
        try:
            self.csv_writer.writerow([f"{elapsed:.3f}", f"{val_x:.4f}", f"{val_y:.4f}"])
        except Exception as e:
            print(f"CSV Write Error: {e}")

    # --- 绘图函数 ---

    def draw_combined_wave(self, x, y, width, height, data1, data2, color1, color2, title):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.canvas, (15, 20, 25), rect)
        pygame.draw.rect(self.canvas, COLOR_FRAME, rect, 2, border_radius=4)
        center_y = y + height / 2
        pygame.draw.line(self.canvas, (60, 70, 80), (x, center_y), (x + width, center_y), 1)
        scale_y = (height / 2) / self.fixed_scale_uv
        for v in [50, 100]:
            py_pos = center_y - v * scale_y
            pygame.draw.line(self.canvas, (30, 40, 50), (x, py_pos), (x + width, py_pos), 1)
            py_neg = center_y + v * scale_y
            pygame.draw.line(self.canvas, (30, 40, 50), (x, py_neg), (x + width, py_neg), 1)
            self.canvas.blit(self.font_tiny.render(f"+{v}", True, (50, 60, 70)), (x + 2, py_pos - 10))
            self.canvas.blit(self.font_tiny.render(f"-{v}", True, (50, 60, 70)), (x + 2, py_neg - 10))
        for i in range(1, 4):
            grid_x = x + i * (width / 4)
            pygame.draw.line(self.canvas, (30, 40, 50), (grid_x, y), (grid_x, y + height), 1)
            self.canvas.blit(self.font_tiny.render(f"-{1.0 - i * 0.25:.2f}s", True, (80, 100, 120)),
                             (grid_x + 2, y + height - 15))
        self.canvas.blit(self.font_small.render(title, True, COLOR_FRAME_LIGHT), (x + 10, y + 5))
        self.canvas.blit(self.font_tiny.render("FP1 (Yel)", True, color1), (x + width - 140, y + 5))
        self.canvas.blit(self.font_tiny.render("FP2 (Blu)", True, color2), (x + width - 70, y + 5))

        def draw_single_ch(data_arr, col):
            points = []
            step_x = width / len(data_arr)
            for i in range(len(data_arr)):
                val = data_arr[i]
                if np.isnan(val): val = 0
                offset_y = val * scale_y
                offset_y = max(-height / 2 + 2, min(height / 2 - 2, offset_y))
                points.append((x + i * step_x, center_y - offset_y))
            if len(points) > 1: pygame.draw.lines(self.canvas, col, False, points, 2)

        draw_single_ch(data1, color1);
        draw_single_ch(data2, color2)
        self.canvas.blit(self.font_tiny.render(f"Fixed Scale: ±{int(self.fixed_scale_uv)}uV", True, (100, 200, 100)),
                         (x + 10, y + 25))

    def draw_combined_psd(self, x, y, width, height, freqs, psd1, psd2, color1, color2, title):
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.canvas, (15, 20, 25), rect)
        pygame.draw.rect(self.canvas, COLOR_FRAME, rect, 2, border_radius=4)

        # 绘制标题，指示当前模式
        mode_str = "(Log X)" if self.psd_log_mode else "(Linear X)"
        full_title = f"{title} {mode_str}"
        self.canvas.blit(self.font_small.render(full_title, True, COLOR_FRAME_LIGHT), (x + 10, y + 5))

        self.canvas.blit(self.font_tiny.render("FP1", True, color1), (x + width - 80, y + 5))
        self.canvas.blit(self.font_tiny.render("FP2", True, color2), (x + width - 40, y + 5))

        # [核心] 根据模式定义 X 轴映射逻辑和刻度
        if self.psd_log_mode:
            # Log 模式: 1-100Hz
            min_log_f = 1.0
            max_log_f = 100.0
            x_ticks = [1, 10, 100]
            log_min = math.log10(min_log_f)
            log_max = math.log10(max_log_f)

            def get_x_ratio(f):
                if f < min_log_f: return 0.0
                return (math.log10(f) - log_min) / (log_max - log_min)
        else:
            # Linear 模式: 0-60Hz (标准)
            x_ticks = [1, 10, 30, 60]

            def get_x_ratio(f):
                return f / 60.0

        # 绘制 X 轴网格和刻度
        for freq in x_ticks:
            ratio = get_x_ratio(freq)
            # 确保在绘图范围内
            if 0 <= ratio <= 1.05:  # 稍微宽容一点允许画边框
                px = x + ratio * width
                # 限制线不画出框
                px_clamped = min(x + width, max(x, px))
                pygame.draw.line(self.canvas, COLOR_GRID, (px_clamped, y), (px_clamped, y + height), 1)
                self.canvas.blit(self.font_tiny.render(f"{freq}Hz", True, COLOR_FRAME_LIGHT),
                                 (px_clamped - 5, y + height - 15))

        # Y 轴 (Log Power) - 保持不变
        log_max = math.log10(self.fixed_psd_max)
        for val in [10, 100, 1000]:
            py = y + height - (math.log10(val) / log_max * (height - 25)) - 10
            pygame.draw.line(self.canvas, (30, 40, 50), (x, py), (x + width, py), 1)
            self.canvas.blit(self.font_tiny.render(f"{val}", True, (60, 70, 80)), (x + 2, py - 10))

        if not freqs or len(freqs) < 2: return

        # 绘制数据曲线
        def draw_single_psd(p_vals, col):
            if not p_vals or len(p_vals) < len(freqs): return
            points = []
            for i in range(len(freqs)):
                f = freqs[i]
                # Log 模式下忽略 <1Hz 的点 (避免 log 负无穷)
                if self.psd_log_mode and f < 1.0: continue
                if f > 100.0: break  # 数据一般只到 85Hz，但也防止溢出

                p = p_vals[i]
                if np.isnan(p): p = 0

                # 计算坐标
                x_ratio = get_x_ratio(f)
                y_ratio = math.log10(max(1.0, min(p, self.fixed_psd_max))) / log_max

                px = x + x_ratio * width
                py = y + height - (y_ratio * (height - 25)) - 10
                points.append((px, py))

            if len(points) > 1:
                surf = pygame.Surface((width, height), pygame.SRCALPHA)
                # 转换到相对坐标
                offset_pts = [(p[0] - x, p[1] - y) for p in points]

                # 绘制填充和线条
                pygame.draw.polygon(surf, (*col, 25),
                                    [(offset_pts[0][0], height)] + offset_pts + [(offset_pts[-1][0], height)])
                pygame.draw.lines(surf, (*col, 128), False, offset_pts, 2)
                self.canvas.blit(surf, (x, y))

        draw_single_psd(psd1, color1)
        draw_single_psd(psd2, color2)

        self.canvas.blit(self.font_tiny.render(f"Log Scale Max: {int(self.fixed_psd_max)}", True, (100, 200, 100)),
                         (x + 10, y + 25))

    def get_brighter_brain_color(self, error_rate, pos_score):
        t = min(1.0, max(0.0, error_rate))
        r = int(50 + (255 - 50) * t)
        g = int(255 - (255 - 50) * t)
        b = int(255 - (255 - 50) * t)
        brightness_boost = 1.5 + pos_score * 1.0
        return (
        min(255, int(r * brightness_boost)), min(255, int(g * brightness_boost)), min(255, int(b * brightness_boost)))

    def draw_custom_line_image(self):
        if not self.img_line: return
        t = time.time()
        breath_factor = (math.sin(t * 2.0) + 1) / 2 * 0.3 + 0.7
        final_alpha = int(self.line_config['alpha'] * breath_factor)
        scaled_img = self.img_line.copy()
        if self.line_config['scale'] != 1.0:
            w = int(scaled_img.get_width() * self.line_config['scale'])
            h = int(scaled_img.get_height() * self.line_config['scale'])
            scaled_img = pygame.transform.smoothscale(scaled_img, (w, h))
        scaled_img.set_alpha(max(0, min(255, final_alpha)))
        shake_x = math.sin(t * 3.0) * 6
        shake_y = math.cos(t * 2.5) * 5
        self.canvas.blit(scaled_img, scaled_img.get_rect(
            center=(self.line_config['pos'][0] + shake_x, self.line_config['pos'][1] + shake_y)))

    # --- [核心修改] 使用多边形绘制实心弧形 (无缝隙) ---
    def draw_solid_arc(self, surface, color, center, r_inner, r_outer, start_rad, end_rad):
        if start_rad > end_rad: start_rad, end_rad = end_rad, start_rad
        span = end_rad - start_rad
        steps = max(10, int(abs(span) / (math.pi / 90)))
        points_outer = []
        points_inner = []
        for i in range(steps + 1):
            theta = start_rad + (span * i / steps)
            cos_t = math.cos(theta); sin_t = math.sin(theta)
            points_outer.append((center[0] + r_outer * cos_t, center[1] + r_outer * sin_t))
            points_inner.append((center[0] + r_inner * cos_t, center[1] + r_inner * sin_t))
        poly_points = points_outer + points_inner[::-1]
        pygame.draw.polygon(surface, color, poly_points)

    def draw_gauge_arc_styled(self, surface, center, radius, start_angle_deg, end_angle_deg, val, color, label,
                              thickness=50, x_text=500):
        size = radius * 2 + 200
        temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        temp_center = (size // 2, size // 2)

        start_rad = math.radians(start_angle_deg)
        end_rad = math.radians(end_angle_deg)

        r_main_center = radius
        r_main_inner = r_main_center - thickness / 2
        r_main_outer = r_main_center + thickness / 2

        bg_color = (20, 40, 50, 120)
        self.draw_solid_arc(temp_surf, bg_color, temp_center, r_main_inner, r_main_outer, start_rad, end_rad)

        r_axis = r_main_inner - 15
        rect_axis = pygame.Rect(temp_center[0] - r_axis, temp_center[1] - r_axis, r_axis * 2, r_axis * 2)
        axis_color = (color[0], color[1], color[2], 150)
        pygame.draw.arc(temp_surf, axis_color, rect_axis, start_rad, end_rad, 3)

        r_tick_outer = r_main_outer + 5;
        r_tick_inner_short = r_main_inner - 5;
        r_tick_inner_long = r_axis - 10
        num_ticks = 20;
        total_angle = end_angle_deg - start_angle_deg

        for i in range(num_ticks + 1):
            angle = start_angle_deg + (total_angle * (i / num_ticks))
            rad = math.radians(angle)
            is_major = (i % 5 == 0)
            inner_r = r_tick_inner_long if is_major else r_tick_inner_short
            p1 = (temp_center[0] + inner_r * math.cos(rad), temp_center[1] + inner_r * math.sin(rad))
            p2 = (temp_center[0] + r_tick_outer * math.cos(rad), temp_center[1] + r_tick_outer * math.sin(rad))
            tick_col = (*color, 200 if is_major else 100)
            pygame.draw.line(temp_surf, tick_col, p1, p2, 3 if is_major else 2)

        val = min(1.0, max(0.0, val))
        fill_color = (*color, 220)

        y_start = math.sin(start_rad);
        y_end = math.sin(end_rad)
        span = end_angle_deg - start_angle_deg

        if y_start > y_end:
            fill_angle_start = start_angle_deg
            fill_angle_end = start_angle_deg + (span * val)
        else:
            fill_angle_start = end_angle_deg - (span * val)
            fill_angle_end = end_angle_deg

        if val > 0.005:
            self.draw_solid_arc(temp_surf, fill_color, temp_center,
                                r_main_inner + 4, r_main_outer - 4,
                                math.radians(fill_angle_start), math.radians(fill_angle_end))

            highlight_color = (255, 255, 255, 180)
            self.draw_solid_arc(temp_surf, highlight_color, temp_center,
                                r_main_inner + 4, r_main_inner + 8,
                                math.radians(fill_angle_start), math.radians(fill_angle_end))

        surface.blit(temp_surf, (center[0] - size // 2, center[1] - size // 2))
        text_surf_label = self.font_large.render(label, True, color)
        text_surf_label.set_alpha(200)
        surface.blit(text_surf_label, (x_text, 1200))
        text_surf_val = self.font_num_large.render(f"{int(val * 100)}", True, color)
        text_surf_val.set_alpha(200)
        surface.blit(text_surf_val, (x_text, 1160))

    def draw_star_radar(self, center, radius_base, powers_left, powers_right):
        keys = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']
        labels = ["DELTA", "THETA", "L-ALPHA", "H-ALPHA", "L-BETA", "H-BETA", "L-GAMMA", "H-GAMMA"]
        SCALE_FACTOR = 10.0
        s = 600;
        surf = pygame.Surface((s, s), pygame.SRCALPHA);
        ctr = (s // 2, s // 2)

        for r_scale in [0.0, 0.33, 0.66, 1.0]:
            pygame.draw.circle(surf, (100, 100, 120, 40), ctr, int(radius_base + 150 * r_scale), 1)
        for i in range(8):
            ang = i * (math.pi / 4) - math.pi / 2
            p1 = (ctr[0] + radius_base * math.cos(ang), ctr[1] + radius_base * math.sin(ang))
            p2 = (ctr[0] + (radius_base + 80) * math.cos(ang), ctr[1] + (radius_base + 80) * math.sin(ang))
            pygame.draw.line(surf, (100, 120, 140, 80), p1, p2, 1)

        def generate_star_control_points(power_dict):
            if not power_dict: return None
            vals = [power_dict.get(k, 0) for k in keys];
            pts = []
            for i in range(8):
                ang = i * (math.pi / 4) - math.pi / 2
                r = radius_base + vals[i] * SCALE_FACTOR
                pts.append((ctr[0] + r * math.cos(ang), ctr[1] + r * math.sin(ang)))
                ang_mid = ang + (math.pi / 8)
                r_mid = (radius_base + vals[i] * SCALE_FACTOR + radius_base + vals[(i + 1) % 8] * SCALE_FACTOR) * 0.45
                pts.append((ctr[0] + r_mid * math.cos(ang_mid), ctr[1] + r_mid * math.sin(ang_mid)))
            return pts

        # [核心修复] 分离填充和描边
        # 1. 预计算形状点
        pts_L = generate_star_control_points(powers_left)
        pts_R = generate_star_control_points(powers_right)

        shape_L = self.catmull_rom_interpolate(pts_L, num_points=10, closed=True) if pts_L else []
        shape_R = self.catmull_rom_interpolate(pts_R, num_points=10, closed=True) if pts_R else []

        # 2. 先画所有填充 (Fill) - 半透明背景
        if shape_L: pygame.draw.polygon(surf, (*COLOR_FP1, 30), shape_L)
        if shape_R: pygame.draw.polygon(surf, (*COLOR_FP2, 30), shape_R)

        # 3. 后画所有描边 (Stroke) - 高亮轮廓，保证永远在最上层
        if shape_L: pygame.draw.lines(surf, (*COLOR_FP1, 220), True, shape_L, 3)
        if shape_R: pygame.draw.lines(surf, (*COLOR_FP2, 220), True, shape_R, 3)

        # 4. 绘制文字标签
        for i in range(8):
            ang = i * (math.pi / 4) - math.pi / 2
            lx = ctr[0] + (radius_base + 105) * math.cos(ang)
            ly = ctr[1] + (radius_base + 105) * math.sin(ang)
            t = self.font_tiny.render(labels[i], True, (200, 220, 230))
            surf.blit(t, t.get_rect(center=(lx, ly)))

        self.canvas.blit(surf, (center[0] - s // 2, center[1] - s // 2))

    def draw_affect_grid(self, center_x, center_y, size):
        half = size // 2
        # 背景
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, (20, 25, 30, 150), (0, 0, size, size), border_radius=8)
        pygame.draw.rect(surf, COLOR_FRAME, (0, 0, size, size), 2, border_radius=8)

        # 网格线
        pygame.draw.line(surf, (60, 70, 80), (0, half), (size, half), 1)  # X axis
        pygame.draw.line(surf, (60, 70, 80), (half, 0), (half, size), 1)  # Y axis

        # 标签
        self.canvas.blit(self.font_small.render("2D Affect Space", True, COLOR_FRAME_LIGHT),
                         (center_x - half, center_y - half - 25))

        # 轴标签
        surf.blit(self.font_tiny.render("Valence (Pos)", True, COLOR_TEXT), (size - 80, half + 5))
        surf.blit(self.font_tiny.render("Arousal (High)", True, COLOR_TEXT), (half + 5, 5))

        # 获取数据 (使用预计算的值)
        val_x, val_y = self.current_affect

        # 绘制点
        # X: 0 -> Left, 1 -> Right. Center is 0.5
        # Y: 0 -> Bottom, 1 -> Top. Center is 0.5. Note: Pygame Y is down.

        px = int(val_x * size)
        py = int((1.0 - val_y) * size)  # Invert Y for display

        # Clamp
        px = max(5, min(size - 5, px))
        py = max(5, min(size - 5, py))

        # Draw Point
        pygame.draw.circle(surf, (*COLOR_GOLD, 100), (px, py), 10)
        pygame.draw.circle(surf, COLOR_FP1, (px, py), 5)

        # Draw Coordinates text
        coord_str = f"V:{val_x:.2f} A:{val_y:.2f}"
        surf.blit(self.font_tiny.render(coord_str, True, (150, 160, 170)), (5, size - 20))

        self.canvas.blit(surf, (center_x - half, center_y - half))

    def draw_ui(self):
        m = self.smooth_metrics
        # 左侧：皮层唤醒 (从下往上填) - 自动识别 start Y > end Y

        a_org = m.get('cortical_arousal', 0)
        a_new = a_org / 100 * 2 - 1
        low = 0.3
        high = 0.8
        a_new = (a_new - low ) / (high - low)
        a_new = 1 / (1 + np.exp(-10 * (np.clip(a_new, -500, 500)-0.5)))  # 0.0 ~ 1.0

        b_org = m.get('emotional_valence', 0)
        b_new = b_org / 100 * 2 - 1
        low = 0.0
        high = 0.75
        b_new = (b_new - low ) / (high - low)
        b_new = 1 / (1 +  np.exp(-10 * (np.clip(b_new, -500, 500)-0.5)))  # 0.0 ~ 1.0
        # print("b_new:", b_new)
        self.draw_gauge_arc_styled(self.canvas, (950, 650), 800, 135, 225,
                                   a_new, COLOR_CYAN, "αPower value",
                                   x_text=500)
        # 右侧：情感愉悦 (从下往上填) - 自动识别 end Y > start Y
        self.draw_gauge_arc_styled(self.canvas, (DESIGN_WIDTH - 950, 650), 800, 315, 405,
                                   b_new , COLOR_GOLD, "αPower  ratio ",
                                   x_text=2200)

        if 'left' in self.current_metrics.get('powers', {}):
            self.draw_star_radar((550, 950), 60, self.current_metrics['powers']['left'],
                                 self.current_metrics['powers']['right'])
        
        # [新增] 绘制 2D 状态空间 (Arousal vs Valence)
        self.draw_affect_grid(DESIGN_WIDTH - 550, 950, 300)
        
        # [新增] 绘制趋势图
        self.trend_viz.draw(self.canvas)

        self.draw_metrics_panel()

    def draw_spiral(self, metrics):
        val_arousal = self.smooth_metrics.get('emotional_arousal', 0) / 100.0
        val_valence = self.smooth_metrics.get('emotional_valence', 50) / 100.0
        t = time.time()
        alpha_light = int(255 * math.pow((math.sin(t * 1.2 + 1) + 1) / 2, 10.0) * (0.2 + val_valence * 0.8))
        self.img_bg.set_alpha(255 - alpha_light)
        self.canvas.blit(self.img_bg, self.img_bg.get_rect(center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)))
        self.img_bg_light.set_alpha(alpha_light)
        self.canvas.blit(self.img_bg_light, self.img_bg_light.get_rect(center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)))

        for item in self.spiral_layout:
            if item['type'] == 'center':
                cx, cy = item['x'], item['y'] - 300
                hover = math.sin(t * 0.8) * 8
                pulse = 1.0 + math.pow((math.sin(t * 2) + 1) / 2, 15.0) * 0.03
                brain = self.tint_image(self.img_center_raw, self.get_brighter_brain_color(val_arousal, val_valence))
                w, h = int(brain.get_width() * pulse), int(brain.get_height() * pulse)
                self.canvas.blit(pygame.transform.scale(brain, (w, h)), brain.get_rect(center=(cx, cy + hover)))
                l_alpha = int(255 * math.pow((math.sin(t * 2) + 1) / 2, 15.0))
                if l_alpha > 5:
                    light = pygame.transform.scale(self.img_brain_light, (w, h))
                    light.set_alpha(l_alpha)
                    self.canvas.blit(light, light.get_rect(center=(cx, cy + hover)),
                                     special_flags=pygame.BLEND_RGBA_ADD)
                if self.line_config['enable']: self.draw_custom_line_image()
                continue

            idx = item['idx'];
            val = self.bars_fp1[idx] if item['type'] == 'fp1' else self.bars_fp2[idx]
            slice_idx = max(0, min(len(self.bar_slices_heatmap) - 1, int(val * (len(self.bar_slices_heatmap) - 1))))
            slices = self.bar_slices_heatmap[slice_idx]
            scale = item['scale']
            h_top, h_mid_orig, h_bot = slices['h_ratios'];
            w_orig, _ = slices['orig_size']
            tw = max(1, int(w_orig * scale * 0.5));
            th_mid = int(max(1, val * scale * 250.0))

            # 提升柱子亮度，底限设为 160
            base_alpha = 0
            alpha_bar = int(base_alpha + (255 - base_alpha) * ((math.cos(idx / self.n_bars * math.pi) + 1) / 2))
            if alpha_bar < 5: continue

            s_top = pygame.transform.scale(slices['top'], (tw, int(h_top * scale * 0.5)))
            s_mid = pygame.transform.scale(slices['mid'], (tw, th_mid))
            s_bot = pygame.transform.scale(slices['bottom'], (tw, int(h_bot * scale * 0.5)))
            for s in [s_top, s_mid, s_bot]: s.set_alpha(alpha_bar)
            x = item['x'] - tw // 2;
            y_bot = item['y'] - int(h_bot * scale * 0.5)
            self.canvas.blit(s_bot, (x, y_bot));
            self.canvas.blit(s_mid, (x, y_bot - th_mid))
            self.canvas.blit(s_top, (x, y_bot - th_mid - int(h_top * scale * 0.5)))

    def draw_metrics_panel(self):
        panel_x, panel_y = 340, 350
        panel_w, panel_h = 550, 420
        surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (*COLOR_FRAME, 180), (0, 0, panel_w, panel_h), 2, border_radius=8)
        pygame.draw.rect(surf, (10, 15, 20, 180), (2, 2, panel_w - 4, panel_h - 4), border_radius=6)
        surf.blit(self.font_small.render("Sensory Evaluation Metrics (感官评估指标)", True, (*COLOR_FRAME_LIGHT, 200)),
                  (15, 10))
        metrics = [
            ("Sensory Attention", 'sensory_attention', 100.0, ""),
            ("Overall Efficacy", 'emotional_arousal', 100.0, ""),
            ("Relaxation", 'neural_relaxation', 100.0, ""),
            ("Cognitive Depth", 'cognitive_depth', 100.0, ""),
            ("Immersion Ratio", 'immersion_ratio', 2.0, ""),
            ("Spectral Stability", 'slope', -5.0, ""),
        ]
        for i, (lbl, key, max_v, unit) in enumerate(metrics):
            val = self.smooth_metrics.get(key, 0)
            y = 50 + i * 55
            surf.blit(self.font_tiny.render(lbl, True, (*COLOR_TEXT, 220)), (20, y))
            pygame.draw.rect(surf, (40, 45, 55, 150), (20, y + 25, 380, 8), border_radius=4)
            if key in ['sensory_attention', 'emotional_arousal']:
                if np.isnan(val) or np.isinf(val):
                    val = 0  # 设置默认值
                fill_w = int(np.clip(val, 0, 100))
                center_x = 210
                if fill_w > 50:
                    r = pygame.Rect(center_x, y + 25, int((fill_w - 50) * 190 / 50), 8); c = COLOR_GOLD
                elif fill_w < 50:
                    w = int((50 - fill_w) * 190 / 50); r = pygame.Rect(center_x - w, y + 25, w, 8); c = COLOR_CYAN
                else:
                    r = None
                if r: pygame.draw.rect(surf, (*c, 200), r, border_radius=4)
                pygame.draw.line(surf, (255, 255, 255, 100), (center_x, y + 20), (center_x, y + 38), 1)
            elif key == 'slope':
                norm = np.clip(abs(val) / 5.0, 0, 1);
                r = pygame.Rect(20, y + 25, int(norm * 380), 8)
                pygame.draw.rect(surf, (*(COLOR_GREEN if norm > 0.25 else COLOR_RED), 200), r, border_radius=4)
            else:
                norm = np.clip(val / (max_v if max_v else 100), 0, 1);
                r = pygame.Rect(20, y + 25, int(norm * 380), 8)
                pygame.draw.rect(surf, (*COLOR_FP2, 200), r, border_radius=4)
            surf.blit(self.font_tiny.render(f"{val:.2f}{unit}", True, (*COLOR_GOLD, 200)), (420, y + 18))
        self.canvas.blit(surf, (panel_x, panel_y))

    def run(self):
        print("System Ready. GUI Loop Running.")
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.csv_file.close()
                    self.processor.stop(); pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.csv_file.close()
                        self.processor.stop(); pygame.quit(); sys.exit()
                    elif event.key == pygame.K_F11 or event.key == pygame.K_f:
                        self.set_window_mode(not self.is_fullscreen)
                    # [新增] 按 'L' 键切换 PSD 坐标模式
                    elif event.key == pygame.K_l:
                        self.toggle_psd_log_mode()
                elif event.type == pygame.VIDEORESIZE and not self.is_fullscreen:
                    self.window_w, self.window_h = event.w, event.h
                    self.real_screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
                    self.scale_factor = min(self.window_w / DESIGN_WIDTH, self.window_h / DESIGN_HEIGHT)
            self.update_state()
            self.canvas.fill(COLOR_BG)
            self.draw_spiral(self.current_metrics)
            self.draw_ui()
            self.draw_combined_wave(50, 50, 450, 250, self.wave_fp1, self.wave_fp2, COLOR_FP1, COLOR_FP2,
                                    "Raw EEG Comparison")
            # draw_combined_psd 会自动根据 self.psd_log_mode 决定如何绘制
            self.draw_combined_psd(DESIGN_WIDTH - 500, 50, 450, 250, self.psd_data['freqs'], self.psd_data['ch1'],
                                   self.psd_data['ch2'], COLOR_FP1, COLOR_FP2, "Spectrum Comparison")
            scaled = pygame.transform.smoothscale(self.canvas, (
            int(DESIGN_WIDTH * self.scale_factor), int(DESIGN_HEIGHT * self.scale_factor)))
            self.real_screen.fill((0, 0, 0))
            self.real_screen.blit(scaled, ((self.real_screen.get_width() - scaled.get_width()) // 2,
                                           (self.real_screen.get_height() - scaled.get_height()) // 2))
            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    app = EEGVisualizer()
    app.run()