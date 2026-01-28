import pygame
import numpy as np
import math
from collections import deque
import random
import csv
import os

import colorsys
# --- 配置参数 ---
WINDOW_W, WINDOW_H = 1200, 900  # 垂直排列
FPS = 60

# Cyberpunk Palette
COLOR_BG = (5, 5, 8)
COLOR_PANEL_BG = (8, 10, 14)
COLOR_GRID = (30, 40, 50)
COLOR_AXIS = (60, 70, 80)
COLOR_TEXT = (200, 220, 255)

# State Colors (用于底部条带 & 散点映射)
# Positive
COLOR_POS_HIGH = (255, 215, 0)  # Gold (Flow)
COLOR_POS_LOW = (0, 255, 255)  # Cyan (Relax)
# Negative
COLOR_NEG_HIGH = (255, 50, 50)  # Red (Anxiety)
COLOR_NEG_LOW = (60, 60, 200)  # Blue (Boredom)


# --- 1. 数据生成器 (用于创建 CSV) ---
class DataGenerator:
    """
    用于生成模拟数据的逻辑核心。
    不再直接驱动UI，而是用于写入文件。
    """

    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.0
        self.target_v = 0.0
        self.target_a = 0.0
        self.state_timer = 0
        self.current_state = "Neutral"

    def step(self):
        """生成下一帧数据"""
        self.state_timer -= 1
        if self.state_timer <= 0:
            self.state_timer = random.randint(60, 180)
            rand_val = random.random()

            if rand_val < 0.25:
                self.current_state = "Flow (High Val, High Aro)"
                self.target_v = 0.8;
                self.target_a = 0.6
            elif rand_val < 0.5:
                self.current_state = "Anxiety (Low Val, High Aro)"
                self.target_v = -0.6;
                self.target_a = 0.8
            elif rand_val < 0.75:
                self.current_state = "Relax (High Val, Low Aro)"
                self.target_v = 0.5;
                self.target_a = -0.5
            else:
                self.current_state = "Boredom (Low Val, Low Aro)"
                self.target_v = -0.6;
                self.target_a = -0.6

        noise_v = (random.random() - 0.5) * 0.02
        noise_a = (random.random() - 0.5) * 0.02

        self.valence += (self.target_v - self.valence) * 0.05 + noise_v
        self.arousal += (self.target_a - self.arousal) * 0.05 + noise_a

        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(-1.0, min(1.0, self.arousal))

        return self.valence, self.arousal, self.current_state


def generate_dummy_csv(filename, duration_min=5):
    """
    生成一个 CSV 文件，包含 timestamp, valence, arousal, state
    """
    print(f"Generating dummy data to {filename} ({duration_min} mins)...")
    generator = DataGenerator()
    total_frames = duration_min * 60 * 60  # 60 FPS

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "valence", "arousal", "state"])

        for i in range(total_frames):
            v, a, state = generator.step()
            ts = i / 60.0
            writer.writerow([f"{ts:.3f}", f"{v:.4f}", f"{a:.4f}", state])

    print("Generation complete.")


# --- 2. 数据读取器 (用于驱动 UI) ---
class CSVDataReader:
    """
    读取本地 CSV 文件并模拟数据流。
    这实现了数据源与可视化的解耦。
    """

    def __init__(self, filename, loop=True):
        self.filename = filename
        self.loop = loop
        self.data = []
        self.index = 0
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.filename):
            print(f"Error: File {self.filename} not found.")
            return

        print(f"Loading data from {self.filename}...")
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    # timestamp, valence, arousal, state
                    self.data.append({
                        'ts': float(row[0]),
                        'v': float(row[1]),
                        'a': float(row[2]),
                        'state': row[3]
                    })
            print(f"Loaded {len(self.data)} frames.")
        except Exception as e:
            print(f"Error reading CSV: {e}")
            self.data = []

    def get_next_frame(self):
        """返回下一帧: (valence, arousal, state)"""
        if not self.data:
            return 0.0, 0.0, "No Data"

        frame = self.data[self.index]
        self.index += 1

        if self.index >= len(self.data):
            if self.loop:
                self.index = 0
            else:
                self.index = len(self.data) - 1  # Stay at end

        return frame['v'], frame['a'], frame['state']


# --- 3. 可视化组件 (保持不变，只关注数据输入) ---
class ScatterTrendVisualizer:
    def __init__(self, x, y, w, h, title, time_window_sec, fps=60):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.max_frames = time_window_sec * fps

        self.history = deque(maxlen=self.max_frames)
        self.current_frame = 0

        try:
            self.font = pygame.font.SysFont("Arial", 12)
            self.font_large = pygame.font.SysFont("Arial", 16, bold=True)
        except:
            self.font = pygame.font.Font(None, 12)
            self.font_large = pygame.font.Font(None, 16)

        self.n_bins = 60


    def get_valence_color(self, val):
        # # 将 -1 到 1 映射为 1 到 0 (1是蓝色，0是红色)
        # # 因为在 HSV 中，0 是红色，0.66(240度) 是蓝色
        # t = (val + 1) / 2.0
        # t = max(0.0, min(1.0, t))
        #
        # # 线性映射色相：从 0.66 (蓝) 降至 0.0 (红)
        # hue = (1.0 - t) * 0.66
        #
        # # 转换为 RGB (colorsys 返回 0-1 之间的浮点数)
        # rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        """
        [修改版] 双色发散色阶
        val > 0: 红色 (Red), 饱和度随数值增加
        val < 0: 蓝色 (Blue), 饱和度随数值绝对值增加
        val = 0: 白色/灰色
        """
        # 1. 决定色相 (Hue)
        if val >= 0:
            hue = 0.0  # Red (0度)
        else:
            hue = 0.6667  # Blue (240度)

        # 2. 决定饱和度 (Saturation) - 绝对值越大越饱和
        sat = min(1.0, abs(val))

        # 3. 决定亮度 (Value) - 保持高亮度，确保中间值是白/灰而不是黑
        # 饱和度为0时，颜色由 Value 决定 (White/Grey)
        value = 0.95

        # 转换为 RGB
        rgb = colorsys.hsv_to_rgb(hue, sat, value)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


    def get_point_color(self, val, aro):
        """
        [修改版] 双色发散色阶
        val > 0: 红色 (Red), 饱和度随数值增加
        val < 0: 蓝色 (Blue), 饱和度随数值绝对值增加
        val = 0: 白色/灰色
        """
        # 1. 决定色相 (Hue)
        if val >= 0:
            hue = 0.0  # Red (0度)
        else:
            hue = 0.6667  # Blue (240度)

        # 2. 决定饱和度 (Saturation) - 绝对值越大越饱和
        sat = min(1.0, abs(val))

        # 3. 决定亮度 (Value) - 保持高亮度，确保中间值是白/灰而不是黑
        # 饱和度为0时，颜色由 Value 决定 (White/Grey)
        value = 0.95

        # 转换为 RGB
        rgb = colorsys.hsv_to_rgb(hue, sat, value)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def add_point(self, val, aro):
        self.current_frame += 1
        # point_color = self.get_valence_color(val, aro)
        point_color = self.get_valence_color(val)
        self.history.append({
            'v': val, 'a': aro,
            't': self.current_frame,
            'c': point_color
        })

    def draw_state_strip(self, screen):
        strip_h = 6
        strip_y = self.rect.bottom - strip_h - 2

        if len(self.history) < 10: return

        bins = [[] for _ in range(self.n_bins)]
        latest_t = self.current_frame
        window_size = self.max_frames
        start_t = max(0, latest_t - window_size)

        for p in self.history:
            rel_t = (p['t'] - start_t) / window_size
            if 0 <= rel_t < 1.0:
                b_idx = int(rel_t * self.n_bins)
                if b_idx < self.n_bins:
                    bins[b_idx].append(p)

        bin_w = self.rect.w / self.n_bins

        for i, bin_data in enumerate(bins):
            if not bin_data: continue

            counts = {'Flow': 0, 'Anxiety': 0, 'Boredom': 0, 'Relax': 0}
            for p in bin_data:
                if p['v'] >= 0 and p['a'] >= 0:
                    counts['Flow'] += 1
                elif p['v'] < 0 and p['a'] >= 0:
                    counts['Anxiety'] += 1
                elif p['v'] >= 0 and p['a'] < 0:
                    counts['Relax'] += 1
                else:
                    counts['Boredom'] += 1

            dom = max(counts, key=counts.get)
            colors = {
                'Flow': COLOR_POS_HIGH,
                'Anxiety': COLOR_NEG_HIGH,
                'Boredom': COLOR_NEG_LOW,
                'Relax': COLOR_POS_LOW
            }
            color = colors[dom]

            rect = (self.rect.x + i * bin_w, strip_y, math.ceil(bin_w), strip_h)
            pygame.draw.rect(screen, color, rect)

    def draw_tiny_point(self, screen, color, x, y):
        pygame.draw.circle(screen, color, (x, y), 1)

    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_PANEL_BG, self.rect)

        cy = self.rect.centery - 10
        h_avail = (self.rect.h - 50) / 2

        pygame.draw.line(screen, (*COLOR_AXIS, 80), (self.rect.x, cy), (self.rect.right, cy), 1)

        latest_t = self.current_frame
        window_size = self.max_frames
        start_t = max(0, latest_t - window_size)

        step = max(1, len(self.history) // 1500)

        for i, p in enumerate(self.history):
            if i % step != 0: continue

            rel_t = (p['t'] - start_t) / window_size
            if rel_t < 0: continue

            px = int(self.rect.x + rel_t * self.rect.w)
            py = int(cy - p['a'] * h_avail)

            if not self.rect.collidepoint(px, py): continue

            self.draw_tiny_point(screen, p['c'], px, py)

        self.draw_state_strip(screen)

        pygame.draw.rect(screen, COLOR_BG, (self.rect.x, self.rect.y, self.rect.w, 25))
        pygame.draw.line(screen, (40, 50, 60), (self.rect.x, self.rect.y + 25), (self.rect.right, self.rect.y + 25), 1)

        title_surf = self.font_large.render(self.title, True, COLOR_TEXT)
        screen.blit(title_surf, (self.rect.x + 8, self.rect.y + 4))

        lbl_col = (100, 100, 100)
        screen.blit(self.font.render("High Energy", True, lbl_col), (self.rect.x + 8, self.rect.y + 30))
        screen.blit(self.font.render("Low Energy", True, lbl_col), (self.rect.x + 8, self.rect.bottom - 20))

        legend_x = self.rect.right - 180
        screen.blit(self.font.render("Color = Valence (Pos/Neg)", True, COLOR_GRID), (legend_x, self.rect.y + 6))


# --- 4. 状态分析与报告组件 (新增) ---
class StateAnalyzer:
    def __init__(self):
        self.history = []  # (timestamp, valence, arousal)

    def add_data(self, ts, v, a):
        self.history.append((ts, v, a))

    def get_quadrant(self, v, a):
        if v >= 0 and a >= 0: return "Flow"
        if v < 0 and a >= 0: return "Anxiety"
        if v >= 0 and a < 0: return "Relax"
        if v < 0 and a < 0: return "Boredom"
        return "Unknown"

    def analyze(self, output_file):
        if not self.history:
            return "No data recorded."

        start_time = self.history[0][0]
        end_time = self.history[-1][0]
        duration = end_time - start_time

        quadrant_counts = {"Flow": 0, "Anxiety": 0, "Relax": 0, "Boredom": 0}

        # 1. Basic Stats
        for _, v, a in self.history:
            q = self.get_quadrant(v, a)
            if q in quadrant_counts:
                quadrant_counts[q] += 1

        total_points = len(self.history)
        report = []
        report.append(f"=== SESSION REPORT ===")
        report.append(f"Duration: {duration:.2f} seconds")
        report.append(f"Total Data Points: {total_points}")
        report.append("-" * 20)
        report.append("State Distribution:")
        for q, count in quadrant_counts.items():
            pct = (count / total_points) * 100 if total_points > 0 else 0
            report.append(f"  {q}: {pct:.1f}%")

        # 2. Stability Analysis (Simple: Longest continuous sequence in a quadrant)
        report.append("-" * 20)
        report.append("Stability Analysis (Continuous State > 5s):")

        current_q = None
        current_start_ts = 0
        
        stable_events = []

        for ts, v, a in self.history:
            q = self.get_quadrant(v, a)
            if q != current_q:
                # State changed
                if current_q is not None:
                    dur = ts - current_start_ts
                    if dur >= 5.0:
                        stable_events.append((current_q, dur, current_start_ts))
                current_q = q
                current_start_ts = ts

        # Check last segment
        if current_q is not None and self.history:
            dur = self.history[-1][0] - current_start_ts
            if dur >= 5.0:
                stable_events.append((current_q, dur, current_start_ts))

        if not stable_events:
            report.append("  No stable states detected (>5s).")
        else:
            for q, dur, start in stable_events:
                report.append(f"  {q}: {dur:.1f}s (started at {start:.1f}s)")

        report_text = "\n".join(report)

        # Write to file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")

        return report_text


# --- 5. 简单 UI 按钮 (新增) ---
class SimpleButton:
    def __init__(self, x, y, w, h, text, font, color_bg, color_hover, color_text, action):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.color_bg = color_bg
        self.color_hover = color_hover
        self.color_text = color_text
        self.action = action
        self.is_hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered and event.button == 1:
                if self.action:
                    self.action()

    def draw(self, surface):
        color = self.color_hover if self.is_hovered else self.color_bg
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2, border_radius=8)

        txt_surf = self.font.render(self.text, True, self.color_text)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        surface.blit(txt_surf, txt_rect)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Neuro-Tasting Scatter Trend Observer (CSV Playback)")
    clock = pygame.time.Clock()

    # --- 阶段 1: 数据生成 (模拟离线数据) ---
    csv_filename = "session_data_dummy.csv"
    # 如果文件不存在，生成 5 分钟的模拟数据
    if not os.path.exists(csv_filename):
        generate_dummy_csv(csv_filename, duration_min=5)
    else:
        print(f"Using existing data: {csv_filename}")

    # --- 阶段 2: 数据读取 (模拟接入主程序) ---
    # 这里初始化 reader，代替原来的 simulator
    data_reader = CSVDataReader(csv_filename, loop=True)

    # 初始化 UI 面板
    panel_h = WINDOW_H // 3

    # 1. Micro (10s)
    viz_micro = ScatterTrendVisualizer(0, 0, WINDOW_W, panel_h, "INSTANT (10s) - Micro Texture", 10)

    # 2. Meso (60s)
    viz_meso = ScatterTrendVisualizer(0, panel_h, WINDOW_W, panel_h, "FLOW (60s) - Short Term Trend", 60)

    # 3. Macro (5min)
    viz_macro = ScatterTrendVisualizer(0, panel_h * 2, WINDOW_W, panel_h, "SESSION (5min) - The Journey", 300)

    running = True
    current_state_text = "Init"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # 按 R 键重新生成数据
                    generate_dummy_csv(csv_filename, duration_min=5)
                    data_reader = CSVDataReader(csv_filename, loop=True)

        # --- 核心修改: 从 CSV 读取器获取下一帧 ---
        # 替代了原来的 simulator.update()
        v, a, state = data_reader.get_next_frame()
        current_state_text = state

        # 更新所有可视化组件
        viz_micro.add_point(v, a)
        viz_meso.add_point(v, a)
        viz_macro.add_point(v, a)

        # 绘制
        screen.fill(COLOR_BG)

        viz_micro.draw(screen)
        viz_meso.draw(screen)
        viz_macro.draw(screen)

        # 显示当前状态文本
        state_surf = viz_micro.font_large.render(f"PLAYBACK STATE: {current_state_text}", True, COLOR_GRID)
        screen.blit(state_surf, (20, WINDOW_H - 30))

        # 提示信息
        hint_surf = viz_micro.font.render("Press 'R' to regenerate new random data", True, (100, 100, 100))
        screen.blit(hint_surf, (WINDOW_W - 250, WINDOW_H - 30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()