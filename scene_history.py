import pygame
import csv
import os
import math
import colorsys
import time
from config import *  # 确保 config.py 中有 get_user_data_dir 和 resource_path
from ui_components import Button

# --- 1. 尝试导入 Tkinter (用于文件对话框) ---
try:
    import tkinter as tk
    from tkinter import filedialog

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: 'tkinter' module not found. Save dialogs disabled (will auto-save).")

# --- 2. 尝试导入 PIL (用于 PDF 导出) ---
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL (Pillow) library not found. Export will be PNG only.")

# --- 状态颜色定义 ---
COLOR_POS_HIGH = (255, 215, 0)  # Gold (Meditation)
COLOR_POS_LOW = (0, 255, 255)  # Cyan (Focus)
COLOR_NEG_HIGH = (255, 50, 50)  # Red (Anxiety)
COLOR_NEG_LOW = (60, 60, 200)  # Blue (Fatigue)

COLOR_AXIS = (80, 90, 100)
COLOR_AXIS_TXT = (150, 160, 170)


class StaticTrendVisualizer:
    def __init__(self, x, y, w, h, title, data_points):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.points = data_points

        # 计算时间范围
        if not self.points:
            self.total_duration = 10.0
        else:
            start_t = self.points[0]['t']
            end_t = self.points[-1]['t']
            self.total_duration = max(1.0, end_t - start_t)
            for p in self.points:
                p['t_rel'] = p['t'] - start_t

        self.stats = self.calculate_statistics()

        # 字体初始化
        try:
            self.font = pygame.font.SysFont("Arial", 20)
            self.font_small = pygame.font.SysFont("Arial", 16)
            self.font_large = pygame.font.SysFont("SimHei", 32, bold=True)
            self.font_label = pygame.font.SysFont("SimHei", 20)
            self.font_summary = pygame.font.SysFont("SimHei", 22)
        except:
            self.font = pygame.font.Font(None, 24)
            self.font_small = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 40)
            self.font_label = pygame.font.Font(None, 24)
            self.font_summary = pygame.font.Font(None, 24)

        # 布局参数
        self.margin_left = 80
        self.margin_right = 100
        self.chart_h = int(h * 0.45)

        self.strips_start_y = self.rect.y + self.chart_h + 30
        self.strip_height = 35
        self.strip_gap = 45
        self.strips_total_h = (self.strip_height * 3) + (self.strip_gap * 2)

        self.axis_y = self.strips_start_y + self.strips_total_h + 15
        self.legend_y = self.axis_y + 50
        self.summary_y = self.legend_y + 60

    def calculate_statistics(self):
        if not self.points:
            return {"total_duration": 0, "counts": {}, "percents": {}, "dominant": "None"}

        counts = {'Meditation': 0, 'Focus': 0, 'Anxiety': 0, 'Fatigue': 0}
        total_pts = len(self.points)

        for p in self.points:
            v, a = p['v'], p['a']
            if v >= 0 and a >= 0:
                counts['Meditation'] += 1
            elif v >= 0 and a < 0:
                counts['Focus'] += 1
            elif v < 0 and a >= 0:
                counts['Anxiety'] += 1
            else:
                counts['Fatigue'] += 1

        percents = {k: (v / total_pts * 100) for k, v in counts.items()}
        dominant = max(counts, key=counts.get)

        return {
            "total_duration": self.total_duration,
            "counts": counts,
            "percents": percents,
            "dominant": dominant
        }

    def get_point_color(self, val, aro):
        if val >= 0:
            hue = 0.0  # Red
        else:
            hue = 0.6667  # Blue
        sat = min(1.0, abs(val))
        value = 0.95
        rgb = colorsys.hsv_to_rgb(hue, sat, value)
        return (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

    def draw_scatter_plot(self, screen):
        chart_x = self.rect.x + self.margin_left
        chart_w = self.rect.w - self.margin_left - self.margin_right
        chart_rect = pygame.Rect(chart_x, self.rect.y + 60, chart_w, self.chart_h)

        pygame.draw.rect(screen, (15, 18, 22), chart_rect)
        pygame.draw.rect(screen, COLOR_FRAME, chart_rect, 1)

        cy = chart_rect.centery
        pygame.draw.line(screen, (40, 50, 60), (chart_rect.x, cy), (chart_rect.right, cy), 1)

        h_half = chart_rect.h / 2
        step = max(1, len(self.points) // 3000)

        for i in range(0, len(self.points), step):
            p = self.points[i]
            x_ratio = p['t_rel'] / self.total_duration
            px = int(chart_rect.x + x_ratio * chart_rect.w)

            py = int(cy - p['a'] * (h_half - 10))
            py = max(chart_rect.top + 2, min(chart_rect.bottom - 2, py))

            if chart_rect.collidepoint(px, py):
                col = self.get_point_color(p['v'], p['a'])
                screen.set_at((px, py), col)
                if step == 1:
                    pygame.draw.circle(screen, col, (px, py), 1)

        return chart_rect

    def draw_y_axis(self, screen, chart_rect):
        x = chart_rect.x
        pygame.draw.line(screen, COLOR_AXIS, (x, chart_rect.top), (x, chart_rect.bottom), 2)
        ticks = [(chart_rect.top, "+1.0"), (chart_rect.centery, "0.0"), (chart_rect.bottom, "-1.0")]
        for y_pos, label in ticks:
            pygame.draw.line(screen, COLOR_AXIS, (x - 5, y_pos), (x, y_pos), 1)
            lbl_surf = self.font_small.render(label, True, COLOR_AXIS_TXT)
            lbl_rect = lbl_surf.get_rect(midright=(x - 8, y_pos))
            screen.blit(lbl_surf, lbl_rect)

        title_surf = self.font_label.render("兴奋度", True, (200, 200, 200))
        title_surf = pygame.transform.rotate(title_surf, 90)
        title_rect = title_surf.get_rect(center=(self.rect.x + 30, chart_rect.centery))
        screen.blit(title_surf, title_rect)

    def draw_colorbar(self, screen, chart_rect):
        bar_x = chart_rect.right + 25
        bar_y = chart_rect.top
        bar_w = 20
        bar_h = chart_rect.height

        steps = 100
        for i in range(steps):
            ratio = i / (steps - 1)
            val = 1.0 - (ratio * 2.0)
            col = self.get_point_color(val, 0)
            rect_y = bar_y + int(ratio * bar_h)
            rect_h = math.ceil(bar_h / steps) + 1
            pygame.draw.rect(screen, col, (bar_x, rect_y, bar_w, rect_h))

        pygame.draw.rect(screen, COLOR_FRAME, (bar_x, bar_y, bar_w, bar_h), 1)

        col_red = (255, 80, 80)
        col_blue = (80, 80, 255)
        top_surf = self.font_label.render("高专注", True, col_red)
        screen.blit(top_surf, (bar_x + bar_w + 8, bar_y))
        bot_surf = self.font_label.render("低专注", True, col_blue)
        screen.blit(bot_surf, (bar_x + bar_w + 8, bar_y + bar_h - 20))

    def draw_x_axis(self, screen, chart_rect):
        y = self.axis_y
        total_sec = self.total_duration
        if total_sec < 60:
            interval = 5
        elif total_sec < 300:
            interval = 30
        elif total_sec < 600:
            interval = 60
        else:
            interval = 300

        pygame.draw.line(screen, COLOR_AXIS, (chart_rect.x, y), (chart_rect.right, y), 2)
        curr_t = 0
        while curr_t <= total_sec:
            x_ratio = curr_t / total_sec
            px = int(chart_rect.x + x_ratio * chart_rect.w)
            pygame.draw.line(screen, COLOR_AXIS, (px, y), (px, y + 10), 1)
            m = int(curr_t // 60)
            s = int(curr_t % 60)
            txt = f"{m:02d}:{s:02d}"
            t_surf = self.font_small.render(txt, True, COLOR_AXIS_TXT)
            t_rect = t_surf.get_rect(midtop=(px, y + 15))
            if t_rect.left < self.rect.left: t_rect.left = self.rect.left
            if t_rect.right > self.rect.right: t_rect.right = self.rect.right
            screen.blit(t_surf, t_rect)
            curr_t += interval

    def calculate_dominant_state(self, points_slice):
        if not points_slice: return None
        counts = {'Meditation': 0, 'Focus': 0, 'Anxiety': 0, 'Fatigue': 0}
        for p in points_slice:
            v, a = p['v'], p['a']
            if v >= 0 and a >= 0:
                counts['Meditation'] += 1
            elif v >= 0 and a < 0:
                counts['Focus'] += 1
            elif v < 0 and a >= 0:
                counts['Anxiety'] += 1
            else:
                counts['Fatigue'] += 1
        dom = max(counts, key=counts.get)
        if counts[dom] == 0: return None
        colors = {
            'Meditation': COLOR_POS_HIGH, 'Focus': COLOR_POS_LOW,
            'Anxiety': COLOR_NEG_HIGH, 'Fatigue': COLOR_NEG_LOW
        }
        return colors[dom]

    def draw_multi_scale_strips(self, screen, chart_rect):
        scales = [("MICRO (1s)", 1.0), ("MESO (10s)", 10.0), ("MACRO (60s)", 60.0)]
        current_y = self.strips_start_y + 50
        bar_start_x = chart_rect.x
        bar_width = chart_rect.w
        label_align_right_x = bar_start_x - 15

        for label, duration_sec in scales:
            lbl_surf = self.font.render(label, True, (200, 200, 200))
            lbl_rect = lbl_surf.get_rect(midright=(label_align_right_x, current_y + self.strip_height // 2))
            screen.blit(lbl_surf, lbl_rect)

            bar_rect = pygame.Rect(bar_start_x, current_y, bar_width, self.strip_height)
            pygame.draw.rect(screen, (20, 25, 30), bar_rect)

            p_idx = 0
            n_points = len(self.points)
            num_blocks = math.ceil(self.total_duration / duration_sec)

            for i in range(num_blocks):
                t_start = i * duration_sec
                t_end = t_start + duration_sec
                slice_points = []
                while p_idx < n_points:
                    p = self.points[p_idx]
                    if p['t_rel'] < t_start: p_idx += 1; continue
                    if p['t_rel'] >= t_end: break
                    slice_points.append(p)
                    p_idx += 1

                if slice_points:
                    color = self.calculate_dominant_state(slice_points)
                    if color:
                        bx = bar_rect.x + (t_start / self.total_duration) * bar_rect.w
                        bw = (duration_sec / self.total_duration) * bar_rect.w
                        draw_rect = pygame.Rect(int(bx), bar_rect.y, math.ceil(bw), self.strip_height)
                        draw_rect = draw_rect.clip(bar_rect)
                        if draw_rect.w > 0: pygame.draw.rect(screen, color, draw_rect)

            pygame.draw.rect(screen, (60, 70, 80), bar_rect, 1)
            current_y += self.strip_gap

    def draw_state_legend(self, screen, chart_rect):
        y = self.legend_y
        center_x = chart_rect.centerx
        legend_items = [
            (COLOR_POS_LOW, "专注"), (COLOR_POS_HIGH, "放松/冥想"),
            (COLOR_NEG_HIGH, "焦虑"), (COLOR_NEG_LOW, "疲劳")
        ]
        item_width = 160
        total_width = len(legend_items) * item_width
        start_x = center_x - total_width // 2
        for i, (color, text) in enumerate(legend_items):
            item_x = start_x + i * item_width
            box_size = 20
            pygame.draw.rect(screen, color, (item_x, y, box_size, box_size))
            pygame.draw.rect(screen, (200, 200, 200), (item_x, y, box_size, box_size), 1)
            txt_surf = self.font_label.render(text, True, (200, 200, 200))
            screen.blit(txt_surf, (item_x + box_size + 10, y + 2))

    def draw_summary(self, screen, chart_rect):
        if not self.stats: return
        y = self.summary_y
        center_x = chart_rect.centerx

        dur_min = int(self.stats['total_duration'] // 60)
        dur_sec = int(self.stats['total_duration'] % 60)
        time_str = f"总时长: {dur_min}分{dur_sec}秒"

        pct = self.stats['percents']
        trans_map = {'Meditation': '冥想', 'Focus': '专注', 'Anxiety': '焦虑', 'Fatigue': '疲劳'}
        dom_en = self.stats['dominant']
        dom_cn = trans_map.get(dom_en, dom_en)
        dom_pct = pct.get(dom_en, 0)

        dom_str = f"主导状态: {dom_cn} ({dom_pct:.1f}%)"
        dist_str = (f"分布:  专注 {pct['Focus']:.1f}%  |  "
                    f"冥想 {pct['Meditation']:.1f}%  |  "
                    f"焦虑 {pct['Anxiety']:.1f}%  |  "
                    f"疲劳 {pct['Fatigue']:.1f}%")

        summary_bg_rect = pygame.Rect(chart_rect.x, y, chart_rect.w, 90)
        pygame.draw.rect(screen, (25, 30, 35), summary_bg_rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_FRAME_LIGHT, summary_bg_rect, 1, border_radius=8)

        surf1 = self.font_summary.render(f"{time_str}        {dom_str}", True, COLOR_GOLD)
        rect1 = surf1.get_rect(centerx=center_x, top=y + 15)
        screen.blit(surf1, rect1)
        surf2 = self.font_summary.render(dist_str, True, (220, 230, 240))
        rect2 = surf2.get_rect(centerx=center_x, top=y + 50)
        screen.blit(surf2, rect2)

    def draw(self, screen):
        pygame.draw.rect(screen, COLOR_BG, self.rect)
        title_surf = self.font_large.render(self.title, True, COLOR_TEXT)
        screen.blit(title_surf, (self.rect.x + 20, self.rect.y + 10))

        chart_rect = self.draw_scatter_plot(screen)
        self.draw_y_axis(screen, chart_rect)
        self.draw_colorbar(screen, chart_rect)
        self.draw_multi_scale_strips(screen, chart_rect)
        self.draw_x_axis(screen, chart_rect)
        self.draw_state_legend(screen, chart_rect)
        self.draw_summary(screen, chart_rect)


class SceneHistory:
    def __init__(self, app, data):
        self.app = app
        self.file_path = data.get('file_path')
        self.points = []

        # [核心] 如果有 tkinter，初始化 root 并隐藏；否则跳过
        if HAS_TKINTER:
            try:
                self.root = tk.Tk()
                self.root.withdraw()
            except Exception as e:
                print(f"Tkinter initialization failed: {e}")

        self.load_data()

        viz_x, viz_y = 40, 60
        viz_w, viz_h = DESIGN_WIDTH - 80, DESIGN_HEIGHT - 120
        file_name = os.path.basename(self.file_path) if self.file_path else "Unknown"
        self.viz = StaticTrendVisualizer(viz_x, viz_y, viz_w, viz_h, f"SESSION REPLAY: {file_name}", self.points)

        self.btn_back = Button(40, 10, 120, 40, "< BACK", self.on_back)
        self.btn_print = Button(DESIGN_WIDTH - 180, 10, 140, 40, "EXPORT PDF", self.on_export)

    def load_data(self):
        print(f"Loading history: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                raw_rows = list(reader)
                if not raw_rows: return
                for row in raw_rows:
                    if len(row) >= 3:
                        try:
                            t = float(row[0]);
                            v = float(row[1]);
                            a = float(row[2])
                            self.points.append({'t': t, 'v': v, 'a': a})
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error loading CSV: {e}")

    def on_back(self):
        self.app.change_state("MENU")

    def on_export(self):
        """[核心修复] 处理没有 Tkinter 的情况"""
        if not self.file_path: return
        print("Exporting visualization...")

        # 默认文件名
        original_name = os.path.basename(self.file_path)
        default_name = os.path.splitext(original_name)[0] + "_report.pdf"

        target_path = ""

        # 1. 尝试弹出对话框
        if HAS_TKINTER:
            try:
                target_path = filedialog.asksaveasfilename(
                    title="导出分析报告",
                    initialfile=default_name,
                    defaultextension=".pdf",
                    filetypes=[("PDF Documents", "*.pdf"), ("PNG Images", "*.png")]
                )
            except Exception as e:
                print(f"Dialog failed: {e}")

        # 2. 如果没有 Tkinter 或用户取消，或者对话框失败
        #    但如果是因为没有库，我们应该自动保存到默认目录
        if not target_path:
            if not HAS_TKINTER:
                # 自动保存到 Documents/NeuroTasting_Data
                save_dir = get_user_data_dir()
                target_path = os.path.join(save_dir, default_name)
                print(f"Tkinter missing. Auto-saving to: {target_path}")
            else:
                print("Export cancelled.")
                return

        try:
            # 3. 截图 (使用 Canvas 保证高清)
            viz_surface = self.app.canvas.subsurface(self.viz.rect)

            if target_path.lower().endswith(".pdf"):
                temp_png = "temp_export.png"
                pygame.image.save(viz_surface, temp_png)
                if HAS_PIL:
                    image = Image.open(temp_png)
                    rgb = image.convert('RGB')
                    rgb.save(target_path)
                    os.remove(temp_png)
                    print(f"SUCCESS: PDF Saved to {target_path}")
                else:
                    print("ERROR: PIL not found, saving as PNG.")
                    pygame.image.save(viz_surface, target_path.replace(".pdf", ".png"))
            else:
                pygame.image.save(viz_surface, target_path)
                print(f"SUCCESS: Image Saved to {target_path}")

        except Exception as e:
            print(f"EXPORT FAILED: {e}")

    def update(self):
        pass

    def handle_events(self, events):
        for e in events:
            self.btn_back.handle_event(e)
            self.btn_print.handle_event(e)

    def draw(self, surface):
        surface.fill(COLOR_BG)
        self.viz.draw(surface)
        self.btn_back.draw(surface)
        self.btn_print.draw(surface)