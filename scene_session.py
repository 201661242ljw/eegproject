import pygame
import numpy as np
import math
import time
import queue
import csv
import datetime
import threading
import sys

from config import *
from ui_components import Button, Panel
from eeg_engine import EEGAnalyzer
from part import ScatterTrendVisualizer
from scene_history import SceneHistory  # [新增] 引入历史场景

# --- 辅助计算函数 (原样搬运自 main_v16.py) ---
def catmull_rom_interpolate(points, num_points=20, closed=True):
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
            t2 = t * t;
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


class AsyncProcessor(threading.Thread):
    def __init__(self, receiver_queue, output_queue):
        super().__init__()
        self.receiver_queue = receiver_queue
        self.output_queue = output_queue
        self.engine = EEGAnalyzer(fs=1000)
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                data_tuple = self.receiver_queue.get(timeout=0.1)
                if isinstance(data_tuple, tuple):
                    raw, reliability = data_tuple
                else:
                    raw = data_tuple; reliability = None

                result = self.engine.process(raw, reliability_mask=reliability)

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except:
                        pass
                self.output_queue.put(result)
            except queue.Empty:
                continue
            except Exception:
                pass

    def stop(self):
        self.running = False

class SceneSession:
    def __init__(self, app, session_data):
        self.app = app
        self.user_id = session_data.get('user_id', 'Guest')

        # --- 数据流 ---
        self.ui_queue = queue.Queue(maxsize=10)
        self.processor = AsyncProcessor(self.app.bt_receiver.get_gui_queue(), self.ui_queue)
        self.processor.start()

        self.init_recording()
        self.init_viz_state()
        self.load_assets()
        self.setup_ui()

    def init_recording(self):
        # [核心修改] 使用智能路径，确保有权限写入
        self.record_dir = get_user_data_dir()

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
        self.filename = os.path.join(self.record_dir, f"{timestamp}_{self.user_id}.csv")
        self.csv_file = open(self.filename, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp_rel", "valence", "arousal", "attention", "relaxation"])
        self.start_time = time.time()
        print(f"Recording to: {self.filename}")
    def init_viz_state(self):
        self.wave_len = 1000
        self.wave_fp1 = np.zeros(self.wave_len)
        self.wave_fp2 = np.zeros(self.wave_len)
        self.n_bars = 60
        self.bars_fp1 = np.zeros(self.n_bars)
        self.bars_fp2 = np.zeros(self.n_bars)

        # PSD 数据
        self.psd_data = {
            'freqs': np.linspace(1, 60, 30).tolist(),
            'ch1': np.zeros(30).tolist(),
            'ch2': np.zeros(30).tolist()
        }
        self.psd_log_mode = False

        # 指标容器
        self.metrics_packet = {
            'metrics': {
                'sensory_attention': 0, 'emotional_valence': 0, 'emotional_arousal': 0,
                'neural_relaxation': 0, 'cortical_arousal': 0, 'cognitive_depth': 0,
                'immersion_ratio': 0, 'slope': -1.0
            },
            'powers': {'left': {}, 'right': {}}
        }
        self.smooth_metrics = self.metrics_packet['metrics'].copy()
        self.smooth_powers = {'left': {}, 'right': {}}

        # 趋势图
        viz_w, viz_h = 600, 200
        self.trend_viz = ScatterTrendVisualizer(
            DESIGN_WIDTH - viz_w - 20, DESIGN_HEIGHT - viz_h - 20,
            viz_w, viz_h, "Affective State Trend", 300, fps=FPS
        )

    def load_assets(self):
        # [核心修改] 使用 config.py 中的 resource_path 包装
        def load_img(name, size=None, alpha=None):
            path = resource_path(os.path.join('imgs', name + ".png"))
            img = None
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                except:
                    pass
            if img is None:
                surf = pygame.Surface(size if size else (64, 64))
                surf.fill((50, 50, 50))
                img = surf
            if size: img = pygame.transform.smoothscale(img, size)
            if alpha is not None: img.set_alpha(alpha)
            return img

        self.img_bg = load_img("bg", (DESIGN_WIDTH, DESIGN_HEIGHT), alpha=255)
        self.img_bg_light = load_img("bg_light", (DESIGN_WIDTH, DESIGN_HEIGHT))
        self.img_center_raw = load_img("center", (800, 650))
        self.img_brain_light = load_img("brain_light", (800, 650))
        self.img_bar_raw = load_img("bar", (180, 330))
        self.img_line = load_img("line")

        gradient_stops = [
            (0.00, (80, 80, 200)), (0.05, (90, 110, 210)), (0.10, (100, 140, 230)),
            (0.15, (100, 180, 240)), (0.20, (100, 210, 250)), (0.25, (100, 230, 240)),
            (0.30, (100, 240, 230)), (0.35, (100, 240, 200)), (0.40, (120, 240, 180)),
            (0.45, (150, 240, 150)), (0.50, (180, 240, 120)), (0.55, (210, 240, 100)),
            (0.60, (240, 240, 80)), (0.65, (250, 220, 70)), (0.70, (255, 200, 60)),
            (0.75, (255, 180, 50)), (0.80, (255, 160, 40)), (0.85, (255, 140, 50)),
            (0.90, (255, 120, 60)), (0.95, (255, 100, 70)), (1.00, (255, 80, 80))
        ]
        self.bar_slices_heatmap = self.generate_heatmap_slices(self.img_bar_raw, gradient_stops, 50)
        self.bar_layout = self.precompute_spiral()

    def generate_heatmap_slices(self, base_img, stops, steps):
        # 简化的切片生成器
        w, h = base_img.get_size()
        h_top, h_bot = int(h * 0.3), int(h * 0.5)
        h_mid = h - h_top - h_bot

        slices = []
        for i in range(steps):
            t = i / (steps - 1)
            # 简单颜色混合
            c = stops[int(t * (len(stops) - 1))][1]  # 简化取色

            colored = base_img.copy()
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill(c)
            colored.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

            sl = {
                'top': colored.subsurface((0, 0, w, h_top)),
                'mid': colored.subsurface((0, h_top, w, h_mid)),
                'bottom': colored.subsurface((0, h_top + h_mid, w, h_bot)),
                'orig_size': (w, h), 'h_ratios': (h_top, h_mid, h_bot)
            }
            slices.append(sl)
        return slices

    def precompute_spiral_(self):
        layout = []
        cx, cy = 1376, 840
        rot = math.pi * 0.65
        r_start, r_end = 600, 750
        for i in range(self.n_bars):
            p = i / (self.n_bars - 1)
            # FP1
            r = r_start + (r_end - r_start) * p
            th = np.linspace(0, -2 * np.pi, self.n_bars, endpoint=False)[i] + rot
            x = cx + r * math.cos(th);
            y = cy + r * math.sin(th) * 0.55
            scale = 0.98 + (y - cy) / 2000
            layout.append({'type': 'fp1', 'x': x, 'y': y, 'scale': scale, 'idx': i, 'y_sort': y})
            # FP2
            r2 = 840 + (1050 - 840) * p
            x2 = cx + r2 * math.cos(th);
            y2 = cy + r2 * math.sin(th) * 0.55
            scale2 = 0.98 + (y2 - cy) / 2000
            layout.append({'type': 'fp2', 'x': x2, 'y': y2, 'scale': scale2, 'idx': i, 'y_sort': y2})
        layout.sort(key=lambda k: k['y_sort'])
        return layout

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


    def setup_ui(self):
        self.btn_stop = Button(DESIGN_WIDTH - 150, 20, 130, 50, "STOP/SAVE", self.on_stop_click)

        self.font_large = pygame.font.SysFont("SimHei", 48, bold=True)
        self.font_small = get_font("SimSun", 24)
        self.font_tiny = get_font("SimSun", 18)
        self.font_num = get_font("SimHei", 32, bold=True)
        self.font_lbl = get_font("SimHei", 48, bold=True)
        self.font_num_large = pygame.font.SysFont("SimHei", 32, bold=True)

        self.line_config = {
            'enable': True,
            'pos': (DESIGN_WIDTH // 2 - 270, DESIGN_HEIGHT // 2 + 100),
            'alpha': 180,
            'scale': 1.0
        }

        self.current_affect = (0.5, 0.5) # Default center
    def on_stop_click(self):
        self.exit_scene()
        self.app.change_state("MENU")

    def exit_scene(self):
        self.processor.stop()
        if self.csv_file: self.csv_file.close()

    def update(self):
        has_new = False
        while not self.ui_queue.empty():
            try:
                pkt = self.ui_queue.get_nowait()
                self.metrics_packet = pkt  # 存储完整包以获取 powers

                if 'psd_info' in pkt and pkt['psd_info'].get('freqs'):
                    self.psd_data = pkt['psd_info']

                c1 = pkt['signals']['fp1']['chunk_clean']
                c2 = pkt['signals']['fp2']['chunk_clean']
                if len(c1) > 0:
                    n = min(len(c1), self.wave_len)
                    self.wave_fp1 = np.roll(self.wave_fp1, -n);
                    self.wave_fp1[-n:] = c1[-n:]
                    self.wave_fp2 = np.roll(self.wave_fp2, -n);
                    self.wave_fp2[-n:] = c2[-n:]

                    # Bar update
                    step = 4
                    if len(self.wave_fp1) >= self.n_bars * step:
                        self.bars_fp1 = np.clip((self.wave_fp1[-self.n_bars * step::step] + 60) / 120, 0, 1)[::-1]
                        self.bars_fp2 = np.clip((self.wave_fp2[-self.n_bars * step::step] + 60) / 120, 0, 1)[::-1]
                    has_new = True
            except:
                break

        # Smooth metrics
        alpha = 0.1
        for k, v in self.metrics_packet['metrics'].items():
            self.smooth_metrics[k] = self.smooth_metrics.get(k, 0) * (1 - alpha) + v * alpha

        if has_new:
            val = self.smooth_metrics['emotional_valence'] / 100 * 2 - 1
            aro = self.smooth_metrics['emotional_arousal'] / 100 * 2 - 1
            self.trend_viz.add_point(val, aro)
            try:
                self.csv_writer.writerow([time.time() - self.start_time, val, aro])
            except:
                pass


    def draw_01a_tint_image(self, image, color):
        image = image.copy()
        colored_surf = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        colored_surf.fill(color)
        image.blit(colored_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return image



    def draw_01b_custom_line_image(self,surface):
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
        surface.blit(scaled_img, scaled_img.get_rect(
            center=(self.line_config['pos'][0] + shake_x, self.line_config['pos'][1] + shake_y)))

    def draw_01_spiral(self, surface):
        val_arousal = self.smooth_metrics.get('emotional_arousal', 0) / 100.0
        val_valence = self.smooth_metrics.get('emotional_valence', 50) / 100.0
        t = time.time()
        alpha_light = int(255 * math.pow((math.sin(t * 1.2 + 1) + 1) / 2, 10.0) * (0.2 + val_valence * 0.8))
        self.img_bg.set_alpha(255 - alpha_light)
        surface.blit(self.img_bg, self.img_bg.get_rect(center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)))
        self.img_bg_light.set_alpha(alpha_light)
        surface.blit(self.img_bg_light, self.img_bg_light.get_rect(center=(DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2)))

        def get_brighter_brain_color(error_rate, pos_score):
            t = min(1.0, max(0.0, error_rate))
            r = int(50 + (255 - 50) * t)
            g = int(255 - (255 - 50) * t)
            b = int(255 - (255 - 50) * t)
            brightness_boost = 1.5 + pos_score * 1.0
            return (
                min(255, int(r * brightness_boost)), min(255, int(g * brightness_boost)),
                min(255, int(b * brightness_boost)))

        for item in self.bar_layout:
            if item['type'] == 'center':
                cx, cy = item['x'], item['y'] - 300
                hover = math.sin(t * 0.8) * 8
                pulse = 1.0 + math.pow((math.sin(t * 2) + 1) / 2, 15.0) * 0.03
                brain = self.draw_01a_tint_image(self.img_center_raw,  get_brighter_brain_color(val_arousal, val_valence))
                w, h = int(brain.get_width() * pulse), int(brain.get_height() * pulse)
                surface.blit(pygame.transform.scale(brain, (w, h)), brain.get_rect(center=(cx, cy + hover)))
                l_alpha = int(255 * math.pow((math.sin(t * 2) + 1) / 2, 15.0))
                if l_alpha > 5:
                    light = pygame.transform.scale(self.img_brain_light, (w, h))
                    light.set_alpha(l_alpha)
                    surface.blit(light, light.get_rect(center=(cx, cy + hover)),
                                     special_flags=pygame.BLEND_RGBA_ADD)
                if self.line_config['enable']: self.draw_01b_custom_line_image(surface)
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
            surface.blit(s_bot, (x, y_bot));
            surface.blit(s_mid, (x, y_bot - th_mid))
            surface.blit(s_top, (x, y_bot - th_mid - int(h_top * scale * 0.5)))

    def draw_02a_solid_arc(self, surface, color, center, r_inner, r_outer, start_rad, end_rad):
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

    def draw_02_gauge_arc_styled(self, surface, center, radius, start_angle_deg, end_angle_deg, val, color, label,x_text):
        thickness = 50
        size = radius * 2 + 200
        temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
        temp_center = (size // 2, size // 2)

        start_rad = math.radians(start_angle_deg)
        end_rad = math.radians(end_angle_deg)

        r_main_center = radius
        r_main_inner = r_main_center - thickness / 2
        r_main_outer = r_main_center + thickness / 2

        bg_color = (20, 40, 50, 120)
        self.draw_02a_solid_arc(temp_surf, bg_color, temp_center, r_main_inner, r_main_outer, start_rad, end_rad)

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
            self.draw_02a_solid_arc(temp_surf, fill_color, temp_center,
                                r_main_inner + 4, r_main_outer - 4,
                                math.radians(fill_angle_start), math.radians(fill_angle_end))

            highlight_color = (255, 255, 255, 180)
            self.draw_02a_solid_arc(temp_surf, highlight_color, temp_center,
                                r_main_inner + 4, r_main_inner + 8,
                                math.radians(fill_angle_start), math.radians(fill_angle_end))

        surface.blit(temp_surf, (center[0] - size // 2, center[1] - size // 2))
        # text_surf_label = self.font_large.render(label, True, color)
        # text_surf_label.set_alpha(200)
        # surface.blit(text_surf_label, (x_text, 850))
        # 垂直排列字符
        char_height = self.font_large.get_height()
        for i, char in enumerate(label):
            char_surface = self.font_large.render(char, True, color)
            char_surface.set_alpha(200)
            # 每个字符垂直偏移
            surface.blit(char_surface, (x_text, 550 + i * char_height))



        text_surf_val = self.font_num_large.render(f"{int(val * 100)}", True, color)
        text_surf_val.set_alpha(200)
        surface.blit(text_surf_val, (x_text, 850))

    def draw_03_star_radar(self,surface, center, radius_base, powers_left, powers_right):
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

        def catmull_rom_interpolate( points, num_points=20, closed=True):
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
        pts_L = generate_star_control_points(powers_left)
        pts_R = generate_star_control_points(powers_right)


        shape_L = catmull_rom_interpolate(pts_L, num_points=10, closed=True) if pts_L else []
        shape_R = catmull_rom_interpolate(pts_R, num_points=10, closed=True) if pts_R else []

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

        surface.blit(surf, (center[0] - s // 2, center[1] - s // 2))


    def draw_04_combined_psd(self,surface, x, y, width, height, freqs, psd1, psd2, color1=COLOR_FP1, color2=COLOR_FP2, title="Spectrum Comparison"):
        fixed_psd_max = 1e11
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, (15, 20, 25), rect)
        pygame.draw.rect(surface, COLOR_FRAME, rect, 2, border_radius=4)

        # 绘制标题，指示当前模式
        mode_str = "(Log X)" if self.psd_log_mode else "(Linear X)"
        full_title = f"{title} {mode_str}"
        surface.blit(self.font_small.render(full_title, True, COLOR_FRAME_LIGHT), (x + 10, y + 5))

        surface.blit(self.font_tiny.render("FP1", True, color1), (x + width - 80, y + 5))
        surface.blit(self.font_tiny.render("FP2", True, color2), (x + width - 40, y + 5))

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
                pygame.draw.line(surface, COLOR_GRID, (px_clamped, y), (px_clamped, y + height), 1)
                surface.blit(self.font_tiny.render(f"{freq}Hz", True, COLOR_FRAME_LIGHT),
                                 (px_clamped - 5, y + height - 15))

        # Y 轴 (Log Power) - 保持不变
        log_max = math.log10(fixed_psd_max)
        for val in [10, 100, 1000]:
            py = y + height - (math.log10(val) / log_max * (height - 25)) - 10
            pygame.draw.line(surface, (30, 40, 50), (x, py), (x + width, py), 1)
            surface.blit(self.font_tiny.render(f"{val}", True, (60, 70, 80)), (x + 2, py - 10))

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
                y_ratio = math.log10(max(1.0, min(p, fixed_psd_max))) / log_max

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
                surface.blit(surf, (x, y))

        draw_single_psd(psd1, color1)
        draw_single_psd(psd2, color2)

        surface.blit(self.font_tiny.render(f"Log Scale Max: {int(fixed_psd_max)}", True, (100, 200, 100)),
                         (x + 10, y + 25))


    def draw_05_raw_wave(self, surface):
        x=50; y=50; width=450; height=250;  color1=COLOR_FP1; color2=COLOR_FP2; title="Raw EEG Comparison"
        fixed_scale_uv = 100.0
        rect = pygame.Rect(x, y, width, height)

        # rect = pygame.Rect(50, 50, 450, 250)
        pygame.draw.rect(surface, (15, 20, 25), rect)
        pygame.draw.rect(surface, COLOR_FRAME, rect, 2, border_radius=4)
        center_y = y + height / 2
        pygame.draw.line(surface, (60, 70, 80), (x, center_y), (x + width, center_y), 1)
        scale_y = (height / 2) / fixed_scale_uv
        for v in [50, 100]:
            py_pos = center_y - v * scale_y
            pygame.draw.line(surface, (30, 40, 50), (x, py_pos), (x + width, py_pos), 1)
            py_neg = center_y + v * scale_y
            pygame.draw.line(surface, (30, 40, 50), (x, py_neg), (x + width, py_neg), 1)
            surface.blit(self.font_tiny.render(f"+{v}", True, (50, 60, 70)), (x + 2, py_pos - 10))
            surface.blit(self.font_tiny.render(f"-{v}", True, (50, 60, 70)), (x + 2, py_neg - 10))
        for i in range(1, 4):
            grid_x = x + i * (width / 4)
            pygame.draw.line(surface, (30, 40, 50), (grid_x, y), (grid_x, y + height), 1)
            surface.blit(self.font_tiny.render(f"-{1.0 - i * 0.25:.2f}s", True, (80, 100, 120)),
                             (grid_x + 2, y + height - 15))
        surface.blit(self.font_small.render(title, True, COLOR_FRAME_LIGHT), (x + 10, y + 5))
        surface.blit(self.font_tiny.render("FP1 (Yel)", True, color1), (x + width - 140, y + 5))
        surface.blit(self.font_tiny.render("FP2 (Blu)", True, color2), (x + width - 70, y + 5))

        def draw_single_ch(data_arr, col):
            points = []
            step_x = width / len(data_arr)
            for i in range(len(data_arr)):
                val = data_arr[i]
                if np.isnan(val): val = 0
                offset_y = val * scale_y
                offset_y = max(-height / 2 + 2, min(height / 2 - 2, offset_y))
                points.append((x + i * step_x, center_y - offset_y))
            if len(points) > 1: pygame.draw.lines(surface, col, False, points, 2)

        draw_single_ch(self.wave_fp1, COLOR_FP1);
        draw_single_ch(self.wave_fp2, COLOR_FP2)
        surface.blit(self.font_tiny.render(f"Fixed Scale: ±{int(fixed_scale_uv)}uV", True, (100, 200, 100)),
                         (x + 10, y + 25))


    def draw_06_metrics_panel(self, surface):
        rect = pygame.Rect(20, DESIGN_HEIGHT - 320, 400, 300)
        Panel(rect.x, rect.y, rect.w, rect.h, "Metrics").draw(surface)
        labels = [("Attention", self.smooth_metrics['sensory_attention']),
                  ("Relaxation", self.smooth_metrics['neural_relaxation']),
                  ("Immersion", self.smooth_metrics['immersion_ratio'] * 50)]
        for i, (txt, val) in enumerate(labels):
            y = rect.y + 60 + i * 60
            surface.blit(self.font_small.render(txt, True, (200, 200, 200)), (rect.x + 20, y))
            pygame.draw.rect(surface, (50, 50, 50), (rect.x + 20, y + 30, 200, 10))
            pygame.draw.rect(surface, COLOR_GOLD, (rect.x + 20, y + 30, int(np.clip(val, 0, 100) * 2), 10))
            surface.blit(self.font_num.render(f"{val:.1f}", True, COLOR_GOLD), (rect.x + 240, y + 15))
    def draw_07_affect_grid(self,surface, center_x, center_y, size):
        half = size // 2
        # 背景
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.rect(surf, (20, 25, 30, 150), (0, 0, size, size), border_radius=8)
        pygame.draw.rect(surf, COLOR_FRAME, (0, 0, size, size), 2, border_radius=8)

        # 网格线
        pygame.draw.line(surf, (60, 70, 80), (0, half), (size, half), 1)  # X axis
        pygame.draw.line(surf, (60, 70, 80), (half, 0), (half, size), 1)  # Y axis

        # 标签
        surface.blit(self.font_small.render("2D Affect Space", True, COLOR_FRAME_LIGHT),
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

        surface.blit(surf, (center_x - half, center_y - half))
    # --- 绘图总入口 ---
    def draw(self, surface):

        # 2. 仪表盘 (还原)
        m = self.smooth_metrics

        a_org = m.get('cortical_arousal', 0)
        a_new = a_org / 100 * 2 - 1
        low = 0.3
        high = 0.8
        a_new = (a_new - low ) / (high - low)
        a_val = 1 / (1 + np.exp(-10 * (np.clip(a_new, -500, 500)-0.5)))  # 0.0 ~ 1.0

        b_org = m.get('emotional_valence', 0)
        b_new = b_org / 100 * 2 - 1
        low = 0.0
        high = 0.75
        b_new = (b_new - low ) / (high - low)
        v_val = 1 / (1 +  np.exp(-10 * (np.clip(b_new, -500, 500)-0.5)))  # 0.0 ~ 1.0

        self.current_affect = (a_val, v_val)

        self.draw_01_spiral(surface)
        self.draw_02_gauge_arc_styled(surface, (750, 650), 600, 150, 210, a_val, COLOR_CYAN, "实时兴奋度", 50)
        self.draw_02_gauge_arc_styled(surface, (DESIGN_WIDTH - 750, 650), 600, 330, 390, v_val, COLOR_GOLD, "实时专注度",
                                   2650)

        # 3. 雷达图 (还原)
        self.draw_03_star_radar(surface, (550, 950), 60, self.metrics_packet['powers']['left'],
                             self.metrics_packet['powers']['right'])

        # 4. PSD (还原)
        self.draw_04_combined_psd(surface, DESIGN_WIDTH - 500, 50, 450, 250, self.psd_data['freqs'], self.psd_data['ch1'],
                               self.psd_data['ch2'])

        # 5. Raw Wave (左上)
        self.draw_05_raw_wave(surface)

        # 6. Metrics (左下)
        self.draw_06_metrics_panel(surface)


        self.draw_07_affect_grid(surface, DESIGN_WIDTH - 1250, 1150, 300)

        # 7. Trend & 2D (右下)
        self.trend_viz.draw(surface)  # 原 part.py 组件

        # Button
        self.btn_stop.draw(surface)

    def handle_events(self, events):
        for e in events:
            self.btn_stop.handle_event(e)
            if e.type == pygame.KEYDOWN and e.key == pygame.K_l:
                self.psd_log_mode = not self.psd_log_mode