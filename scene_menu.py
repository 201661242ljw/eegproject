import pygame
import os
import glob
from config import *
from ui_components import Panel, Button, TextInput


class SceneMenu:
    def __init__(self, app):
        self.app = app
        self.scan_angle = 0
        self.history_buttons = []
        self.setup_ui()
        self.refresh_history_list()

    def setup_ui(self):
        cx, cy = DESIGN_WIDTH // 2, DESIGN_HEIGHT // 2

        self.panel_login = Panel(cx - 400, cy - 300, 800, 600, "SYSTEM ACCESS")
        self.input_uid = TextInput(cx - 150, cy - 100, 300, 50, "User01")
        self.btn_start = Button(cx - 100, cy + 150, 200, 60, "INITIALIZE", self.on_start_click, enabled=False)

    def refresh_history_list(self):
        """扫描 [文档/NeuroTasting_Data] 目录并生成按钮"""
        self.history_buttons = []

        # [核心修改] 使用智能路径
        record_dir = get_user_data_dir()

        if not os.path.exists(record_dir):
            return

        files = sorted(glob.glob(os.path.join(record_dir, "*.csv")), key=os.path.getmtime, reverse=True)[:6]

        start_x = 50
        start_y = 250

        for i, f_path in enumerate(files):
            f_name = os.path.basename(f_path)
            btn = Button(start_x, start_y + i * 50, 350, 40, f"> {f_name}",
                         callback=lambda f=f_path: self.on_history_click(f))
            self.history_buttons.append(btn)

    def on_start_click(self):
        uid = self.input_uid.text if self.input_uid.text else "Guest"
        print(f"Starting session for: {uid}")
        self.app.change_state("SESSION", {'user_id': uid})

    def on_history_click(self, file_path):
        self.app.change_state("HISTORY", {'file_path': file_path})

    def draw_status_light(self, surface, x, y, label, status_code):
        colors = [COLOR_RED, COLOR_YELLOW, COLOR_GREEN]
        c = colors[status_code]
        pygame.draw.circle(surface, c, (x, y), 10)
        s = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(s, (*c, 50), (20, 20), 18)
        surface.blit(s, (x - 20, y - 20))
        font = get_font("SimHei", 20)
        lbl = font.render(label, True, COLOR_TEXT)
        surface.blit(lbl, (x + 25, y - 10))

    def update(self):
        is_connected = self.app.bt_receiver.connected
        self.btn_start.enabled = is_connected
        self.scan_angle = (self.scan_angle + 2) % 360

    def handle_events(self, events):
        for event in events:
            self.input_uid.handle_event(event)
            self.btn_start.handle_event(event)
            for btn in self.history_buttons:
                btn.handle_event(event)

    def draw(self, surface):
        surface.fill(COLOR_BG)
        self.panel_login.draw(surface)

        font_lbl = get_font("SimHei", 24)
        surface.blit(font_lbl.render("SUBJECT ID / 用户标识:", True, COLOR_TEXT_DIM),
                     (self.input_uid.rect.x, self.input_uid.rect.y - 40))

        self.input_uid.draw(surface)
        self.btn_start.draw(surface)

        cx = DESIGN_WIDTH // 2
        cy = DESIGN_HEIGHT // 2
        bt_status = 0
        status_text = "Scanning..."
        if self.app.bt_receiver.connected:
            bt_status = 2;
            status_text = "DEVICE READY: RFstar_7FEA"
        elif self.app.bt_receiver.retry_count > 0:
            bt_status = 1;
            status_text = f"SEARCHING (Try {self.app.bt_receiver.retry_count})..."

        self.draw_status_light(surface, cx - 150, cy + 50, status_text, bt_status)

        if bt_status != 2:
            radar_center = (cx + 250, cy + 50)
            pygame.draw.circle(surface, COLOR_FRAME, radar_center, 40, 2)
            import math
            ex = radar_center[0] + 38 * math.cos(math.radians(self.scan_angle))
            ey = radar_center[1] + 38 * math.sin(math.radians(self.scan_angle))
            pygame.draw.line(surface, COLOR_CYAN, radar_center, (ex, ey), 2)

        hist_x = 50
        title_font = get_font("SimHei", 24, True)
        surface.blit(title_font.render("DATA RECORDS (Click to View)", True, COLOR_FRAME_LIGHT), (hist_x, 200))

        for btn in self.history_buttons:
            btn.draw(surface)