# main_app.py
import pygame
import sys
from config import *
from bluetooth_core import EEGBluetoothReceiver
from scene_menu import SceneMenu
from scene_session import SceneSession
from scene_history import SceneHistory  # [新增] 引入历史场景


class GameApp:
    def __init__(self):
        pygame.init()
        self.info = pygame.display.Info()
        self.screen_w = int(self.info.current_w * 0.9)
        self.screen_h = int(self.info.current_h * 0.9)

        # 计算缩放比例
        scale_w = self.screen_w / DESIGN_WIDTH
        scale_h = self.screen_h / DESIGN_HEIGHT
        self.scale_factor = min(scale_w, scale_h)

        self.window_w = int(DESIGN_WIDTH * self.scale_factor)
        self.window_h = int(DESIGN_HEIGHT * self.scale_factor)

        self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
        pygame.display.set_caption("Neuro-Tasting System v2.1")

        self.canvas = pygame.Surface((DESIGN_WIDTH, DESIGN_HEIGHT))
        self.clock = pygame.time.Clock()

        print("INIT: Starting Bluetooth Receiver...")
        self.bt_receiver = EEGBluetoothReceiver()

        self.states = {}
        self.current_state_name = None
        self.current_scene = None

        self.change_state("MENU")

    def change_state(self, state_name, data=None):
        print(f"STATE: Switching to {state_name}")
        if self.current_scene and hasattr(self.current_scene, 'exit_scene'):
            self.current_scene.exit_scene()

        self.current_state_name = state_name

        if state_name == "MENU":
            self.current_scene = SceneMenu(self)
        elif state_name == "SESSION":
            self.current_scene = SceneSession(self, data if data else {})
        elif state_name == "HISTORY":
            # [新增] 切换到历史回放场景
            self.current_scene = SceneHistory(self, data if data else {})

    def transform_mouse_pos(self, pos):
        # 1. 计算缩放后的实际渲染区域大小
        scaled_w = int(DESIGN_WIDTH * self.scale_factor)
        scaled_h = int(DESIGN_HEIGHT * self.scale_factor)

        # 2. 计算黑边偏移量 (画面居中)
        offset_x = (self.window_w - scaled_w) // 2
        offset_y = (self.window_h - scaled_h) // 2

        # 3. 减去偏移量
        adj_x = pos[0] - offset_x
        adj_y = pos[1] - offset_y

        # 4. 除以缩放比例，还原到设计分辨率
        design_x = int(adj_x / self.scale_factor)
        design_y = int(adj_y / self.scale_factor)

        return (design_x, design_y)

    def run(self):
        running = True
        while running:
            # 1. 获取原始事件
            raw_events = pygame.event.get()
            processed_events = []

            for event in raw_events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.current_state_name != "MENU":
                            self.change_state("MENU")
                        else:
                            running = False
                    # 确保键盘事件被传入
                    processed_events.append(event)

                elif event.type == pygame.VIDEORESIZE:
                    self.window_w, self.window_h = event.w, event.h
                    self.screen = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
                    self.scale_factor = min(self.window_w / DESIGN_WIDTH, self.window_h / DESIGN_HEIGHT)

                # 拦截鼠标事件并修改坐标
                elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                    design_pos = self.transform_mouse_pos(event.pos)

                    new_dict = event.__dict__.copy()
                    new_dict['pos'] = design_pos
                    if 'rel' in new_dict:
                        new_dict['rel'] = (int(event.rel[0] / self.scale_factor), int(event.rel[1] / self.scale_factor))

                    new_event = pygame.event.Event(event.type, new_dict)
                    processed_events.append(new_event)
                else:
                    processed_events.append(event)

            # 2. Update & Draw
            if self.current_scene:
                self.current_scene.update()
                if hasattr(self.current_scene, 'handle_events'):
                    self.current_scene.handle_events(processed_events)
                self.current_scene.draw(self.canvas)

            # 3. 缩放并绘制到屏幕
            scaled_surf = pygame.transform.smoothscale(
                self.canvas,
                (int(DESIGN_WIDTH * self.scale_factor), int(DESIGN_HEIGHT * self.scale_factor))
            )

            # 计算居中位置
            dest_x = (self.window_w - scaled_surf.get_width()) // 2
            dest_y = (self.window_h - scaled_surf.get_height()) // 2

            self.screen.fill(COLOR_BG)
            self.screen.blit(scaled_surf, (dest_x, dest_y))

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    app = GameApp()
    app.run()