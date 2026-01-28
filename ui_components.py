# ui_components.py
import pygame
import time
from config import *


class Panel:
    """通用的半透明背景面板"""

    def __init__(self, x, y, w, h, title=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.title = title
        self.font = get_font("SimHei", 24, bold=True)

    def draw(self, surface):
        # 1. 背景 (带 Alpha)
        s = pygame.Surface((self.rect.w, self.rect.h), pygame.SRCALPHA)
        s.fill(COLOR_PANEL_BG)

        # 2. 边框装饰
        border_col = COLOR_FRAME
        pygame.draw.rect(s, border_col, (0, 0, self.rect.w, self.rect.h), 2)

        # 角落高亮
        corn_len = 20
        hi_col = COLOR_FRAME_LIGHT
        pygame.draw.line(s, hi_col, (0, 0), (corn_len, 0), 3)
        pygame.draw.line(s, hi_col, (0, 0), (0, corn_len), 3)
        pygame.draw.line(s, hi_col, (self.rect.w - corn_len, self.rect.h), (self.rect.w, self.rect.h), 3)
        pygame.draw.line(s, hi_col, (self.rect.w, self.rect.h - corn_len), (self.rect.w, self.rect.h), 3)

        surface.blit(s, (self.rect.x, self.rect.y))

        # 3. 标题
        if self.title:
            t_surf = self.font.render(self.title, True, COLOR_TEXT)
            t_rect = t_surf.get_rect(topleft=(self.rect.x + 15, self.rect.y + 10))
            pygame.draw.line(surface, COLOR_CYAN, (t_rect.left, t_rect.bottom + 2),
                             (t_rect.right + 20, t_rect.bottom + 2), 2)
            surface.blit(t_surf, t_rect)


class Button:
    def __init__(self, x, y, w, h, text, callback, enabled=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.enabled = enabled
        self.is_hovered = False
        self.font = get_font("SimHei", 24, bold=True)

    def handle_event(self, event):
        if not self.enabled: return

        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                # 再次检查碰撞，确保准确
                if self.rect.collidepoint(event.pos):
                    if self.callback: self.callback()

    def draw(self, surface):
        if not self.enabled:
            col_bg = (30, 30, 30)
            col_border = (60, 60, 60)
            col_text = (100, 100, 100)
        elif self.is_hovered:
            col_bg = (40, 60, 80)
            col_border = COLOR_CYAN
            col_text = COLOR_CYAN
        else:
            col_bg = (20, 30, 40)
            col_border = COLOR_FRAME_LIGHT
            col_text = COLOR_TEXT

        pygame.draw.rect(surface, col_bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, col_border, self.rect, 2, border_radius=6)

        t_surf = self.font.render(self.text, True, col_text)
        t_rect = t_surf.get_rect(center=self.rect.center)
        surface.blit(t_surf, t_rect)


class TextInput:
    def __init__(self, x, y, w, h, default_text="Guest"):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = default_text
        self.active = False
        self.font = get_font("Consolas", 28)
        self.color_inactive = COLOR_FRAME
        self.color_active = COLOR_CYAN

        # 光标相关
        self.cursor_visible = True
        self.last_cursor_toggle = time.time()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                # [关键修改] 只要是可打印字符都允许输入
                if len(event.unicode) > 0 and event.unicode.isprintable():
                    self.text += event.unicode

    def draw(self, surface):
        color = self.color_active if self.active else self.color_inactive
        # 背景
        pygame.draw.rect(surface, (10, 10, 10), self.rect)
        # 边框
        pygame.draw.rect(surface, color, self.rect, 2)

        t_surf = self.font.render(self.text, True, COLOR_TEXT)
        # 垂直居中
        text_y = self.rect.y + (self.rect.h - t_surf.get_height()) // 2
        surface.blit(t_surf, (self.rect.x + 10, text_y))

        # [新增] 绘制光标
        if self.active:
            if time.time() - self.last_cursor_toggle > 0.5:
                self.cursor_visible = not self.cursor_visible
                self.last_cursor_toggle = time.time()

            if self.cursor_visible:
                # 光标位置在文字末尾
                cursor_x = self.rect.x + 10 + t_surf.get_width() + 2
                cursor_h = t_surf.get_height()
                pygame.draw.line(surface, COLOR_CYAN, (cursor_x, text_y), (cursor_x, text_y + cursor_h), 2)