import math

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont

from ev.ui import state as ev_state

# Circle geometry
CX, CY = 60, 55
R = 38

COLORS = {
    "idle":      QColor(100, 116, 139),  # slate
    "listening": QColor(59,  130, 246),  # blue
    "thinking":  QColor(245, 158,  11),  # amber
    "speaking":  QColor(16,  185, 129),  # emerald
}


class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self._tick = 0
        self._drag_pos = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(120, 130)

        # Default: bottom-right corner
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 140, screen.height() - 155)

        timer = QTimer(self)
        timer.timeout.connect(self._step)
        timer.start(40)  # 25 fps

    def _step(self):
        self._tick += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        state, _ = ev_state.get_state()
        t = self._tick
        color = COLORS.get(state, COLORS["idle"])

        # --- outer effects per state ---
        if state == "listening":
            for i in range(3):
                phase = (t * 1.6 + i * 22) % 65
                rr = R + phase * 0.9
                alpha = max(0, int(150 - phase * 2.3))
                c = QColor(color)
                c.setAlpha(alpha)
                painter.setPen(QPen(c, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(CX - rr), int(CY - rr), int(rr * 2), int(rr * 2))

        elif state == "thinking":
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(5):
                angle = math.radians((t * 6 + i * 72) % 360)
                dx = (R + 15) * math.cos(angle)
                dy = (R + 15) * math.sin(angle)
                c = QColor(color)
                c.setAlpha(220 - i * 35)
                painter.setBrush(c)
                painter.drawEllipse(int(CX + dx - 5), int(CY + dy - 5), 10, 10)

        elif state == "speaking":
            painter.setPen(Qt.PenStyle.NoPen)
            bars = 7
            bar_w, gap = 5, 3
            total_w = bars * bar_w + (bars - 1) * gap
            sx = CX - total_w // 2
            for i in range(bars):
                phase = math.radians((t * 9 + i * 45) % 360)
                h = int(5 + 20 * abs(math.sin(phase)))
                c = QColor(color)
                c.setAlpha(200)
                painter.setBrush(c)
                bx = sx + i * (bar_w + gap)
                painter.drawRoundedRect(bx, CY + R + 6, bar_w, h, 2, 2)

        elif state == "idle":
            pulse = math.sin(t * 0.05) * 3
            rr = R + 8 + pulse
            c = QColor(color)
            c.setAlpha(45)
            painter.setPen(QPen(c, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(CX - rr), int(CY - rr), int(rr * 2), int(rr * 2))

        # --- main circle ---
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(CX - R, CY - R, R * 2, R * 2)

        # "EV" text
        font = QFont("Arial", 15, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 230))
        painter.drawText(CX - 20, CY - 11, 40, 22, Qt.AlignmentFlag.AlignCenter, "EV")

        # state label
        font2 = QFont("Arial", 7)
        painter.setFont(font2)
        painter.setPen(QColor(180, 190, 210, 170))
        painter.drawText(0, CY + R + 2, 120, 14, Qt.AlignmentFlag.AlignCenter, state.upper())

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
