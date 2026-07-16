import math
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPixmap

from ev.ui import state as ev_state

ICONS_DIR = Path(__file__).parent / "icons"

# Image display geometry
IMG_SIZE = 110
IMG_X = 10
IMG_Y = 5
CX = IMG_X + IMG_SIZE // 2   # 65
CY = IMG_Y + IMG_SIZE // 2   # 60
R  = IMG_SIZE // 2            # 55  — radius used for animation positioning

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

        self._pixmap = QPixmap(str(ICONS_DIR / "EV.png"))

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(IMG_SIZE + 20, IMG_SIZE + 40)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - (IMG_SIZE + 40), screen.height() - (IMG_SIZE + 60))

        timer = QTimer(self)
        timer.timeout.connect(self._step)
        timer.start(40)  # 25 fps

    def _step(self):
        self._tick += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

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
                dx = (R + 14) * math.cos(angle)
                dy = (R + 14) * math.sin(angle)
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
                h = int(5 + 16 * abs(math.sin(phase)))
                c = QColor(color)
                c.setAlpha(200)
                painter.setBrush(c)
                bx = sx + i * (bar_w + gap)
                painter.drawRoundedRect(bx, IMG_Y + IMG_SIZE + 4, bar_w, h, 2, 2)

        elif state == "idle":
            pulse = math.sin(t * 0.05) * 3
            rr = R + 6 + pulse
            c = QColor(color)
            c.setAlpha(40)
            painter.setPen(QPen(c, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(CX - rr), int(CY - rr), int(rr * 2), int(rr * 2))

        # --- robot image ---
        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                IMG_SIZE, IMG_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(IMG_X, IMG_Y, scaled)

        # --- state label ---
        font = QFont("Arial", 7)
        painter.setFont(font)
        painter.setPen(QColor(180, 190, 210, 170))
        painter.drawText(
            0, IMG_Y + IMG_SIZE + 2, IMG_SIZE + 20, 14,
            Qt.AlignmentFlag.AlignCenter,
            state.upper(),
        )

        painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
