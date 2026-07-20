import math
import os
from pathlib import Path

try:
    import PyQt6 as _pyqt6
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
        Path(_pyqt6.__file__).parent / "Qt6" / "plugins"
    )
except Exception:
    pass
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

from PyQt6.QtWidgets import QApplication, QWidget, QMenu, QTextEdit, QLabel
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QPoint, QEasingCurve
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPixmap

from ev.ui import state as ev_state

ICONS_DIR = Path(__file__).parent / "icons"

IMG_SIZE = 110
IMG_X    = 10
IMG_Y    = 5
CX = IMG_X + IMG_SIZE // 2
CY = IMG_Y + IMG_SIZE // 2
R  = IMG_SIZE // 2

ROBOT_W = IMG_SIZE + 20   # 130
ROBOT_H = IMG_SIZE + 40   # 150
CODE_W   = 420
CODE_H   = 260
EXPAND_STEP = 16           # px per frame when growing/shrinking

COLORS = {
    "idle":      QColor(100, 116, 139),
    "listening": QColor(59,  130, 246),
    "thinking":  QColor(245, 158,  11),
    "speaking":  QColor(16,  185, 129),
}


class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()
        self._tick         = 0
        self._drag_pos     = None
        self._home         = None
        self._last_code    = ""
        self._panel_h      = 0      # current panel height (animates toward _panel_target)
        self._panel_target = 0
        self._pos_anim     = None   # keeps QPropertyAnimation alive
        self._animating    = False  # True while floating up/down

        self._pixmap = QPixmap(str(ICONS_DIR / "EV.png"))

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(ROBOT_W, ROBOT_H)

        # --- code panel (hidden until code arrives) ---
        self._code_edit = QTextEdit(self)
        self._code_edit.setReadOnly(True)
        self._code_edit.setFont(QFont("Monospace", 9))
        self._code_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: none;
                border-top: 1px solid #313244;
                padding: 8px;
            }
        """)
        self._code_edit.setGeometry(0, ROBOT_H, ROBOT_W, 0)

        # --- "Copied" flash label ---
        self._copied_lbl = QLabel("✓ Copied to clipboard", self)
        self._copied_lbl.setStyleSheet(
            "color: #a6e3a1; font-size: 8px; background: transparent; padding: 2px;"
        )
        self._copied_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._copied_lbl.setGeometry(0, ROBOT_H - 14, ROBOT_W, 14)
        self._copied_lbl.hide()

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - ROBOT_W - 20, screen.height() - ROBOT_H - 40)
        self._home = self.pos()

        timer = QTimer(self)
        timer.timeout.connect(self._step)
        timer.start(40)   # 25 fps

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #

    def _step(self):
        self._tick += 1

        code = ev_state.get_code()

        # New code arrived
        if code and code != self._last_code:
            self._last_code = code
            self._on_new_code(code)

        # Code was cleared
        elif not code and self._last_code:
            self._last_code = ""
            self._panel_target = 0   # start collapsing

        # Animate panel height
        if self._panel_h != self._panel_target:
            step = EXPAND_STEP if self._panel_h < self._panel_target else -EXPAND_STEP
            self._panel_h = max(0, min(self._panel_target, self._panel_h + step))
            self._apply_panel_size()

            # Once fully collapsed, float back home
            if self._panel_h == 0 and self._panel_target == 0 and not self._animating:
                self._float_to(self._home)

        # Idle float (only when fully settled)
        if (self._home and not self._animating
                and self._panel_h == 0 and self._panel_target == 0
                and self._drag_pos is None):
            ox = int(3 * math.sin(self._tick * 0.03))
            oy = int(2 * math.sin(self._tick * 0.05))
            self.move(self._home.x() + ox, self._home.y() + oy)

        self.update()

    # ------------------------------------------------------------------ #
    #  Code panel logic                                                    #
    # ------------------------------------------------------------------ #

    def _on_new_code(self, code: str):
        self._code_edit.setPlainText(code)
        QApplication.clipboard().setText(code)

        # Flash "copied" label
        self._copied_lbl.show()
        QTimer.singleShot(2500, self._copied_lbl.hide)

        # Float to top, then start expanding
        screen = QApplication.primaryScreen().availableGeometry()
        target = QPoint(self.x(), screen.top() + 10)
        self._float_to(target, on_done=lambda: setattr(self, "_panel_target", CODE_H))

    def _apply_panel_size(self):
        w = CODE_W if self._panel_h > 0 else ROBOT_W
        self.resize(w, ROBOT_H + self._panel_h)
        self._code_edit.setGeometry(0, ROBOT_H, w, self._panel_h)

    # ------------------------------------------------------------------ #
    #  Position animation                                                  #
    # ------------------------------------------------------------------ #

    def _float_to(self, target: QPoint, on_done=None):
        self._animating = True
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(550)
        anim.setStartValue(self.pos())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _finished():
            self._animating = False
            if on_done:
                on_done()

        anim.finished.connect(_finished)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        self._pos_anim = anim   # prevent GC

    # ------------------------------------------------------------------ #
    #  Paint                                                               #
    # ------------------------------------------------------------------ #

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        state, _ = ev_state.get_state()
        t     = self._tick
        color = COLORS.get(state, COLORS["idle"])

        if state == "listening":
            for i in range(3):
                phase = (t * 1.6 + i * 22) % 65
                rr    = R + phase * 0.9
                alpha = max(0, int(150 - phase * 2.3))
                c = QColor(color); c.setAlpha(alpha)
                painter.setPen(QPen(c, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(CX - rr), int(CY - rr), int(rr * 2), int(rr * 2))

        elif state == "thinking":
            painter.setPen(Qt.PenStyle.NoPen)
            for i in range(5):
                angle = math.radians((t * 6 + i * 72) % 360)
                dx = (R + 14) * math.cos(angle)
                dy = (R + 14) * math.sin(angle)
                c = QColor(color); c.setAlpha(220 - i * 35)
                painter.setBrush(c)
                painter.drawEllipse(int(CX + dx - 5), int(CY + dy - 5), 10, 10)

        elif state == "speaking":
            painter.setPen(Qt.PenStyle.NoPen)
            bars = 7; bar_w, gap = 5, 3
            sx = CX - (bars * bar_w + (bars - 1) * gap) // 2
            for i in range(bars):
                phase = math.radians((t * 9 + i * 45) % 360)
                h = int(5 + 16 * abs(math.sin(phase)))
                c = QColor(color); c.setAlpha(200)
                painter.setBrush(c)
                painter.drawRoundedRect(sx + i * (bar_w + gap), IMG_Y + IMG_SIZE + 4, bar_w, h, 2, 2)

        elif state == "idle":
            pulse = math.sin(t * 0.05) * 3
            rr = R + 6 + pulse
            c = QColor(color); c.setAlpha(40)
            painter.setPen(QPen(c, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(CX - rr), int(CY - rr), int(rr * 2), int(rr * 2))

        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                IMG_SIZE, IMG_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(IMG_X, IMG_Y, scaled)

        # State / mode label
        label = "CODING" if self._panel_h > 0 or self._panel_target > 0 else state.upper()
        font  = QFont("Arial", 7)
        painter.setFont(font)
        painter.setPen(QColor(180, 190, 210, 170))
        painter.drawText(
            0, IMG_Y + IMG_SIZE + 2, ROBOT_W, 14,
            Qt.AlignmentFlag.AlignCenter, label,
        )

        painter.end()

    # ------------------------------------------------------------------ #
    #  Mouse                                                               #
    # ------------------------------------------------------------------ #

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._pos_anim:
                self._pos_anim.stop()
            self._animating = False
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseReleaseEvent(self, event):
        self._home = self.pos()
        self._drag_pos = None
        event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.addAction("Close EV", QApplication.instance().quit)
        menu.exec(event.globalPos())
