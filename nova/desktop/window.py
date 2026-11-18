"""NOVA AI main window: dark-themed desktop shell with six working tabs."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..backend.api import state
from ..orchestrator import Orchestrator


DARK_STYLESHEET = """
QMainWindow { background: #1e1e2e; }
QTabWidget::pane { border: 1px solid #3a3a4c; background: #1e1e2e; }
QTabBar::tab {
    background: #2a2a3c; color: #c0c0d0; padding: 8px 16px; margin: 2px;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}
QTabBar::tab:selected { background: #1e1e2e; color: #ffffff; border-bottom: 2px solid #7c3aed; }
QTabBar::tab:hover { background: #35354b; }
QPushButton {
    background: #3a3a4c; color: #ffffff; border: 1px solid #555566;
    border-radius: 6px; padding: 6px 12px; min-width: 90px;
}
QPushButton:hover { background: #4a4a5c; }
QPushButton:disabled { background: #2a2a3c; color: #777788; border-color: #444455; }
QPushButton:primary { background: #7c3aed; }
QPushButton:primary:hover { background: #8b5cf6; }
QTextEdit { background: #252535; color: #cbd5e1; border: 1px solid #3a3a4c; border-radius: 6px; }
QLineEdit { background: #252535; color: #cbd5e1; border: 1px solid #3a3a4c; border-radius: 4px; padding: 4px; }
QComboBox { background: #252535; color: #cbd5e1; border: 1px solid #3a3a4c; border-radius: 4px; padding: 4px; }
QListWidget { background: #252535; color: #cbd5e1; border: 1px solid #3a3a4c; border-radius: 6px; }
QTableWidget { background: #252535; color: #cbd5e1; border: 1px solid #3a3a4c; border-radius: 6px; }
QHeaderView::section { background: #2a2a3c; color: #c0c0d0; border: none; padding: 4px; }
QSplitter::handle { background: #3a3a4c; }
QStatusBar { background: #2a2a3c; color: #a0a0b8; border-top: 1px solid #3a3a4c; }
QGroupBox { border: 1px solid #3a3a4c; border-radius: 6px; margin-top: 12px; color: #c0c0d0; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QScrollBar:vertical { background: #1e1e2e; width: 12px; border-radius: 6px; }
QScrollBar::handle:vertical { background: #555566; border-radius: 6px; min-height: 24px; }
"""


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NOVA AI — Local-First AI")
        self.resize(1100, 700)
        self.setStyleSheet(DARK_STYLESHEET)

        # Backend state (shared with the API if it runs).
        self.model_manager = state.model_manager
        self.conversation_store = state.conversation_store
        self.message_store = state.message_store
        self.document_store = state.document_store
        self.orchestrator = Orchestrator(
            self.model_manager,
            self.conversation_store,
            self.message_store,
            rag_pipeline=state.rag,
        )

        self._current_conversation_id = None

        self._build_ui()
        self._refresh_models()
        self._refresh_conversations()
        self._refresh_documents()
        self._start_system_monitor()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.chat_widget = self._build_chat_tab()
        self.model_manager_widget = self._build_model_manager_tab()
        self.knowledge_widget = self._build_knowledge_tab()
        self.training_widget = self._build_training_tab()
        self.monitor_widget = self._build_monitor_tab()
        self.settings_widget = self._build_settings_tab()

        self.tabs.addTab(self.chat_widget, "Chat")
        self.tabs.addTab(self.model_manager_widget, "Model Manager")
        self.tabs.addTab(self.knowledge_widget, "Knowledge Library")
        self.tabs.addTab(self.training_widget, "Training Studio")
        self.tabs.addTab(self.monitor_widget, "System Monitor")
        self.tabs.addTab(self.settings_widget, "Settings")

        layout.addWidget(self.tabs)

        self.status = self.statusBar()
        self.status.showMessage("NOVA AI ready — offline")

    def _title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-size: 15px; font-weight: 600; color: #e0e0f0; padding: 4px 0;")
        return lbl

    def _hrule(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background: #3a3a4c; height: 1px; border: none;")
        return line
    # ------------------------------------------------------------------ #
    # Refresh helpers
    # ------------------------------------------------------------------ #
    def _refresh_models(self) -> None:
        if hasattr(self, "model_list"):
            self.model_list.clear()
            for entry in self.model_manager.list_models():
                self.model_list.addItem(entry["name"])

    def _refresh_conversations(self) -> None:
        if hasattr(self, "conv_list"):
            self.conv_list.clear()
            for row in self.conversation_store.list():
                self.conv_list.addItem(row["id"], row["title"])

    def _refresh_documents(self) -> None:
        if hasattr(self, "doc_list"):
            self.doc_list.clear()
            for row in self.document_store.list():
                self.doc_list.addItem(
                    row["id"],
                    f"{row['title']} ({row['source_type']}) — {int(row['chunk_count'] or 0)} chunks",
                )

    # ------------------------------------------------------------------ #
    # Chat tab
    # ------------------------------------------------------------------ #
    def _build_chat_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        self.new_chat_btn = QPushButton("New Chat")
        self.new_chat_btn.clicked.connect(self._new_chat)
        self.conv_list = QListWidget()
        self.conv_list.setMaximumWidth(260)
        self.conv_list.itemClicked.connect(self._load_conversation)
        top.addWidget(self.conv_list)
        top.addWidget(self.new_chat_btn)
        top.addStretch()
        layout.addLayout(top)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display, stretch=1)

        input_row = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(70)
        self.chat_input.setPlaceholderText("Type a message... (Shift+Enter to newline)")
        self.send_btn = QPushButton("Send")
        self.send_btn.setProperty("class", "primary")
        self.send_btn.clicked.connect(self._send_chat)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_chat)
        input_row.addWidget(self.chat_input, stretch=1)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.clear_btn)
        layout.addLayout(input_row)

        opt = QHBoxLayout()
        opt.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self._populate_model_combo()
        opt.addWidget(self.model_combo)
        self.rag_checkbox = QPushButton("RAG: ON" if state.rag else "RAG: unavailable")
        self.rag_checkbox.setCheckable(True)
        self.rag_checkbox.setChecked(bool(state.rag))
        self.rag_checkbox.setEnabled(bool(state.rag))
        opt.addWidget(self.rag_checkbox)
        opt.addStretch()
        layout.addLayout(opt)

        self.chat_input.installEventFilter(self)
        return widget

    def _populate_model_combo(self) -> None:
        self.model_combo.clear()
        for entry in self.model_manager.list_models():
            self.model_combo.addItem(entry["name"], entry["name"])

    def _new_chat(self) -> None:
        cid = os.urandom(8).hex()
        title = f"Conversation {len(self.conversation_store.list()) + 1}"
        model = self.model_combo.currentText() or "nova-tiny"
        self.conversation_store.create(cid, title, model, rag_enabled=self.rag_checkbox.isChecked())
        self._current_conversation_id = cid
        self.chat_display.clear()
        self._refresh_conversations()
        self.status.showMessage(f"New conversation {cid[:8]}")

    def _load_conversation(self, item) -> None:
        cid = item.data(Qt.UserRole) or item.text()
        self._current_conversation_id = cid
        self.chat_display.clear()
        self.status.showMessage(f"Loaded conversation {cid[:8]}")
        self._render_messages(cid)

    def _render_messages(self, conversation_id: str) -> None:
        messages = self.message_store.list_for_conversation(conversation_id)
        html_parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            color = "#7c3aed" if role == "assistant" else "#3b82f6"
            html_parts.append(
                f'<div style="margin: 8px 0;"><b style="color:{color}">{role.upper()}</b>: '
                f"{content.replace(chr(10), '<br>')}</div>"
            )
        self.chat_display.setHtml("\n".join(html_parts))
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)

    def _send_chat(self) -> None:
        text = self.chat_input.toPlainText().strip()
        if not self._current_conversation_id:
            self._new_chat()
        if not text:
            return
        model = self.model_combo.currentText() or "nova-tiny"
        rag_mode = "enabled" if self.rag_checkbox.isChecked() else "disabled"
        self.chat_input.clear()
        self.status.showMessage("Generating...")
        QApplication.processEvents()
        try:
            response = self.orchestrator.chat(
                user_input=text,
                model_name=model,
                conversation_id=self._current_conversation_id,
                rag_mode=rag_mode,
            )
            self._render_messages(self._current_conversation_id)
            self.status.showMessage(
                f"Done (model={response.model}, rag={response.rag_used})"
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Chat error", str(exc))
            self.status.showMessage("Chat failed")

    def _append_message(self, role: str, content: str, sources=None) -> None:
        import uuid

        if not self._current_conversation_id:
            self._new_chat()
        msg_id = uuid.uuid4().hex
        self.message_store.add(
            msg_id, self._current_conversation_id, role, content, sources=sources or []
        )
        self.conversation_store.update_touch(self._current_conversation_id)

    def _clear_chat(self) -> None:
        self.chat_display.clear()
        self.chat_input.clear()

    def eventFilter(self, obj, event):  # noqa: ARG002
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        if obj is self.chat_input and event.type() == QEvent.KeyPress:
            if isinstance(event, QKeyEvent) and event.key() in (
                Qt.Key_Return,
                Qt.Key_Enter,
            ):
                if not event.modifiers() & Qt.ShiftModifier:
                    self._send_chat()
                    return True
        return super().eventFilter(event)

    # ------------------------------------------------------------------ #
    # Model Manager tab
    # ------------------------------------------------------------------ #
    def _build_model_manager_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(self._title("Model Manager"))
        layout.addWidget(self._hrule())

        row = QHBoxLayout()
        self.model_list = QListWidget()
        self.model_list.setMaximumWidth(220)
        self.model_list.itemClicked.connect(self._model_selected)
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setMaximumHeight(200)
        row.addWidget(self.model_list)
        row.addWidget(self.model_info, stretch=1)
        layout.addLayout(row)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Model")
        self.btn_load.clicked.connect(self._load_selected_model)
        self.btn_unload = QPushButton("Unload")
        self.btn_unload.clicked.connect(self._unload_selected_model)
        self.btn_remove = QPushButton("Remove (confirm)")
        self.btn_remove.clicked.connect(self._remove_selected_model)
        self.btn_register = QPushButton("Register NOVA Tiny")
        self.btn_register.clicked.connect(self._register_nova_tiny)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_unload)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_register)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addStretch()
        return widget

    def _model_selected(self, item) -> None:
        name = item.text()
        entry = self.model_manager.get_model(name)
        if not entry:
            return
        info = (
            f"Name: {entry['name']}\n"
            f"Type: {entry['model_type']}\n"
            f"Architecture: {entry['architecture']}\n"
            f"Format: {entry.get('model_format') or '-'}\n"
            f"Context: {entry.get('context_length') or '-'}\n"
            f"Path: {entry.get('file_path') or '-'}\n"
            f"Description: {entry.get('description') or '-'}"
        )
        self.model_info.setPlainText(info)
        self.status.showMessage(f"Selected model {name}")

    def _load_selected_model(self) -> None:
        item = self.model_list.currentItem()
        if not item:
            QMessageBox.information(self, "Model Manager", "Select a model first.")
            return
        try:
            self.model_manager.load_model(item.text())
            self.status.showMessage(f"Loaded {item.text()}")
            QMessageBox.information(self, "Model Manager", f"Loaded {item.text()}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load failed", str(exc))

    def _unload_selected_model(self) -> None:
        self.model_manager.unload_model()
        self.status.showMessage("Model unloaded")
        QMessageBox.information(self, "Model Manager", "Model unloaded.")

    def _remove_selected_model(self) -> None:
        item = self.model_list.currentItem()
        if not item:
            QMessageBox.information(self, "Model Manager", "Select a model first.")
            return
        name = item.text()
        reply = QMessageBox.question(
            self, "Remove model", f"Unregister model '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.model_manager.unregister(name)
            self._refresh_models()
            self.status.showMessage(f"Removed {name}")

    def _register_nova_tiny(self) -> None:
        from ..core.config import get_preset

        config = get_preset("tiny")
        from ..runtime.model_manager import ModelInfo

        info = ModelInfo(
            name="nova-tiny",
            model_type="nova-tiny",
            architecture="decoder-only Transformer",
            context_length=config.context_length,
            model_format="pytorch",
            description="NOVA Tiny custom Transformer (train from scratch)",
        )
        try:
            self.model_manager.register(info)
            self._refresh_models()
            self.status.showMessage("Registered nova-tiny")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Register failed", str(exc))

