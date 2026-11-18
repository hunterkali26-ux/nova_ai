"""NOVA AI desktop application entry point.

Launches the PySide6 main window and wires together the model runtime,
orchestrator, knowledge engine, and database stores.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("NOVA AI")
    app.setOrganizationName("NOVA AI")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
