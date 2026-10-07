"""Dev launcher: run with no arguments (IDLE, double-click) to open the GUI.

``python run_gpustacker.py stack ...`` still exposes the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    if len(sys.argv) <= 1 or sys.argv[1] == "gui":
        from gpustacker.gui import main as gui_main

        raise SystemExit(gui_main())
    from gpustacker.cli import main

    raise SystemExit(main())
