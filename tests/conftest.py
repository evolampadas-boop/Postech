"""Garante que a raiz do projeto está no PYTHONPATH ao rodar os testes."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
