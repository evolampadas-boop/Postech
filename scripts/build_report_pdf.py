"""Gera o PDF do relatório a partir do Markdown (docs/relatorio_tech_challenge.md).

Converte o Markdown em HTML e usa o Chrome/Edge em modo headless pra imprimir em PDF
(assim os links ficam clicáveis). Uso: `python scripts/build_report_pdf.py`.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
MD_PATH = DOCS_DIR / "relatorio_tech_challenge.md"
HTML_PATH = DOCS_DIR / "relatorio_tech_challenge.html"
PDF_PATH = DOCS_DIR / "relatorio_tech_challenge.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

CSS = """
@page { size: A4; margin: 2.2cm 2.2cm; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 11pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 .2em; }
h2 { font-size: 14pt; margin: 1.4em 0 .4em; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 12pt; margin: 1em 0 .3em; }
p { margin: .5em 0; text-align: justify; }
ul { margin: .4em 0 .8em 1.2em; }
img { max-width: 100%; height: auto; display: block; margin: .8em auto; }
code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: 90%; }
pre { background: #f6f6f6; padding: .7em; border-radius: 4px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: .8em 0; }
th, td { border: 1px solid #bbb; padding: 4px 8px; }
a { color: #0b5cad; }
.subtitle { color: #555; font-size: 13pt; margin: 0 0 .2em; }
.meta { color: #777; font-size: 10pt; margin: 0 0 1.2em; }
"""


def parse_front_matter(text):
    """Separa o bloco YAML do topo (entre ---) do corpo do Markdown."""
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            block = text[3:end].strip()
            for line in block.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    meta[key.strip()] = val.strip().strip('"')
            text = text[end + 4:].lstrip("\n")
    return meta, text


def main():
    raw = MD_PATH.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)

    html_body = markdown.markdown(
        body, extensions=["extra", "smarty", "tables", "fenced_code", "codehilite"]
    )

    header = ""
    if meta.get("title"):
        header += f"<h1>{meta['title']}</h1>"
    if meta.get("subtitle"):
        header += f"<p class='subtitle'>{meta['subtitle']}</p>"
    line = " — ".join(v for v in (meta.get("author"), meta.get("date")) if v)
    if line:
        header += f"<p class='meta'>{line}</p>"

    full_html = (
        "<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{header}{html_body}</body></html>"
    )
    HTML_PATH.write_text(full_html, encoding="utf-8")

    chrome = next((p for p in CHROME_CANDIDATES if Path(p).exists()), None) or shutil.which("chrome")
    if not chrome:
        print("Chrome/Edge não encontrado. HTML gerado em", HTML_PATH)
        print("Abra o HTML no navegador e use Imprimir > Salvar como PDF.")
        return 1

    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH.as_uri(),
        ],
        check=True,
    )
    print("PDF gerado em", PDF_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
