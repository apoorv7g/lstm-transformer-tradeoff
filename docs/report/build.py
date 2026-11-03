"""Render report.html to an A4 PDF with Playwright (headless Chrome, no window).

The page header, page numbers and A4 margins come from the CSS @page rules in report.html.
Usage, from the repo root, with a Python that has playwright installed:
    py -3.11 docs/report/build.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "Compute_vs_Accuracy_LSTM_Transformer_Report.pdf"

with sync_playwright() as pw:
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception:
        browser = pw.chromium.launch(headless=True, channel="chrome")  # fall back to installed Chrome
    page = browser.new_page()
    page.goto((HERE / "report.html").as_uri(), wait_until="networkidle")
    page.pdf(path=str(OUT), format="A4", print_background=True, prefer_css_page_size=True)
    browser.close()
print(f"wrote {OUT}")
