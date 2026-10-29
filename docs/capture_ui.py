"""Capture screenshots of real app states (no model call is involved in any of them).

Start the app first, with no key and no .env, for example:

    AEGIS_NO_DOTENV=1 streamlit run app.py --server.headless true --server.port 8599

then run (Playwright with Chromium installed):

    python docs/capture_ui.py http://localhost:8599 openai
    # restart the app with AEGIS_PROVIDER=gemini, then:
    python docs/capture_ui.py http://localhost:8599 gemini

A completed-review screenshot is only taken after a real, authorised review;
it is never produced from fixture data.
"""

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
SAMPLE = FIGURES / "synthetic-contract-v2.pdf"
IMAGE_ONLY = FIGURES / "image-only-sample.pdf"

OPENAI_STATES = [
    ("ui-start-no-key-no-index.png", None, None, 1000),
    ("ui-image-only-rejected.png", IMAGE_ONLY, None, 1000),
    ("ui-synthetic-v2-checked-review-blocked.png", SAMPLE, None, 1500),
    ("ui-eu-chat-index-not-built.png", None, "EU AI Act questions", 1100),
]
GEMINI_STATES = [("ui-gemini-mode-banner-and-gates.png", SAMPLE, None, 1650)]


def capture(url, name, pdf, expand, height):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": height})
        page.goto(url)
        page.wait_for_selector("h1", timeout=30000)
        page.wait_for_timeout(2500)
        if pdf:
            page.set_input_files("input[type=file]", str(pdf))
            page.wait_for_timeout(4000)
        if expand:
            page.get_by_text(expand).first.click()
            page.wait_for_timeout(1500)
        page.screenshot(path=str(FIGURES / name))
        browser.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
    states = GEMINI_STATES if len(sys.argv) > 2 and sys.argv[2] == "gemini" else OPENAI_STATES
    for state in states:
        capture(url, *state)
        print("saved", state[0])
