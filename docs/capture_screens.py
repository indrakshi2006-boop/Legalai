"""Capture the running Streamlit app. Does not click through a live model review."""

from pathlib import Path

from playwright.sync_api import sync_playwright

FIG = Path(__file__).resolve().parent / "figures"
URL = "http://127.0.0.1:8501"


def shot(page, name):
    page.screenshot(path=str(FIG / name), full_page=True)
    print("saved", name)


def main():
    with sync_playwright() as playwright:
        browser = None
        last = None
        for channel in ("msedge", "chrome"):
            try:
                browser = playwright.chromium.launch(channel=channel, headless=True)
                print("browser", channel)
                break
            except Exception as exc:
                last = exc
        if browser is None:
            raise SystemExit(f"No installed browser could be launched: {last}")
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.get_by_text("AEGIS contract review").wait_for(timeout=30000)
        shot(page, "screenshot-upload-page.png")

        page.locator("input[type='file']").set_input_files(str(FIG / "synthetic-sample.pdf"))
        page.get_by_text("PDF read:").wait_for(timeout=20000)
        shot(page, "screenshot-synthetic-accepted.png")

        page.get_by_role("button", name="Review contract").click()
        page.get_by_text("playbook index is missing").wait_for(timeout=20000)
        shot(page, "screenshot-missing-playbook.png")

        page.locator("input[type='file']").set_input_files(str(FIG / "image-only-sample.pdf"))
        page.get_by_text("OCR is not available").wait_for(timeout=20000)
        shot(page, "screenshot-image-only-rejected.png")
        browser.close()


if __name__ == "__main__":
    main()
