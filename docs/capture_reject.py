from pathlib import Path

from playwright.sync_api import sync_playwright

FIG = Path(__file__).resolve().parent / "figures"


def main():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto("http://127.0.0.1:8501", wait_until="networkidle", timeout=60000)
        page.get_by_text("AEGIS contract review").wait_for()
        page.locator("input[type='file']").set_input_files(str(FIG / "image-only-sample.pdf"))
        page.get_by_text("no extractable text").wait_for(timeout=20000)
        page.screenshot(path=str(FIG / "screenshot-image-only-rejected.png"), full_page=True)
        browser.close()


if __name__ == "__main__":
    main()
