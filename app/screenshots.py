import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = Path(__file__).resolve().parent
OUT = APP.parent / 'out/r3'
PORT = 8611
PAGES = [('', 'рис_app_overview.png', None), ('Перевірка_операції', 'рис_app_check.png', 'Перевірити'),
         ('Пакетна_перевірка', 'рис_app_batch.png', None), ('Моделі_і_поріг', 'рис_app_models.png', None),
         ('Серії_й_схеми', 'рис_app_series.png', None)]


def main():
    server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', str(APP / 'Огляд.py'), '--server.port', str(PORT),
                               '--server.headless', 'true', '--browser.gatherUsageStats', 'false'], cwd=APP)
    try:
        time.sleep(8)
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='chrome', headless=True)
            page = browser.new_page(viewport={'width': 1400, 'height': 1000}, device_scale_factor=1.5)
            errors = []
            for path, name, button in PAGES:
                page.goto(f'http://localhost:{PORT}/{path}', timeout=120000)
                page.wait_for_selector('[data-testid="stApp"]', timeout=120000)
                page.wait_for_timeout(9000)
                if button:
                    page.get_by_role('button', name=button).click()
                    page.wait_for_timeout(15000)
                alerts = page.locator('[data-testid="stException"]')
                if alerts.count():
                    errors.append((path, alerts.first.inner_text()[:400]))
                page.screenshot(path=str(OUT / name), full_page=True)
                print(name)
            browser.close()
        for e in errors:
            print('ERROR', e)
    finally:
        server.terminate()


if __name__ == '__main__':
    main()
