from typing import List, Optional, Tuple, Dict
from cog import BasePredictor, Input, Path

import os
import time
import requests
from urllib.parse import quote_plus

from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# ---------- Free services ----------
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ---------- Elevation ----------
def get_elevation_open_elevation(lat: float, lon: float) -> Optional[float]:
    r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=20)
    if not r.ok:
        return None
    js = r.json()
    res = js.get("results") or []
    return float(res[0].get("elevation")) if res else None

# ---------- URL Building ----------
def build_earth_url_with_search(address: str, lat: float, lon: float,
                                a: float, d: float, y: float, h: float, t: float, r: float = 0.0) -> str:
    addr = quote_plus(address) if address else ""
    return (f"https://earth.google.com/web/search/{addr}/"
            f"@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,{y:.2f}y,{h:.3f}h,{t:.3f}t,{r:.1f}r")

# ---------- Image utils ----------
def center_crop(image_path: str, output_path: str, crop_margin: float) -> None:
    crop_margin = max(0.0, min(0.49, float(crop_margin)))
    with Image.open(image_path) as im:
        w, h = im.size
        left, right = int(w * crop_margin), int(w * (1.0 - crop_margin))
        top, bottom = int(h * crop_margin), int(h * (1.0 - crop_margin))
        im.crop((left, top, right, bottom)).save(output_path)

# ---------- Predictor ----------
class Predictor(BasePredictor):
    def setup(self) -> None:
        options = webdriver.ChromeOptions()
        options.binary_location = '/root/chrome-linux/chrome'
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # --- ADD THIS LINE ---
        options.add_argument('--disable-gpu') # Often resolves hangs in headless environments
        # ---------------------
        self.browser = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.browser, 30)

    def _wait_scene_ready(self, index: int, debug: bool) -> None:
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
            if debug: print(f"[Step {index:02d}] Canvas detected.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] No canvas before timeout.", flush=True)

        try:
            self.wait.until(lambda d: d.execute_script(
                "return document.body && document.body.innerText.indexOf('Google Earth') === -1;"
            ))
            if debug: print(f"[Step {index:02d}] Splash gone.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] Splash still visible; continuing.", flush=True)

    def _open_and_capture_new_tab(self, url: str, w: int, h: int, wait_seconds: int, index: int, debug: bool) -> str:
        self.browser.switch_to.new_window('tab')
        self.browser.set_window_size(w, h)
        if debug: print(f"[Step {index:02d}] Opening NEW TAB: {url}", flush=True)
        self.browser.get(url)

        try:
            time.sleep(0.5)
            self.browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        except Exception:
            pass

        self._wait_scene_ready(index, debug)

        # Reverted to a simple, fixed wait time.
        if wait_seconds > 0:
            if debug: print(f"[Step {index:02d}] Starting fixed wait: {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)
            if debug: print(f"[Step {index:02d}] Fixed wait finished.", flush=True)

        try:
            self.browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
        except Exception:
            pass

        full_path = f"view_{index:02d}_full.png"
        ok = self.browser.save_screenshot(full_path)
        if debug:
            print(f"[Step {index:02d}] Screenshot saved: {full_path} (ok={ok})", flush=True)

        self.browser.close()
        self.browser.switch_to.window(self.browser.window_handles[0])
        return full_path

    def predict(
        self,
        latitude: float = Input(description="Latitude of the target location"),
        longitude: float = Input(description="Longitude of the target location"),
        address: str = Input(description="Optional address or label for the location", default=""),
        w: int = Input(description="Viewport width", default=1920),
        h: int = Input(description="Viewport height", default=1080),
        wait_seconds: int = Input(description="Fixed time (seconds) to wait before taking screenshot", default=15),
        crop_margin: float = Input(description="Center-crop margin per side (0–0.49)", default=0.15),
        near_distance_min: float = Input(description="Camera distance for the shot", default=90.0),
        start_heading_deg: float = Input(description="Camera heading (0=N,90=E,180=S,270=W)", default=0.0),
        use_elevation: bool = Input(description="Use Open-Elevation for target altitude", default=True),
        default_alt: float = Input(description="Fallback target altitude (m ASL)", default=30.0),
        debug_urls: bool = Input(description="Print URL and step logs", default=True),
    ) -> List[Path]:

        elev = get_elevation_open_elevation(latitude, longitude) if use_elevation else None
        target_alt = elev if elev is not None else default_alt

        hero_tilt = 72.0
        hero_distance = near_distance_min
        hero_fov = 35.0
        hero_roll = 0.0
        hero_heading = start_heading_deg % 360.0

        hero_url = build_earth_url_with_search(
            address, latitude, longitude,
            a=target_alt,
            d=hero_distance,
            y=hero_fov,
            h=hero_heading,
            t=hero_tilt,
            r=hero_roll
        )

        if debug_urls:
            print("\n=== Generating single hero shot URL ===", flush=True)
            print(f"URL: {hero_url}", flush=True)
            print("======================================\n", flush=True)

        full_img_path = self._open_and_capture_new_tab(hero_url, w, h, wait_seconds, 1, debug_urls)
        
        cropped_img_path = "final_view.png"
        center_crop(full_img_path, cropped_img_path, crop_margin)
        
        return [Path(cropped_img_path)]