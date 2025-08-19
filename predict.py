from typing import List, Optional, Tuple, Dict
from cog import BasePredictor, Input, Path

import os
import time
import requests
from urllib.parse import quote_plus

from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
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
        options.add_argument('--disable-gpu')
        self.browser = webdriver.Chrome(options=options)
        
    def _open_and_capture(self, url: str, w: int, h: int, wait_seconds: int, debug: bool) -> str:
        main_window = self.browser.current_window_handle
        
        self.browser.switch_to.new_window('tab')
        self.browser.set_window_size(w, h)
        if debug: print(f"Opening URL: {url}", flush=True)
        self.browser.get(url)

        # --- DEFINITIVE MULTI-STEP WAIT ---
        try:
            wait = WebDriverWait(self.browser, 60) # Generous 60-second timeout for the entire process

            # Step 1: Wait for the canvas element to exist.
            if debug: print("Waiting for canvas to appear...", flush=True)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
            if debug: print("✅ Canvas appeared.", flush=True)

            # Step 2: Wait for a key UI element (the search button) to be ready.
            if debug: print("Waiting for UI to be ready (checking for search button)...", flush=True)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'button[aria-label="Search"]')))
            if debug: print("✅ UI is ready.", flush=True)

        except TimeoutException:
            if debug: print("⚠️ Main interface did not load within the timeout period.", flush=True)
        # ------------------------------------

        # Final fixed wait for high-resolution 3D tiles to stream in.
        if wait_seconds > 0:
            if debug: print(f"Starting final wait for 3D textures: {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)
            if debug: print("Final wait finished.", flush=True)

        # Screenshot and close the tab
        full_path = "screenshot_full.png"
        if debug: print("Attempting to save screenshot...", flush=True)
        self.browser.save_screenshot(full_path)
        if debug: print(f"Screenshot saved to {full_path}", flush=True)
        
        self.browser.close()
        self.browser.switch_to.window(main_window)
        return full_path

    def predict(
        self,
        latitude: float = Input(description="Latitude of the target location"),
        longitude: float = Input(description="Longitude of the target location"),
        address: str = Input(description="Optional address or label for the location", default=""),
        w: int = Input(description="Viewport width", default=1920),
        h: int = Input(description="Viewport height", default=1080),
        wait_seconds: int = Input(description="Extra time (seconds) to wait for high-res graphics to load", default=10),
        crop_margin: float = Input(description="Center-crop margin per side (0–0.49)", default=0.15),
        near_distance_min: float = Input(description="Camera distance for the shot", default=90.0),
        start_heading_deg: float = Input(description="Camera heading (0=N,90=E,180=S,270=W)", default=0.0),
        use_elevation: bool = Input(description="Use Open-Elevation for target altitude", default=True),
        default_alt: float = Input(description="Fallback target altitude (m ASL)", default=30.0),
        debug_urls: bool = Input(description="Print URL and step logs", default=True),
    ) -> Path:

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
            print(f"\nURL: {hero_url}", flush=True)

        full_img_path = self._open_and_capture(hero_url, w, h, wait_seconds, debug_urls)
        
        cropped_img_path = "final_view.png"
        center_crop(full_img_path, cropped_img_path, crop_margin)
        
        return Path(cropped_img_path)