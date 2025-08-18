# predict.py — orbit-only route shots
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
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ---------- Geocode / elevation ----------
def geocode_nominatim(address: str, contact_email: str) -> Tuple[float, float]:
    headers = {"User-Agent": f"earth-route-cog/1.0 ({contact_email or 'contact@example.com'})"}
    params = {"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 0}
    r = requests.get(NOMINATIM_URL, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise RuntimeError("Geocoding failed: no results from Nominatim.")
    return float(js[0]["lat"]), float(js[0]["lon"])

def get_elevation_open_elevation(lat: float, lon: float) -> Optional[float]:
    r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=20)
    if not r.ok:
        return None
    js = r.json()
    res = js.get("results") or []
    return float(res[0].get("elevation")) if res else None

# ---------- Route & URLs ----------
def ease_in_out_quad(x: float) -> float:
    return 2*x*x if x < 0.5 else 1 - ((-2*x + 2) ** 2) / 2

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def build_earth_url_with_search(address: str, lat: float, lon: float,
                                a: float, d: float, y: float, h: float, t: float, r: float = 0.0) -> str:
    addr = quote_plus(address)
    return (f"https://earth.google.com/web/search/{addr}/"
            f"@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,{y:.2f}y,{h:.3f}h,{t:.3f}t,{r:.1f}r")

def make_orbit_route_only(
    lat: float, lon: float, address: str,
    use_elevation: bool, default_alt: float,
    max_distance: float, near_distance_min: float,
    start_heading_deg: float, clockwise: bool,
    sweep_degrees: float,                      # NEW: choose full/partial orbit
    contact_email: str, count: int = 10
) -> List[Dict]:
    """
    Returns ONLY orbit shots (length = count), ordered as a smooth route.
    Each step has: {step, label, d, y, h, t, r, url}
    """
    elev = get_elevation_open_elevation(lat, lon) if use_elevation else None
    a_target = elev if elev is not None else default_alt

    N = max(1, count)
    max_d = clamp(max_distance, 1.0, 195.0)
    near_d = max(near_distance_min, max_d - 70.0)  # push-in target

    tilt_orbit = 67.0
    fov_base = 35.0
    roll = 0.0

    # Headings across an arc (default 360°). Do not repeat the start angle on the last frame.
    step_sign = 1.0 if clockwise else -1.0
    step_deg = step_sign * (sweep_degrees / max(1, N))  # N frames, last < start+sweep
    headings = [(start_heading_deg + step_deg * i) % 360.0 for i in range(N)]

    # Distances: ease from far -> near across the N orbit shots
    if N == 1:
        dists = [round((max_d + near_d) / 2.0, 1)]
    else:
        dists = [round(max_d - (max_d - near_d) * ease_in_out_quad(i/(N-1)), 1) for i in range(N)]

    route: List[Dict] = []
    for i in range(N):
        y = fov_base + (((i % 2) * 2 - 1) * 3.0) if i % 2 else fov_base  # subtle FOV breathing
        row = {
            "step": i + 1,
            "label": f"Route {i+1}/{N}",
            "d": dists[i],
            "y": y,
            "h": headings[i],
            "t": tilt_orbit,
            "r": roll,
        }
        row["url"] = build_earth_url_with_search(address, lat, lon,
                                                 a=a_target, d=row["d"], y=row["y"],
                                                 h=row["h"], t=row["t"], r=row["r"])
        route.append(row)

    return route

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
        options.binary_location = '/root/chrome-linux/chrome'  # same path you used successfully
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # These help WebGL in some containers (optional)
        options.add_argument("--use-gl=egl")
        options.add_argument("--enable-webgl")
        options.add_argument("--ignore-gpu-blocklist")

        self.browser = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.browser, 30)

    def _wait_scene_ready(self, index: int, debug: bool) -> None:
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
            if debug: print(f"[Step {index:02d}] Canvas detected.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] No canvas before timeout.", flush=True)

        # Wait for splash text to disappear
        try:
            self.wait.until(lambda d: d.execute_script(
                "return document.body && document.body.innerText.indexOf('Google Earth') === -1;"
            ))
            if debug: print(f"[Step {index:02d}] Splash gone.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] Splash still visible; continuing.", flush=True)

    def _open_and_capture_new_tab(self, url: str, w: int, h: int, wait_seconds: int, index: int, debug: bool) -> str:
        self.browser.switch_to.new_window('tab')            # NEW TAB per shot
        self.browser.set_window_size(w, h)
        if debug: print(f"[Step {index:02d}] Opening NEW TAB: {url}", flush=True)

        self.browser.get(url)
        try:
            time.sleep(0.5)
            self.browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        except Exception:
            pass

        self._wait_scene_ready(index, debug)

        if wait_seconds > 0:
            if debug: print(f"[Step {index:02d}] Extra wait: {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)

        # Tiny nudge to force a render tick
        try:
            self.browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
        except Exception:
            pass

        full_path = f"view_{index:02d}_full.png"
        ok = self.browser.save_screenshot(full_path)
        if debug: print(f"[Step {index:02d}] Screenshot saved: {full_path} (ok={ok})", flush=True)

        self.browser.close()
        self.browser.switch_to.window(self.browser.window_handles[0])
        return full_path

    def predict(
        self,
        address: str = Input(description="Address or place name"),
        w: int = Input(description="Viewport width", default=1920),
        h: int = Input(description="Viewport height", default=1080),
        wait_seconds: int = Input(description="Extra wait after load", default=12),
        crop_margin: float = Input(description="Center-crop margin per side (0–0.49)", default=0.15),
        count: int = Input(description="Number of route shots (orbit frames)", default=10),
        max_distance: float = Input(description="Max camera distance (≤195)", default=195.0),
        near_distance_min: float = Input(description="Min near distance", default=90.0),
        start_heading_deg: float = Input(description="Start heading (0=N,90=E,180=S,270=W)", default=0.0),
        clockwise: bool = Input(description="Orbit clockwise?", default=True),
        sweep_degrees: float = Input(description="Total orbit sweep in degrees (e.g., 360, 180, 90)", default=360.0),
        use_elevation: bool = Input(description="Use Open-Elevation for target altitude", default=True),
        default_alt: float = Input(description="Fallback target altitude (m ASL)", default=30.0),
        contact_email: str = Input(description="Contact email for Nominatim UA", default="you@example.com"),
        debug_urls: bool = Input(description="Print URLs and step logs", default=True),
    ) -> List[Path]:

        # 1) Geocode
        lat, lon = geocode_nominatim(address, contact_email)
        time.sleep(1.0)  # polite pause

        # 2) Build ordered ORBIT-ONLY route
        route = make_orbit_route_only(
            lat=lat, lon=lon, address=address,
            use_elevation=use_elevation, default_alt=default_alt,
            max_distance=max_distance, near_distance_min=near_distance_min,
            start_heading_deg=start_heading_deg, clockwise=clockwise,
            sweep_degrees=sweep_degrees, contact_email=contact_email,
            count=count
        )

        # 3) Log URLs
        if debug_urls:
            print("\n=== Orbit-only route URLs (ordered) ===", flush=True)
            for step in route:
                print(f"[{step['step']:02d}] {step['label']}: {step['url']}", flush=True)
            print("=== End URLs ===\n", flush=True)
            try:
                with open("urls.txt", "w", encoding="utf-8") as f:
                    for step in route:
                        f.write(f"{step['step']:02d}\t{step['label']}\t{step['url']}\n")
            except Exception as e:
                print(f"Failed to write urls.txt: {e}", flush=True)

        # 4) Capture each view (fresh tab per URL), crop, return
        outputs: List[Path] = []
        for i, step in enumerate(route, start=1):
            full_img = self._open_and_capture_new_tab(step["url"], w, h, wait_seconds, i, debug_urls)
            cropped = f"view_{i:02d}.png"
            center_crop(full_img, cropped, crop_margin)
            outputs.append(Path(cropped))

        return outputs
