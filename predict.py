# predict.py — orbit-only, robust against "first frame only"
from typing import List, Optional, Tuple, Dict
from cog import BasePredictor, Input, Path

import os
import time
import hashlib
import requests
from urllib.parse import quote_plus

from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains

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
                                a: float, d: float, y: float, h: float, t: float,
                                r: float = 0.0, cb: str = "") -> str:
    addr = quote_plus(address)
    base = (f"https://earth.google.com/web/search/{addr}/"
            f"@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,{y:.2f}y,{h:.3f}h,{t:.3f}t,{r:.1f}r")
    if cb:
        sep = "&" if "?" in base else "?"
        base = f"{base}{sep}cb={cb}"
    return base

def make_orbit_route_only(
    lat: float, lon: float, address: str,
    use_elevation: bool, default_alt: float,
    max_distance: float, near_distance_min: float,
    start_heading_deg: float, clockwise: bool,
    sweep_degrees: float, contact_email: str,
    count: int = 10
) -> List[Dict]:
    elev = get_elevation_open_elevation(lat, lon) if use_elevation else None
    a_target = elev if elev is not None else default_alt

    N = max(1, count)
    max_d = clamp(max_distance, 1.0, 195.0)
    near_d = max(near_distance_min, max_d - 70.0)

    tilt_orbit = 67.0
    fov_base = 35.0
    roll = 0.0

    step_sign = 1.0 if clockwise else -1.0
    # Spread N frames over the arc without repeating the final angle
    step_deg = step_sign * (sweep_degrees / max(1, N))
    headings = [(start_heading_deg + step_deg * i) % 360.0 for i in range(N)]

    # Distances: smooth push-in from far -> near
    if N == 1:
        dists = [round((max_d + near_d) / 2.0, 1)]
    else:
        dists = [round(max_d - (max_d - near_d) * ease_in_out_quad(i/(N-1)), 1) for i in range(N)]

    route: List[Dict] = []
    ts_base = str(int(time.time()))
    for i in range(N):
        y = fov_base + (((i % 2) * 2 - 1) * 3.0) if i % 2 else fov_base  # subtle FOV breathing
        cachebuster = f"{ts_base}_{i}"
        row = {
            "step": i + 1,
            "label": f"Route {i+1}/{N}",
            "d": dists[i],
            "y": y,
            "h": headings[i],
            "t": tilt_orbit,
            "r": roll,
        }
        row["url"] = build_earth_url_with_search(
            address, lat, lon, a=a_target, d=row["d"], y=row["y"],
            h=row["h"], t=row["t"], r=row["r"], cb=cachebuster
        )
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
    def _make_driver(self) -> webdriver.Chrome:
        """Create a fresh Chrome session configured for software WebGL."""
        options = webdriver.ChromeOptions()
        # Use the known working Chrome build you already download in cog.yaml:
        options.binary_location = '/root/chrome-linux/chrome'
        # Headless NEW supports GPU in headless; combine with swiftshader for WebGL.
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # IMPORTANT: do not disable GPU; use software WebGL fallback
        options.add_argument('--use-gl=swiftshader')
        options.add_argument('--enable-webgl')
        options.add_argument('--ignore-gpu-blocklist')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--lang=en-US')
        return webdriver.Chrome(options=options)

    def setup(self) -> None:
        """Keep one driver for optional reuse; we’ll also support fresh-per-shot."""
        self.driver = self._make_driver()
        self.wait = WebDriverWait(self.driver, 30)

    def _wait_scene_ready(self, driver: webdriver.Chrome, index: int, debug: bool) -> None:
        wait = WebDriverWait(driver, 30)
        # 1) canvas present
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
            if debug: print(f"[Step {index:02d}] Canvas detected.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] No canvas before timeout.", flush=True)

        # 2) splash “Google Earth” text gone
        try:
            wait.until(lambda d: d.execute_script(
                "return document.body && document.body.innerText.indexOf('Google Earth') === -1;"
            ))
            if debug: print(f"[Step {index:02d}] Splash gone.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] Splash still visible; continuing.", flush=True)

        # 3) ensure canvas visible (opacity ~1)
        try:
            visible = driver.execute_script("""
                const c = document.querySelector('canvas');
                if (!c) return false;
                const s = getComputedStyle(c);
                return (s && s.visibility !== 'hidden' && s.opacity === '1');
            """)
            if debug: print(f"[Step {index:02d}] Canvas visible: {visible}", flush=True)
        except Exception:
            pass

        # 4) interact to “wake” renderer
        try:
            ActionChains(driver).move_by_offset(10, 10).click().perform()
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
        except Exception:
            pass

    def _open_and_capture(self, driver: webdriver.Chrome, url: str, w: int, h: int,
                          wait_seconds: int, index: int, debug: bool) -> str:
        driver.set_window_size(w, h)
        if debug: print(f"[Step {index:02d}] GET: {url}", flush=True)
        driver.get(url)

        # Try to dismiss overlays
        try:
            time.sleep(0.5)
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        except Exception:
            pass

        self._wait_scene_ready(driver, index, debug)

        if wait_seconds > 0:
            if debug: print(f"[Step {index:02d}] Extra wait: {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)

        # Diagnostics
        if debug:
            try:
                print(f"[Step {index:02d}] Loaded URL: {driver.current_url}", flush=True)
                print(f"[Step {index:02d}] Title: {driver.title}", flush=True)
                print(f"[Step {index:02d}] Canvases: {len(driver.find_elements(By.TAG_NAME, 'canvas'))}", flush=True)
            except Exception as e:
                print(f"[Step {index:02d}] Debug err: {e}", flush=True)

        path_full = f"view_{index:02d}_full.png"
        driver.save_screenshot(path_full)
        if debug: print(f"[Step {index:02d}] Saved: {path_full}", flush=True)
        return path_full

    def predict(
        self,
        address: str = Input(description="Address or place name"),
        w: int = Input(description="Viewport width", default=1920),
        h: int = Input(description="Viewport height", default=1080),
        wait_seconds: int = Input(description="Extra wait after load", default=10),
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
        fresh_session_per_shot: bool = Input(description="Open a brand-new Chrome for each shot", default=True),
    ) -> List[Path]:

        # 1) Geocode
        lat, lon = geocode_nominatim(address, contact_email)
        time.sleep(1.0)

        # 2) Build ordered orbit-only route
        route = make_orbit_route_only(
            lat=lat, lon=lon, address=address,
            use_elevation=use_elevation, default_alt=default_alt,
            max_distance=max_distance, near_distance_min=near_distance_min,
            start_heading_deg=start_heading_deg, clockwise=clockwise,
            sweep_degrees=sweep_degrees, contact_email=contact_email,
            count=count
        )

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
                print(f"urls.txt write failed: {e}", flush=True)

        outputs: List[Path] = []

        # 3) Capture
        for i, step in enumerate(route, start=1):
            if fresh_session_per_shot:
                driver = self._make_driver()
                try:
                    path_full = self._open_and_capture(driver, step["url"], w, h, wait_seconds, i, debug_urls)
                finally:
                    try:
                        driver.quit()
                    except Exception:
                        pass
            else:
                path_full = self._open_and_capture(self.driver, step["url"], w, h, wait_seconds, i, debug_urls)
                # Reset SPA state between shots when reusing:
                try:
                    self.driver.delete_all_cookies()
                    self.driver.get("about:blank")
                except Exception:
                    pass

            # 4) Crop
            cropped = f"view_{i:02d}.png"
            center_crop(path_full, cropped, crop_margin)
            outputs.append(Path(cropped))

        return outputs
