# predict.py — orbit-only shots, but PRIME each shot first (fixes "first frame only")
from typing import List, Optional, Tuple, Dict
from cog import BasePredictor, Input, Path

import os, time, requests
from urllib.parse import quote_plus

# ---- Optional Pillow (for cropping). If missing, we skip crop gracefully.
try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False
    print("Warning: pillow not installed; cropping will be skipped.")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"

# ----------------- Geocode / Elevation -----------------
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

# ----------------- Route & URLs -----------------
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
        base += ("&" if "?" in base else "?") + f"cb={cb}"
    return base

def make_orbit_route_only(
    lat: float, lon: float, address: str,
    use_elevation: bool, default_alt: float,
    max_distance: float, near_distance_min: float,
    start_heading_deg: float, clockwise: bool,
    sweep_degrees: float, contact_email: str,
    count: int = 10
) -> Tuple[List[Dict], float]:
    elev = get_elevation_open_elevation(lat, lon) if use_elevation else None
    a_target = elev if elev is not None else default_alt

    N = max(1, count)
    max_d = clamp(max_distance, 1.0, 195.0)
    near_d = max(near_distance_min, max_d - 70.0)

    tilt_orbit, fov_base, roll = 67.0, 35.0, 0.0

    step_sign = 1.0 if clockwise else -1.0
    step_deg = step_sign * (sweep_degrees / max(1, N))
    headings = [(start_heading_deg + step_deg * i) % 360.0 for i in range(N)]

    if N == 1:
        dists = [round((max_d + near_d) / 2.0, 1)]
    else:
        dists = [round(max_d - (max_d - near_d) * ease_in_out_quad(i/(N-1)), 1) for i in range(N)]

    route: List[Dict] = []
    ts_base = str(int(time.time()))
    for i in range(N):
        y = fov_base + (((i % 2) * 2 - 1) * 3.0) if i % 2 else fov_base
        route.append({
            "step": i + 1,
            "label": f"Route {i+1}/{N}",
            "d": dists[i], "y": y, "h": headings[i], "t": tilt_orbit, "r": roll,
            "url": build_earth_url_with_search(address, lat, lon, a=a_target, d=dists[i],
                                               y=y, h=headings[i], t=tilt_orbit, r=roll,
                                               cb=f"{ts_base}_{i}")
        })
    return route, a_target

def build_prime_url(address: str, lat: float, lon: float, a_target: float,
                    start_heading_deg: float, max_distance: float) -> str:
    """An establishing view to 'warm' Earth; not returned."""
    return build_earth_url_with_search(
        address, lat, lon,
        a=a_target, d=clamp(max_distance, 1.0, 195.0),
        y=35.0, h=start_heading_deg % 360.0, t=55.0, r=0.0, cb=str(time.time())
    )

# ----------------- Image utils -----------------
def center_crop(image_path: str, output_path: str, crop_margin: float) -> None:
    if not HAVE_PIL:
        # no pillow; just copy
        import shutil
        shutil.copyfile(image_path, output_path)
        return
    crop_margin = max(0.0, min(0.49, float(crop_margin)))
    with Image.open(image_path) as im:
        w, h = im.size
        left, right = int(w * crop_margin), int(w * (1.0 - crop_margin))
        top, bottom = int(h * crop_margin), int(h * (1.0 - crop_margin))
        im.crop((left, top, right, bottom)).save(output_path)

# ----------------- Predictor -----------------
class Predictor(BasePredictor):
    def _make_driver(self) -> webdriver.Chrome:
        opts = webdriver.ChromeOptions()
        opts.binary_location = '/root/chrome-linux/chrome'
        opts.add_argument('--headless=new')
        opts.add_argument('--no-sandbox')
        opts.add_argument('--disable-dev-shm-usage')
        # software WebGL is most reliable in headless
        opts.add_argument('--use-gl=swiftshader')
        opts.add_argument('--enable-webgl')
        opts.add_argument('--ignore-gpu-blocklist')
        opts.add_argument('--window-size=1920,1080')
        opts.add_argument('--lang=en-US')
        return webdriver.Chrome(options=opts)

    def setup(self) -> None:
        # one reusable driver (we'll still be able to use fresh sessions per shot)
        self.driver = self._make_driver()
        self.wait = WebDriverWait(self.driver, 30)

    # waits until canvas exists and splash is gone
    def _wait_scene_ready(self, driver, index: int, debug: bool) -> None:
        w = WebDriverWait(driver, 30)
        try:
            w.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
            if debug: print(f"[Step {index:02d}] Canvas detected.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] No canvas before timeout.", flush=True)
        try:
            w.until(lambda d: d.execute_script(
                "return document.body && document.body.innerText.indexOf('Google Earth') === -1;"
            ))
            if debug: print(f"[Step {index:02d}] Splash gone.", flush=True)
        except TimeoutException:
            if debug: print(f"[Step {index:02d}] Splash still visible; continuing.", flush=True)
        try:
            ActionChains(driver).move_by_offset(5, 5).click().perform()
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ARROW_UP)
        except Exception:
            pass

    def _open_and_capture(self, driver, url: str, w: int, h: int, wait_seconds: int,
                          index: int, debug: bool, note: str) -> str:
        driver.set_window_size(w, h)
        if debug: print(f"[Step {index:02d}] GET ({note}): {url}", flush=True)
        driver.get(url)
        try:
            time.sleep(0.5)
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        except Exception:
            pass
        self._wait_scene_ready(driver, index, debug)
        if wait_seconds > 0:
            if debug: print(f"[Step {index:02d}] Extra wait: {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)
        if debug:
            try:
                print(f"[Step {index:02d}] Loaded: {driver.current_url}", flush=True)
                print(f"[Step {index:02d}] Title: {driver.title}", flush=True)
            except Exception: pass
        path_full = f"view_{index:02d}_full.png"
        driver.save_screenshot(path_full)
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
        fresh_session_per_shot: bool = Input(description="NEW Chrome per shot", default=True),
        prime_each_shot: bool = Input(description="Load a hidden establishing view before each route shot", default=True),
    ) -> List[Path]:

        lat, lon = geocode_nominatim(address, contact_email)
        time.sleep(1.0)

        route, a_target = make_orbit_route_only(
            lat=lat, lon=lon, address=address,
            use_elevation=use_elevation, default_alt=default_alt,
            max_distance=max_distance, near_distance_min=near_distance_min,
            start_heading_deg=start_heading_deg, clockwise=clockwise,
            sweep_degrees=sweep_degrees, contact_email=contact_email,
            count=count
        )

        if debug_urls:
            print("\n=== Orbit route URLs (ordered) ===", flush=True)
            for s in route: print(f"[{s['step']:02d}] {s['label']}: {s['url']}", flush=True)
            print("=== End URLs ===\n", flush=True)

        outputs: List[Path] = []
        prime_url = build_prime_url(address, lat, lon, a_target, start_heading_deg, max_distance)

        for i, step in enumerate(route, start=1):
            # fresh session per shot is the most reliable
            driver = self._make_driver() if fresh_session_per_shot else self.driver
            try:
                if prime_each_shot:
                    # Warm Earth in this session (not returned)
                    self._open_and_capture(driver, prime_url, w, h, wait_seconds//2, i, debug_urls, note="PRIME")
                full_img = self._open_and_capture(driver, step["url"], w, h, wait_seconds, i, debug_urls, note="ROUTE")
            finally:
                if fresh_session_per_shot:
                    try: driver.quit()
                    except Exception: pass
                else:
                    try:
                        self.driver.delete_all_cookies()
                        self.driver.get("about:blank")
                    except Exception:
                        pass

            cropped = f"view_{i:02d}.png"
            center_crop(full_img, cropped, crop_margin)
            outputs.append(Path(cropped))

        return outputs
