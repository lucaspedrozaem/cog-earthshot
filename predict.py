# predict.py
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

# ------------------------
# External services
# ------------------------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


# ------------------------
# Helpers (geocoding, elevation)
# ------------------------
def geocode_nominatim(address: str, contact_email: str) -> Tuple[float, float]:
    """
    Free geocoding via OpenStreetMap Nominatim (no key).
    Respect usage policy: identify your app via User-Agent and keep rate modest.
    """
    headers = {"User-Agent": f"earth-route-cog/1.0 ({contact_email or 'contact@example.com'})"}
    params = {"q": address, "format": "jsonv2", "limit": 1, "addressdetails": 0}
    r = requests.get(NOMINATIM_URL, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    if not js:
        raise RuntimeError("Geocoding failed: no results from Nominatim.")
    return float(js[0]["lat"]), float(js[0]["lon"])


def get_elevation_open_elevation(lat: float, lon: float) -> Optional[float]:
    """Optional free elevation lookup (community service; be gentle)."""
    r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=20)
    if not r.ok:
        return None
    js = r.json()
    res = js.get("results") or []
    return float(res[0].get("elevation")) if res else None


# ------------------------
# URL building & route design
# ------------------------
def build_earth_url_with_search(
    address: str,
    lat: float,
    lon: float,
    a: float,
    d: float,
    y: float,
    h: float,
    t: float,
    r: float = 0.0,
) -> str:
    """
    Google Earth Web URL including the search path:
      https://earth.google.com/web/search/{ADDRESS_URLENC}/@lat,lon,{a}a,{d}d,{y}y,{h}h,{t}t,{r}r
    """
    addr = quote_plus(address)
    return (
        f"https://earth.google.com/web/search/{addr}/"
        f"@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,{y:.2f}y,{h:.3f}h,{t:.3f}t,{r:.1f}r"
    )


def ease_in_out_quad(x: float) -> float:
    """Smooth easing 0..1 -> 0..1."""
    return 2 * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_drone_route(
    lat: float,
    lon: float,
    address: str,
    use_elevation: bool,
    default_alt: float,
    max_distance: float,
    near_distance_min: float,
    start_heading_deg: float,
    clockwise: bool,
    contact_email: str,
    count: int = 10,
) -> List[Dict]:
    """
    Returns an ordered list of steps with fields:
    {step, label, d, y, h, t, r, url}
    representing a smooth establishing shot -> clockwise/CCW orbit -> hero close-up.
    """
    # Elevation for target altitude
    elev = get_elevation_open_elevation(lat, lon) if use_elevation else None
    a_target = elev if elev is not None else default_alt

    # Route design
    N = max(3, count)  # need >=3 for establish/orbit/hero
    max_d = clamp(max_distance, 1.0, 195.0)
    near_d = max(near_distance_min, max_d - 70.0)  # push-in target

    # Camera flavors
    tilt_approach = 55.0
    tilt_orbit = 67.0
    tilt_hero = 72.0
    fov_base = 35.0
    roll = 0.0

    orbit_steps = N - 2  # views 2..N-1 are the orbit
    step_sign = 1.0 if clockwise else -1.0
    heading_step = step_sign * (360.0 / orbit_steps)

    def dist_for_orbit_idx(i: int, total: int) -> float:
        x = i / max(1, total - 1)
        eased = ease_in_out_quad(x)
        return round(max_d - (max_d - near_d) * eased, 1)

    route: List[Dict] = []

    # 1) Establishing
    route.append(
        {
            "step": 1,
            "label": "Establishing (far, lower tilt)",
            "d": round(max_d, 1),
            "y": fov_base,
            "h": (start_heading_deg) % 360.0,
            "t": tilt_approach,
            "r": roll,
        }
    )

    # 2..N-1) Orbit
    for i in range(orbit_steps):
        h = (start_heading_deg + heading_step * (i + 1)) % 360.0
        d = dist_for_orbit_idx(i, orbit_steps)
        # Subtle FOV breathing to add life
        y = fov_base + (((i % 2) * 2 - 1) * 3.0) if i % 2 else fov_base
        route.append(
            {
                "step": 2 + i,
                "label": f"Orbit {i + 1}/{orbit_steps}",
                "d": d,
                "y": y,
                "h": h,
                "t": tilt_orbit,
                "r": roll,
            }
        )

    # N) Hero
    route.append(
        {
            "step": N,
            "label": "Hero close-up",
            "d": round(near_d, 1),
            "y": fov_base,
            "h": (start_heading_deg + heading_step * orbit_steps) % 360.0,
            "t": tilt_hero,
            "r": roll,
        }
    )

    # Build URLs with search segment
    for row in route:
        row["url"] = build_earth_url_with_search(
            address=address,
            lat=lat,
            lon=lon,
            a=a_target,
            d=row["d"],
            y=row["y"],
            h=row["h"],
            t=row["t"],
            r=row["r"],
        )

    return route


# ------------------------
# Image utilities
# ------------------------
def center_crop(image_path: str, output_path: str, crop_margin: float) -> None:
    """
    Center-crop by a per-side margin fraction.
    crop_margin = 0.15 -> crop 15% off each side (keeps central 70%).
    """
    crop_margin = max(0.0, min(0.49, float(crop_margin)))
    with Image.open(image_path) as im:
        w, h = im.size
        left = int(w * crop_margin)
        right = int(w * (1.0 - crop_margin))
        top = int(h * crop_margin)
        bottom = int(h * (1.0 - crop_margin))
        cropped = im.crop((left, top, right, bottom))
        cropped.save(output_path)


# ------------------------
# Cog Predictor
# ------------------------
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Prepare headless Chrome once for all predictions."""
        options = webdriver.ChromeOptions()
        # Use env override if provided; otherwise default to a known path (adjust to your image)
        options.binary_location = os.getenv("CHROME_BINARY", "/root/chrome-linux/chrome")

        # Headless + container-friendly flags
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--lang=en-US")

        # Try to keep WebGL enabled for Google Earth
        options.add_argument("--use-gl=egl")
        options.add_argument("--enable-webgl")
        options.add_argument("--ignore-gpu-blocklist")

        self.browser = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.browser, 30)

    def _open_and_capture(
        self,
        url: str,
        w: int,
        h: int,
        wait_seconds: int,
        index: int,
        debug: bool = False,
    ) -> str:
        """Open URL, wait for Earth canvas, then screenshot."""
        if debug:
            print(f"[Step {index:02d}] Navigating to: {url}", flush=True)

        self.browser.set_window_size(w, h)
        self.browser.get(url)

        # Try dismissing overlays (escape)
        try:
            time.sleep(0.5)
            self.browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        except Exception:
            pass

        # Wait for at least one canvas (Earth WebGL)
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
            if debug:
                print(f"[Step {index:02d}] Canvas detected.", flush=True)
        except TimeoutException:
            if debug:
                print(f"[Step {index:02d}] No canvas detected before timeout.", flush=True)

        # Extra wait to let tiles/3D load
        if wait_seconds > 0:
            if debug:
                print(f"[Step {index:02d}] Extra wait: {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)

        # Post-load diagnostics
        if debug:
            try:
                print(f"[Step {index:02d}] Loaded URL: {self.browser.current_url}", flush=True)
                print(f"[Step {index:02d}] Title: {self.browser.title}", flush=True)
                canvases = self.browser.find_elements(By.TAG_NAME, "canvas")
                print(f"[Step {index:02d}] Canvas elements: {len(canvases)}", flush=True)
            except Exception as e:
                print(f"[Step {index:02d}] Debug read failed: {e}", flush=True)

        # Screenshot full window
        full_path = f"view_{index:02d}_full.png"
        ok = self.browser.save_screenshot(full_path)
        if debug:
            print(f"[Step {index:02d}] Screenshot saved: {full_path} (ok={ok})", flush=True)
        return full_path

    def predict(
        self,
        address: str = Input(description="Address or place name to view in Google Earth Web"),
        w: int = Input(description="Viewport width for screenshots", default=1920),
        h: int = Input(description="Viewport height for screenshots", default=1080),
        wait_seconds: int = Input(
            description="Extra wait (seconds) after load before shooting", default=6
        ),
        crop_margin: float = Input(
            description="Center-crop per-side margin fraction (e.g., 0.15 crops 15%% off each side)",
            default=0.15,
        ),
        count: int = Input(description="How many views to generate in the route", default=10),
        max_distance: float = Input(
            description="Max camera distance (meters, ≤195)", default=195.0
        ),
        near_distance_min: float = Input(
            description="Min 'near' distance for the hero shot", default=90.0
        ),
        start_heading_deg: float = Input(
            description="Starting heading in degrees (0=N,90=E,180=S,270=W)", default=0.0
        ),
        clockwise: bool = Input(description="Orbit direction (True=clockwise, False=CCW)", default=True),
        use_elevation: bool = Input(
            description="Use free Open-Elevation for target altitude", default=True
        ),
        default_alt: float = Input(
            description="Fallback target altitude (meters ASL) if no elevation", default=30.0
        ),
        contact_email: str = Input(
            description="Contact email for Nominatim User-Agent (polite use)",
            default="you@example.com",
        ),
        debug_urls: bool = Input(
            description="Print URLs and loaded pages to logs", default=True
        ),
    ) -> List[Path]:
        """
        Build a 3D 'drone' route (establish -> orbit -> hero), screenshot each view,
        center-crop, and return the cropped images in route order.
        """

        # 1) Geocode
        lat, lon = geocode_nominatim(address, contact_email)
        # polite pause for Nominatim (esp. if batching)
        time.sleep(1.0)

        # 2) Route + URLs
        route = make_drone_route(
            lat=lat,
            lon=lon,
            address=address,
            use_elevation=use_elevation,
            default_alt=default_alt,
            max_distance=max_distance,
            near_distance_min=near_distance_min,
            start_heading_deg=start_heading_deg,
            clockwise=clockwise,
            contact_email=contact_email,
            count=count,
        )

        # Debug: print URLs in order
        if debug_urls:
            print("\n=== Generated route URLs (ordered) ===", flush=True)
            for step in route:
                print(f"[{step['step']:02d}] {step['label']}: {step['url']}", flush=True)
            print("=== End route URLs ===\n", flush=True)
            # Also write a urls.txt artifact (not returned)
            try:
                with open("urls.txt", "w", encoding="utf-8") as f:
                    for step in route:
                        f.write(f"{step['step']:02d}\t{step['label']}\t{step['url']}\n")
                print("Wrote urls.txt with all route URLs.", flush=True)
            except Exception as e:
                print(f"Failed to write urls.txt: {e}", flush=True)

        # 3) Visit each URL and capture
        output_paths: List[Path] = []
        for i, step in enumerate(route, start=1):
            url = step["url"]
            full_img = self._open_and_capture(
                url, w=w, h=h, wait_seconds=wait_seconds, index=i, debug=debug_urls
            )

            # 4) Center-crop with intensity
            cropped_img = f"view_{i:02d}.png"
            center_crop(full_img, cropped_img, crop_margin=crop_margin)

            # Return only the cropped image
            output_paths.append(Path(cropped_img))

        return output_paths
