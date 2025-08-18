# predict.py
from typing import List, Optional, Tuple
from cog import BasePredictor, Input, Path

import time
import math
import requests
from urllib.parse import quote_plus
from PIL import Image
import io
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"


# ------------------------
# Helpers (geocoding, route, urls)
# ------------------------
def geocode_nominatim(address: str, contact_email: str) -> Tuple[float, float]:
    """Free geocoding via OpenStreetMap Nominatim (no key)."""
    headers = {"User-Agent": f"earth-route-cog/1.0 ({contact_email or 'contact@videotour.ai'})"}
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


def build_earth_url_with_search(address: str, lat: float, lon: float,
                                a: float, d: float, y: float, h: float, t: float, r: float = 0.0) -> str:
    """
    Google Earth Web URL including the search path:
    https://earth.google.com/web/search/{ADDRESS_URLENC}/@lat,lon,{a}a,{d}d,{y}y,{h}h,{t}t,{r}r
    """
    addr = quote_plus(address)
    return (f"https://earth.google.com/web/search/{addr}/"
            f"@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,{y:.2f}y,{h:.3f}h,{t:.3f}t,{r:.1f}r")


def ease_in_out_quad(x: float) -> float:
    return 2 * x * x if x < 0.5 else 1 - ((-2 * x + 2) ** 2) / 2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_drone_route(lat: float, lon: float, address: str,
                     use_elevation: bool, default_alt: float,
                     max_distance: float, near_distance_min: float,
                     start_heading_deg: float, clockwise: bool,
                     contact_email: str,
                     count: int = 10) -> List[dict]:
    """
    Returns an ordered list of steps with fields:
    {step, label, d, y, h, t, r, url}
    """
    # Elevation
    elev = get_elevation_open_elevation(lat, lon) if use_elevation else None
    a_target = elev if elev is not None else default_alt

    # Route design
    N = max(3, count)  # need >=3 to have establish/orbit/hero
    max_d = clamp(max_distance, 1.0, 195.0)
    near_d = max(near_distance_min, max_d - 70.0)

    tilt_approach = 55.0
    tilt_orbit = 67.0
    tilt_hero = 72.0
    fov_base = 35.0
    roll = 0.0

    orbit_steps = N - 2  # views 2..N-1 are the orbit
    step_sign = 1.0 if clockwise else -1.0
    heading_step = step_sign * (360.0 / orbit_steps)

    def dist_for_orbit_idx(i, total):
        x = i / max(1, total - 1)
        eased = ease_in_out_quad(x)
        return round(max_d - (max_d - near_d) * eased, 1)

    route = []

    # 1) Establishing
    route.append({
        "step": 1,
        "label": "Establishing (far, lower tilt)",
        "d": round(max_d, 1),
        "y": fov_base,
        "h": (start_heading_deg) % 360.0,
        "t": tilt_approach,
        "r": roll,
    })

    # 2..N-1) Orbit
    for i in range(orbit_steps):
        h = (start_heading_deg + heading_step * (i + 1)) % 360.0
        d = dist_for_orbit_idx(i, orbit_steps)
        y = fov_base + (((i % 2) * 2 - 1) * 3.0) if i % 2 else fov_base  # subtle FOV breathing
        route.append({
            "step": 2 + i,
            "label": f"Orbit {i + 1}/{orbit_steps}",
            "d": d,
            "y": y,
            "h": h,
            "t": tilt_orbit,
            "r": roll,
        })

    # N) Hero
    route.append({
        "step": N,
        "label": "Hero close-up",
        "d": round(near_d, 1),
        "y": fov_base,
        "h": (start_heading_deg + heading_step * orbit_steps) % 360.0,
        "t": tilt_hero,
        "r": roll,
    })

    # Build URLs
    for row in route:
        row["url"] = build_earth_url_with_search(
            address=address, lat=lat, lon=lon,
            a=a_target, d=row["d"], y=row["y"], h=row["h"], t=row["t"], r=row["r"]
        )

    return route


# ------------------------
# Image utilities
# ------------------------
def center_crop(image_path: str, output_path: str, crop_margin: float) -> None:
    """
    Center-crop by a per-side margin fraction.
    crop_margin = 0.15  -> crop 15% off each side (keeps central 70%).
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
        # Adjust paths/flags for your environment
        options.binary_location = os.getenv("CHROME_BINARY", "/root/chrome-linux/chrome")
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--lang=en-US")
        # WebGL friendliness (varies by container)
        options.add_argument("--use-gl=egl")
        options.add_argument("--enable-webgl")
        options.add_argument("--ignore-gpu-blocklist")
        self.browser = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.browser, 30)

    def _open_and_capture(self, url: str, w: int, h: int, wait_seconds: int, index: int) -> str:
        """Open URL, wait for Earth canvas to appear, then screenshot full window."""
        self.browser.set_window_size(w, h)
        self.browser.get(url)

        # Best-effort: dismiss potential dialogs/consents
        try:
            time.sleep(0.5)
            self.browser.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        except Exception:
            pass

        # Wait for any canvas to appear (Earth runs on WebGL canvas)
        try:
            self.wait.until(EC.presence_of_element_located((By.TAG_NAME, "canvas")))
        except Exception:
            # Continue anyway; screenshot whatever is present
            pass

        # Extra wait to let tiles/3D load
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        # Save full screenshot
        full_path = f"view_{index:02d}_full.png"
        self.browser.save_screenshot(full_path)
        return full_path

    def predict(
        self,
        address: str = Input(description="Address or place name to view in Google Earth Web"),
        w: int = Input(description="Viewport width for screenshots", default=1920),
        h: int = Input(description="Viewport height for screenshots", default=1080),
        wait_seconds: int = Input(description="Extra wait (seconds) after load before shooting", default=6),
        crop_margin: float = Input(
            description="Center-crop per-side margin fraction (e.g., 0.15 crops 15% off each side)",
            default=0.15,
        ),
        count: int = Input(description="How many views to generate in the route", default=10),
        max_distance: float = Input(description="Max camera distance (meters, ≤195)", default=195.0),
        near_distance_min: float = Input(description="Min 'near' distance for the hero shot", default=90.0),
        start_heading_deg: float = Input(description="Starting heading in degrees (0=N,90=E,180=S,270=W)", default=0.0),
        clockwise: bool = Input(description="Orbit direction (True=clockwise, False=counter-clockwise)", default=True),
        use_elevation: bool = Input(description="Use free Open-Elevation for target altitude", default=True),
        default_alt: float = Input(description="Fallback target altitude (meters ASL) if no elevation", default=30.0),
        contact_email: str = Input(description="Contact email for Nominatim User-Agent (polite use)", default="you@example.com"),
    ) -> List[Path]:
        """
        Generate a 10-step (default) 'drone' route around the address, screenshot each,
        center-crop with the given intensity, and return the cropped images in route order.
        """

        # 1) Geocode address
        lat, lon = geocode_nominatim(address, contact_email)
        # polite pause for Nominatim
        time.sleep(1.0)

        # 2) Build ordered route & URLs
        route = make_drone_route(
            lat=lat, lon=lon, address=address,
            use_elevation=use_elevation, default_alt=default_alt,
            max_distance=max_distance, near_distance_min=near_distance_min,
            start_heading_deg=start_heading_deg, clockwise=clockwise,
            contact_email=contact_email, count=count
        )

        # 3) Visit each URL and capture
        output_paths: List[Path] = []
        for i, step in enumerate(route, start=1):
            url = step["url"]
            full_img = self._open_and_capture(url, w=w, h=h, wait_seconds=wait_seconds, index=i)

            # 4) Center-crop with intensity
            cropped_img = f"view_{i:02d}.png"
            center_crop(full_img, cropped_img, crop_margin=crop_margin)

            # (Optional) keep or remove the full image; here we keep it on disk but only return cropped
            output_paths.append(Path(cropped_img))

        return output_paths
