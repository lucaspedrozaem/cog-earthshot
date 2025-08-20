from typing import List, Optional, Tuple
from cog import BasePredictor, Input, Path

import os
import time
import requests
from urllib.parse import quote_plus
from PIL import Image

from selenium import webdriver

# ---------- Service URLs ----------
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# New: Google Geocoding API URL
GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"


# ---------- Utils ----------
def get_elevation_open_elevation(lat: float, lon: float) -> Optional[float]:
    r = requests.get(OPEN_ELEVATION_URL, params={"locations": f"{lat},{lon}"}, timeout=20)
    if not r.ok:
        return None
    js = r.json()
    res = js.get("results") or []
    return float(res[0].get("elevation")) if res else None

def build_earth_url_with_search(address: str, lat: float, lon: float,
                                a: float, d: float, y: float, h: float, t: float, r: float = 0.0) -> str:
    addr = quote_plus(address) if address else ""
    return (f"https://earth.google.com/web/search/{addr}/"
            f"@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,{y:.2f}y,{h:.3f}h,{t:.3f}t,{r:.1f}r")

def center_crop(image_path: str, output_path: str, crop_margin: float) -> None:
    crop_margin = max(0.0, min(0.49, float(crop_margin)))
    with Image.open(image_path) as im:
        w, h = im.size
        left, right = int(w * crop_margin), int(w * (1.0 - crop_margin))
        top, bottom = int(h * crop_margin), int(h * (1.0 - crop_margin))
        im.crop((left, top, right, bottom)).save(output_path)

def try_parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    try:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None

# ---------- Geocoding ----------
def geocode_open_meteo(address: str) -> Optional[Tuple[float, float, str]]:
    """Primary free geocoder (no key). Returns (lat, lon, label) or None."""
    if not address:
        return None
    params = {"name": address, "count": 1, "language": "en", "format": "json"}
    r = requests.get(OPEN_METEO_GEOCODE_URL, params=params, timeout=20)
    if not r.ok:
        return None
    js = r.json() or {}
    results = js.get("results") or []
    if not results:
        return None
    hit = results[0]
    lat = float(hit["latitude"])
    lon = float(hit["longitude"])
    label = ", ".join([p for p in [hit.get("name"), hit.get("admin1"), hit.get("country_code")] if p])
    return lat, lon, label

def geocode_nominatim(address: str) -> Optional[Tuple[float, float, str]]:
    """Fallback free geocoder (keyless). Respect Nominatim's UA policy."""
    if not address:
        return None
    ua = os.getenv("GEOCODER_UA", "videotour-geocoder/1.0 (+https://example.com)")
    headers = {"User-Agent": ua}
    params = {"q": address, "format": "json", "limit": 1}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    if not r.ok:
        return None
    arr = r.json() or []
    if not arr:
        return None
    hit = arr[0]
    lat = float(hit["lat"])
    lon = float(hit["lon"])
    label = hit.get("display_name", address)
    return lat, lon, label

# New: Google Geocoding function
def geocode_google(address: str) -> Optional[Tuple[float, float, str]]:
    """Final fallback geocoder using Google Maps API. Requires GOOGLE_API_KEY env var."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not address or not api_key:
        return None
        
    params = {"address": address, "key": api_key}
    r = requests.get(GOOGLE_GEOCODE_URL, params=params, timeout=20)
    if not r.ok:
        return None
        
    js = r.json()
    if js.get("status") != "OK" or not js.get("results"):
        return None
        
    hit = js["results"][0]
    lat = float(hit["geometry"]["location"]["lat"])
    lon = float(hit["geometry"]["location"]["lng"])
    label = hit.get("formatted_address", address)
    return lat, lon, label

# Updated: geocode_address function
def geocode_address(address: str) -> Tuple[Optional[float], Optional[float], str]:
    """Resolve address → (lat, lon, label). Tries Open-Meteo, Nominatim, then Google."""
    # Accept "lat,lon" directly
    parsed = try_parse_latlon(address)
    if parsed:
        lat, lon = parsed
        return lat, lon, f"{lat:.6f}, {lon:.6f}"

    # Try services in order
    res = geocode_open_meteo(address)
    if res:
        return res
    res = geocode_nominatim(address)
    if res:
        return res
    res = geocode_google(address)
    if res:
        return res
        
    # If all fail, return None for coords
    return None, None, address


# ---------- Predictor ----------
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        options = webdriver.ChromeOptions()
        options.binary_location = '/root/chrome-linux/chrome'
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        self.browser = webdriver.Chrome(options=options)

    # open page + screenshot helper
    def _open_and_capture_new_tab(
        self,
        url: str,
        w: int,
        h: int,
        wait_until: int,
        index: int = 1,
        debug: bool = True
    ) -> str:
        self.browser.set_window_size(w, h)
        self.browser.get(url)

        if wait_until > 0:
            for i in range(wait_until):
                if debug:
                    print(f"Elapsed time: {i+1}/{wait_until} seconds", flush=True)
                time.sleep(1)

        if debug:
            print(f"Page title: {self.browser.title}", flush=True)
            print(f"Page URL: {self.browser.current_url}", flush=True)

        out_path = f"view_{index:02d}.png"
        self.browser.save_screenshot(out_path)
        return out_path

    def predict(
        self,
        address: str = Input(description="Address or 'lat,lon' of the target location"),
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

        # --- Geocode ---
        lat, lon, label = geocode_address(address)
        
        # Updated: Handle geocoding failure
        if lat is None or lon is None:
            fallback_url = f"https://earth.google.com/web/search/{quote_plus(address)}/"
            raise ValueError(
                f"Could not geocode address '{address}' with any service. "
                f"You can try this fallback URL manually: {fallback_url}"
            )
            
        if debug_urls:
            print(f"[Geocoding] {address!r} → ({lat:.6f}, {lon:.6f})  label={label}", flush=True)

        # --- Altitude ---
        elev = get_elevation_open_elevation(lat, lon) if use_elevation else None
        target_alt = elev if elev is not None else default_alt
        if debug_urls:
            src = "Open-Elevation" if elev is not None else "default_alt"
            print(f"[Altitude] Using {src}: {target_alt:.1f} m", flush=True)

        # --- Camera params ---
        hero_tilt = 72.0
        hero_distance = near_distance_min
        hero_fov = 35.0
        hero_roll = 0.0
        hero_heading = start_heading_deg % 360.0

        # --- Build URL ---
        hero_url = build_earth_url_with_search(
            label or address, lat, lon,
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

        # --- Capture ---
        full_img_path = self._open_and_capture_new_tab(
            hero_url, w, h, wait_until=wait_seconds, index=1, debug=debug_urls
        )

        # --- Crop & return ---
        cropped_img_path = "final_view.png"
        center_crop(full_img_path, cropped_img_path, crop_margin)
        return [Path(cropped_img_path)]