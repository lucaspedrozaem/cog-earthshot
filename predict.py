from cog import BasePredictor, Input, Path
from urllib.parse import quote_plus
from PIL import Image
import time

from selenium import webdriver

# ---------- URL Building ----------
def build_earth_url(lat: float, lon: float, a: float, d: float, h: float, t: float) -> str:
    """Builds the direct Google Earth URL without the search parameter."""
    return f"https://earth.google.com/web/@{lat:.7f},{lon:.7f},{a:.1f}a,{d:.1f}d,35y,{h:.3f}h,{t:.3f}t,0r"

# ---------- Image Cropping ----------
def center_crop(image_path: str, output_path: str, crop_margin: float) -> None:
    """Crops the center of an image."""
    crop_margin = max(0.0, min(0.49, float(crop_margin)))
    with Image.open(image_path) as im:
        w, h = im.size
        left, right = int(w * crop_margin), int(w * (1.0 - crop_margin))
        top, bottom = int(h * crop_margin), int(h * (1.0 - crop_margin))
        im.crop((left, top, right, bottom)).save(output_path)

# ---------- Predictor ----------
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Sets up the browser."""
        options = webdriver.ChromeOptions()
        options.binary_location = '/root/chrome-linux/chrome'
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        self.browser = webdriver.Chrome(options=options)

    def predict(
        self,
        latitude: float = Input(description="Latitude of the target location"),
        longitude: float = Input(description="Longitude of the target location"),
        wait_seconds: int = Input(description="Time in seconds to wait before taking the screenshot", default=15),
        distance: float = Input(description="Camera distance from the target in meters", default=100.0),
        heading: float = Input(description="Camera heading in degrees (0=N, 90=E)", default=0.0),
        tilt: float = Input(description="Camera tilt in degrees (0=top-down, 90=horizon)", default=70.0),
        altitude: float = Input(description="Altitude of the ground in meters (ASL)", default=30.0),
        w: int = Input(description="Width of the screenshot", default=1920),
        h: int = Input(description="Height of the screenshot", default=1080),
        crop_margin: float = Input(description="Center-crop margin per side (0–0.49)", default=0.15),
    ) -> Path:
        """Runs the prediction."""
        
        # 1. Build the Google Earth URL from the inputs
        url = build_earth_url(
            lat=latitude,
            lon=longitude,
            a=altitude,
            d=distance,
            h=heading,
            t=tilt,
        )
        print(f"Generated URL: {url}")

        # 2. Open the page and wait for a fixed time
        self.browser.set_window_size(w, h)
        self.browser.get(url)
        
        print(f"Waiting for {wait_seconds} seconds...")
        time.sleep(wait_seconds)
        print("Wait finished.")

        # 3. Save the screenshot
        full_screenshot_path = "screenshot_full.png"
        self.browser.save_screenshot(full_screenshot_path)
        print(f"Saved full screenshot to {full_screenshot_path}")
        
        # 4. Crop the screenshot
        cropped_output_path = "screenshot_cropped.png"
        center_crop(full_screenshot_path, cropped_output_path, crop_margin)
        print(f"Saved cropped screenshot to {cropped_output_path}")

        return Path(cropped_output_path)