import metavision_sdk_driver
from metavision_sdk_driver import Camera

try:
    cam = Camera.from_first_available()
    print(f"✅ USPJEH! Kamera je spremna. Geometrija: {cam.geometry().width}x{cam.geometry().height}")
    cam.start()
    cam.stop()
except Exception as e:
    print(f"❌ SDK je učitan, ali kamera nije dostupna: {e}")