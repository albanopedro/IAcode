"""Draw the JARVIS icon (1024 px): a glowing cyan orb with rings on a dark rounded square."""
import struct, sys, zlib
import numpy as np

N = 1024
y, x = np.mgrid[0:N, 0:N].astype(np.float32) + 0.5
c = N / 2
r = np.hypot(x - c, y - c) / (N / 2)  # 0 at the centre, 1 at the edge

# Rounded square (macOS icon grid: ~824 px body with ~185 px corner radius), anti-aliased.
half, rad = 412, 185
qx, qy = np.abs(x - c) - (half - rad), np.abs(y - c) - (half - rad)
dist = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - rad
alpha = np.clip(0.5 - dist, 0, 1)

# Background: deep navy with a soft vertical gradient.
t = (y / N)[..., None]
rgb = (1 - t) * np.array([0.05, 0.09, 0.17]) + t * np.array([0.01, 0.03, 0.08])

def add(color, weight):
    global rgb
    rgb = rgb + np.asarray(color) * weight[..., None]

add([0.10, 0.55, 0.85], np.exp(-((r / 0.62) ** 2)) * 0.55)          # outer glow
for ring_r, width, strength in [(0.70, 0.010, 0.55), (0.58, 0.006, 0.35)]:  # rings
    add([0.35, 0.85, 1.0], np.exp(-(((r - ring_r) / width) ** 2)) * strength)
sphere = np.clip((0.36 - r) / 0.012, 0, 1)                            # crisp orb edge
add([0.05, 0.45, 0.75], sphere * 0.6)                                  # orb body
add([0.30, 0.90, 1.0], np.clip(1 - r / 0.36, 0, 1) ** 1.2 * sphere * 0.9)  # inner light
add([0.45, 0.95, 1.0], np.exp(-(((r - 0.36) / 0.012) ** 2)) * 0.8)     # bright rim
add([1.0, 1.0, 1.0], np.exp(-((r / 0.12) ** 2)) * 0.85)                # hot centre
rgba = np.dstack([np.clip(rgb, 0, 1), alpha])
data = (rgba * 255 + 0.5).astype(np.uint8)
raw = b"".join(b"\x00" + row.tobytes() for row in data)
def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", N, N, 8, 6, 0, 0, 0)) \
    + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
open(sys.argv[1], "wb").write(png)
