import numpy as np
from scipy.optimize import minimize_scalar
import requests
from PIL import Image
from io import BytesIO


def mapping2d_float(h, l=256):
    cum_sum = 0
    t = np.zeros(l, dtype=np.float32)
    for i in range(l):
        cum_sum += h[i]
        t[i] = (l - 1) * cum_sum
    return t


def f2d_float(lam, h_i, h_u):
    h_tilde = 1 / (1 + lam) * h_i + lam / (1 + lam) * h_u
    t = mapping2d_float(h_tilde)
    d = 0
    for i in range(len(h_tilde)):
        for j in range(i + 1):
            if h_tilde[i] > 0 and h_tilde[j] > 0 and abs(t[i] - t[j]) < 1e-6:
                d = max(d, i - j)
    return d


def apply_glcae_2d_float(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 1))
    h_i = hist / image.size
    h_u = np.ones_like(h_i) / len(h_i)

    result = minimize_scalar(f2d_float, method="brent", args=(h_i, h_u))
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping2d_float(h_tilde)

    bins = np.linspace(0, 1, 256)
    enhanced = np.interp(image, bins, t / 255.0)
    return enhanced


url = "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes/20241024131838/06508.tif"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img_array = np.array(img, dtype=np.float32)
img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

enhanced = apply_glcae_2d_float(img_array)

for i in range(21):
    threshold = i * 0.05
    thresholded = np.where(enhanced >= threshold, enhanced, 0)
    output = Image.fromarray((thresholded * 255).astype(np.uint8))
    output.save(f'glcae_threshold_{threshold:.2f}.tif')