import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# for reference: downsampling from 30ppm to 5ppm -> 6x
n_repeats = 1000
full_size = 2048
shrink_factors = np.logspace(0, 1, num=10)
shape_full = (full_size, full_size)

# Generate two random float32 images
img1 = np.random.rand(*shape_full).astype(np.float32)
img2 = np.random.rand(*shape_full).astype(np.float32)

# --- Benchmark full-size subtraction ---
start_full = time.perf_counter()
for _ in range(n_repeats):
    diff_full = img1 - img2
end_full = time.perf_counter()
avg_time_full = (end_full - start_full) / n_repeats

# --- Benchmark resize then subtract ---
resize_times = []
speedups = []

for factor in shrink_factors:
    new_size = int(full_size / factor)
    shape_resized = (new_size, new_size)

    print(f'{factor=}')

    start_resize = time.perf_counter()
    for _ in range(n_repeats):
        img2_small = cv2.resize(img2, shape_resized, interpolation=cv2.INTER_NEAREST)
        img1_small = cv2.resize(img1, shape_resized, interpolation=cv2.INTER_NEAREST)
        diff_small = img1_small - img2_small
    end_resize = time.perf_counter()
    avg_time_resize = (end_resize - start_resize) / n_repeats

    resize_times.append(avg_time_resize)
    speedups.append(avg_time_full / avg_time_resize)

# --- Report results ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(shrink_factors, np.array(resize_times) * 1e3, marker='o', label='Resize + Subtraction')
plt.axhline(avg_time_full * 1e3, color='r', linestyle='--', label='Full-size Subtraction')
plt.xlabel('Shrink Factor')
plt.ylabel('Average Time (ms)')
plt.title('Timing vs. Shrink Factor')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(shrink_factors, speedups, marker='s', color='green')
plt.axhline(1, color='r', linestyle='--', label='speedup = 1X')
plt.xlabel('Shrink Factor')
plt.ylabel('Speedup over Full-size Subtraction')
plt.title('Speedup vs. Shrink Factor')
plt.grid(True)

plt.tight_layout()
plt.show()