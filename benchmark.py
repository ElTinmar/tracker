import numpy as np
import cv2
import time

# downsampling from 30ppm to 5ppm -> 6x
full_size = 2048
new_size = int(full_size/6)

shape_full = (full_size, full_size)
shape_resized = (new_size, new_size)
n_repeats = 100

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
#img1_small = cv2.resize(img1, shape_resized, interpolation=cv2.INTER_NEAREST)

start_resize = time.perf_counter()
for _ in range(n_repeats):
    img2_small = cv2.resize(img2, shape_resized, interpolation=cv2.INTER_NEAREST)
    img1_small = cv2.resize(img1, shape_resized, interpolation=cv2.INTER_NEAREST)
    diff_small = img1_small - img2_small
end_resize = time.perf_counter()
avg_time_resize = (end_resize - start_resize) / n_repeats

# --- Report results ---
print(f"Average full-size subtraction time: {avg_time_full * 1e3:.3f} ms")
print(f"Average resize + subtraction time:  {avg_time_resize * 1e3:.3f} ms")
print(f"speedup: {avg_time_full/avg_time_resize}X")

# I got ~30X speedup 