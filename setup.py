from distutils.core import setup

setup(
    name='tracker',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.14.1',
    packages=['tracker','tracker.animal','tracker.body','tracker.eyes','tracker.multifish','tracker.tail', 'tracker.singlefish', 'tracker.head_embedded'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='tracking zebrafish larvae',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "opencv-python-headless",
        "tqdm",
        "numba",
        "filterpy",
        "video_tools @ git+https://github.com/ElTinmar/video_tools.git@v0.6.8",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@v0.9.4",
        "geometry @ git+https://github.com/ElTinmar/geometry.git@v0.3.0",
        "kalman @ git+https://github.com/ElTinmar/Kalman.git@v0.1.7"
    ]
)
