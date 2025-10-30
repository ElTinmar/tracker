from distutils.core import setup

setup(
    name='tracker',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.12.3',
    packages=['tracker','tracker.animal','tracker.body','tracker.eyes','tracker.multifish','tracker.tail', 'tracker.singlefish'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='tracking zebrafish larvae',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "opencv-python-headless",
        "PyQt5",
        "tqdm",
        "numba",
        "filterpy",
        "video_tools @ git+https://github.com/ElTinmar/video_tools.git@main",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@main",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main",
        "geometry @ git+https://github.com/ElTinmar/geometry.git@main",
        "kalman @ git+https://github.com/ElTinmar/Kalman.git@main"
    ]
)
