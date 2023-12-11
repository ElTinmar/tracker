from distutils.core import setup

setup(
    name='tracker',
    author='Martin Privat',
    version='0.1.18',
    packages=['tracker','tracker.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='tracking zebrafish larvae',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "scikit-learn",
        "opencv-contrib-python-rolling @ https://github.com/ElTinmar/build_opencv/raw/main/opencv_contrib_python_rolling-4.8.0.20231210-cp38-cp38-linux_x86_64.whl",
        "PyQt5",
        "tqdm",
        "video_tools @ git+https://github.com/ElTinmar/video_tools.git@main",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@main",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main",
        "geometry @ git+https://github.com/ElTinmar/geometry.git@main"
    ],
    extras_require={
        'gpu': ["image_tools[gpu] @ git+https://github.com/ElTinmar/image_tools.git@main",]
    }
)