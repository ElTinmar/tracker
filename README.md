# tracker

Works best with background subtracted images

code to track zebrafish larvae:
    - centroid
    - orientation
    - eyes angle and vergence
    - tail skeleton

and associated qt widgets to set parameters

TODO create GPU class to do the tracking on GPU

```
git clone  
conda env create -f tracker_GPU.yml
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
conda env create -f tracker_GPU.yml
```

```
pip install git+https://github.com/ElTinmar/tracker.git@main
```

