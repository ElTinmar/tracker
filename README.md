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
conda install -c conda-forge mamba
mamba env create -f tracker_GPU.yml
```

```
pip install git+https://github.com/ElTinmar/tracker.git@main
```

