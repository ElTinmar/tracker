# tracker

Works best with background subtracted images

code to track zebrafish larvae:
    - centroid
    - orientation
    - eyes angle and vergence
    - tail skeleton

and associated qt widgets to set parameters

```
pip install git+https://github.com/ElTinmar/tracker.git@main
```

# coordinate system

- global space: global coordinate system
- input space: coordinate space of image given to tracker
- cropped space: tracker crops to this space (translation)
- resized space: tracker resizes to this space before tracking