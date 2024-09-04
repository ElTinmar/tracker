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

# troubleshooting

There seems to be a problem with the version of Glib used by rapids and distributed on ubuntu
such that compiling opencv and using it with rapids is hard to pull off without 
triggering all kinds of errors
