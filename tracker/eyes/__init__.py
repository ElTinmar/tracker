from .core import *
from .tracker import *
from .overlay import *
from .widget import *

try:
    from .tracker_gpu import *
except:
    print('eyes::No GPU available')