from .core import *
from .tracker import *
from .overlay import *
from .widget import *
from .assignment import *
from .assignment_widget import *

try:
    from .tracker_gpu import *
except:
    print('animal::No GPU available')