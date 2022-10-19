from .version import __version__

# set simplified alias
from . import preprocessing as pp
from . import plot as pl
from . import analysis as al
from . import model

from .model import phasing as phasing
from .model import mixture as mixture
from .model import XClone_VB_model as XClone_VB_model
