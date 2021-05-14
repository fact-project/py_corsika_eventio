from .base import EventIOFile, KNOWN_OBJECTS
from . import iact
from .iact import IACTFile
from . import simtel
from .simtel import SimTelFile
from .histograms import Histograms


__version__ = '2.0.0'


__all__ = [
    'EventIOFile',
    'IACTFile',
    'SimTelFile',
    'simtel',
    'iact',
]


KNOWN_OBJECTS[Histograms.eventio_type] = Histograms