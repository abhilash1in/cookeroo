# -*- coding: utf-8 -*-
from .data_prep import DataPrep
from .model import CookerooModel

from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __author__, __author_email__, __copyright__
from .exceptions import InvalidDirectoryStructureException

# Set default logging handler to avoid "No handler found" warnings.
import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
