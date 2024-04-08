import os
from enum import Enum


class Config(Enum):
    # Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__name__))
    SRC_DIR = os.path.join(ROOT_DIR, 'src')
    DATA_DIR = os.path.join(ROOT_DIR, 'utils')