import os
from enum import Enum

class Config(Enum):
    ROOT_DIR = os.path.dirname(os.path.abspath(__name__))
    SRC_DIR = os.path.join(ROOT_DIR, 'hand_filter')
    