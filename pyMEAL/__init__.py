__authors__ = 'Adeleke Maradesa, Abdulmojeed Ilyas'

__date__ = '21st May, 2025'

import os
print(f"Initializing MEAL from {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")

from . import basics
from . import builder_block
from . import encoder_concatenation
from . import traditional_augmentation
from . import without_augmentation
from . import fusion_layer
from . import plotting
from . import utils



modules = [
    "basics", 
    "builder_block", 
    "encoder_concatenation", 
    "traditional_augmentation", 
    "without_augmentation",
    "fusion_layer",
    "plotting",
    "utils",
]

for module in modules:
    try:
        __import__(f"pyMEAL.{module}", fromlist=[''])
        print(f"Imported {module}")
    except ImportError as e:
        print(f"Failed to import {module}: {e}")



__all__ = [
    'basics',
    'builder_block',
    'encoder_concatenation',
    'traditional_augmentation',
    'without_augmentation',
    'fusion_layer',
    'plotting',
    'utils'

]

print(f"Contents of pyMEAL package: {__all__}")