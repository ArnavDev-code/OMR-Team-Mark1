"""
OMR Evaluation System
Source code package for automated OMR sheet evaluation
"""

__version__ = "1.0.0"
__author__ = "OMR Hackathon Team"

# Import main modules
from .image_processing import OMRImageProcessor
from .answer_detection import OMRAnswerDetector
from .evaluation import OMREvaluator
from .utils import load_config, setup_logging

__all__ = [
    'OMRImageProcessor',
    'OMRAnswerDetector', 
    'OMREvaluator',
    'load_config',
    'setup_logging'
]