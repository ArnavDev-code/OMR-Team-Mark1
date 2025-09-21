"""
Basic system tests for OMR evaluation system
"""

import unittest
import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_processing import OMRImageProcessor
from answer_detection import OMRAnswerDetector
from evaluation import OMREvaluator
from utils import load_config

class TestOMRSystem(unittest.TestCase):
    """Test cases for OMR system components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = load_config()
        self.image_processor = OMRImageProcessor()
        self.answer_detector = OMRAnswerDetector()
        self.evaluator = OMREvaluator()
        
        # Create a simple test image
        self.test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsInstance(self.config, dict)
        self.assertIn('omr_settings', self.config)
        self.assertIn('subjects', self.config)
        
    def test_image_processor_initialization(self):
        """Test image processor initialization"""
        self.assertIsNotNone(self.image_processor)
        self.assertIsInstance(self.image_processor.config, dict)
        
    def test_answer_detector_initialization(self):
        """Test answer detector initialization"""
        self.assertIsNotNone(self.answer_detector)
        self.assertIsInstance(self.answer_detector.config, dict)
        
    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        self.assertIsNotNone(self.evaluator)
        self.assertIsInstance(self.evaluator.subjects, list)
        
    def test_image_preprocessing(self):
        """Test basic image preprocessing"""
        try:
            processed = self.image_processor.preprocess_image(self.test_image)
            self.assertIsNotNone(processed)
            self.assertEqual(len(processed.shape), 2)  # Should be grayscale
        except Exception as e:
            self.fail(f"Image preprocessing failed: {e}")
            
    def test_answer_validation(self):
        """Test answer validation"""
        # Valid answers
        valid_answers = ['a', 'b', 'c', 'd'] * 25  # 100 answers
        is_valid, issues = self.evaluator.validate_detected_answers(valid_answers)
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Invalid answers
        invalid_answers = ['x', 'y'] * 50  # 100 invalid answers
        is_valid, issues = self.evaluator.validate_detected_answers(invalid_answers)
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
        
    def test_sample_images_exist(self):
        """Test if sample images directory exists"""
        sample_dirs = ['samples/Set A', 'samples/Set B']
        
        for sample_dir in sample_dirs:
            if os.path.exists(sample_dir):
                files = os.listdir(sample_dir)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                self.assertGreater(len(image_files), 0, f"No image files found in {sample_dir}")
                
    def test_answer_key_file_exists(self):
        """Test if answer key file exists"""
        answer_key_path = "Key (Set A and B).xlsx"
        if os.path.exists(answer_key_path):
            # Try to load the answer key
            success = self.evaluator.load_answer_key(answer_key_path)
            self.assertTrue(success, "Failed to load answer key file")
        else:
            self.skipTest("Answer key file not found - this is expected during development")

if __name__ == '__main__':
    # Create test directories if they don't exist
    test_dirs = ['results', 'results/processed_images', 'results/reports', 'results/logs']
    for test_dir in test_dirs:
        os.makedirs(test_dir, exist_ok=True)
    
    # Run tests
    unittest.main()
