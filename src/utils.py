"""
Utility functions for OMR evaluation system
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file {config_path} not found!")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {e}")
        return {}

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        
    Returns:
        Logger instance
    """
    # Create results directory if it doesn't exist
    os.makedirs("results/logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('results/logs/omr_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_image(image_path: str) -> bool:
    """
    Validate if image file is valid and readable
    
    Args:
        image_path: Path to image file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def convert_pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to OpenCV format
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        OpenCV image array
    """
    # Convert PIL to RGB if not already
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array
    opencv_image = np.array(pil_image)
    
    # Convert RGB to BGR for OpenCV
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    
    return opencv_image

def convert_cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert OpenCV image to PIL format
    
    Args:
        cv2_image: OpenCV image array
        
    Returns:
        PIL Image object
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL
    pil_image = Image.fromarray(rgb_image)
    
    return pil_image

def create_directories():
    """
    Create necessary directories for the application
    """
    directories = [
        "results",
        "results/processed_images", 
        "results/reports",
        "results/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def resize_image(image: np.ndarray, target_width: int = 800) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_width: Target width for resizing
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    aspect_ratio = height / width
    target_height = int(target_width * aspect_ratio)
    
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return resized

def calculate_confidence_score(bubble_filled_ratio: float, threshold: float = 0.3) -> float:
    """
    Calculate confidence score for bubble detection
    
    Args:
        bubble_filled_ratio: Ratio of filled pixels in bubble
        threshold: Threshold for considering bubble as filled
        
    Returns:
        Confidence score between 0 and 1
    """
    if bubble_filled_ratio >= threshold:
        # Filled bubble - confidence increases with fill ratio
        return min(1.0, bubble_filled_ratio / threshold)
    else:
        # Empty bubble - confidence decreases as fill ratio increases
        return max(0.0, 1.0 - (bubble_filled_ratio / threshold))

def get_subject_questions(question_num: int, questions_per_subject: int = 20) -> str:
    """
    Get subject name based on question number
    
    Args:
        question_num: Question number (1-100)
        questions_per_subject: Number of questions per subject
        
    Returns:
        Subject name
    """
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
    subject_index = (question_num - 1) // questions_per_subject
    return subjects[subject_index] if subject_index < len(subjects) else "Unknown"

def format_results_summary(results: Dict[str, Any]) -> str:
    """
    Format results for display
    
    Args:
        results: Evaluation results dictionary
        
    Returns:
        Formatted results string
    """
    summary = f"""
    📊 **OMR Evaluation Results**
    
    **Overall Score:** {results.get('total_score', 0)}/100
    **Accuracy:** {results.get('accuracy', 0):.1f}%
    
    **Subject-wise Scores:**
    """
    
    subject_scores = results.get('subject_scores', {})
    for subject, score in subject_scores.items():
        summary += f"  • **{subject}:** {score}/20\n"
    
    return summary

def save_results_to_excel(results: List[Dict], filename: str):
    """
    Save evaluation results to Excel file
    
    Args:
        results: List of result dictionaries
        filename: Output filename
    """
    try:
        df = pd.DataFrame(results)
        output_path = f"results/reports/{filename}"
        df.to_excel(output_path, index=False)
        return output_path
    except Exception as e:
        st.error(f"Error saving results to Excel: {e}")
        return None

def validate_answer_key_format(df: pd.DataFrame, config: Dict) -> bool:
    """
    Validate answer key DataFrame format
    
    Args:
        df: Answer key DataFrame
        config: Configuration dictionary
        
    Returns:
        True if valid format, False otherwise
    """
    required_columns = config['subjects']
    
    # Check if all required columns exist
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing column '{col}' in answer key")
            return False
    
    # Check if we have enough rows (questions)
    expected_questions = config['omr_settings']['questions_per_subject']
    if len(df) < expected_questions:
        st.error(f"Answer key should have at least {expected_questions} questions per subject")
        return False
        
    return True

def get_file_size_mb(file) -> float:
    """
    Get file size in MB
    
    Args:
        file: File object
        
    Returns:
        File size in MB
    """
    if hasattr(file, 'size'):
        return file.size / (1024 * 1024)
    return 0.0
