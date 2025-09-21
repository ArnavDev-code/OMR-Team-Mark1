"""
OMR Answer Detection Module
Handles bubble detection, marking analysis, and answer extraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import streamlit as st
from .utils import load_config, setup_logging, calculate_confidence_score

class OMRAnswerDetector:
    """
    Answer detection class for OMR sheets
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the answer detector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging()
        self.processing_config = self.config.get('processing', {})
        self.omr_config = self.config.get('omr_settings', {})
        
        # Bubble detection parameters
        self.min_bubble_area = self.processing_config.get('min_bubble_area', 50)
        self.max_bubble_area = self.processing_config.get('max_bubble_area', 500)
        self.bubble_threshold = 0.3  # Threshold for considering bubble as filled
        
    def detect_bubbles(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all bubbles in the OMR sheet
        
        Args:
            image: Processed OMR image
            
        Returns:
            List of bubble information dictionaries
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply threshold to create binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bubbles = []
            
            for i, contour in enumerate(contours):
                # Calculate contour properties
                area = cv2.contourArea(contour)
                
                # Filter by area
                if self.min_bubble_area <= area <= self.max_bubble_area:
                    # Calculate aspect ratio to filter circular shapes
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    
                    # Check if it's roughly circular (aspect ratio close to 1)
                    if 0.7 <= aspect_ratio <= 1.3:
                        # Calculate center and other properties
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            bubble_info = {
                                'contour': contour,
                                'center': (cx, cy),
                                'area': area,
                                'bounding_rect': (x, y, w, h),
                                'aspect_ratio': aspect_ratio
                            }
                            
                            bubbles.append(bubble_info)
            
            # Sort bubbles by position (top to bottom, left to right)
            bubbles.sort(key=lambda b: (b['center'][1], b['center'][0]))
            
            self.logger.info(f"Detected {len(bubbles)} potential bubbles")
            return bubbles
            
        except Exception as e:
            self.logger.error(f"Error in bubble detection: {e}")
            return []
    
    def analyze_bubble_filling(self, image: np.ndarray, bubble_info: Dict) -> Tuple[bool, float]:
        """
        Analyze if a bubble is filled or not
        
        Args:
            image: Original image
            bubble_info: Bubble information dictionary
            
        Returns:
            Tuple of (is_filled, confidence_score)
        """
        try:
            # Extract bubble region
            x, y, w, h = bubble_info['bounding_rect']
            bubble_roi = image[y:y+h, x:x+w]
            
            if bubble_roi.size == 0:
                return False, 0.0
            
            # Convert to grayscale if needed
            if len(bubble_roi.shape) == 3:
                gray_roi = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = bubble_roi.copy()
            
            # Apply threshold
            _, binary_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Create a mask for the bubble (circular region)
            mask = np.zeros(binary_roi.shape, dtype=np.uint8)
            center = (w // 2, h // 2)
            radius = min(w, h) // 2 - 2  # Slightly smaller radius to avoid edges
            cv2.circle(mask, center, radius, 255, -1)
            
            # Apply mask to get only the bubble interior
            masked_roi = cv2.bitwise_and(binary_roi, mask)
            
            # Calculate fill ratio
            total_pixels = cv2.countNonZero(mask)
            filled_pixels = cv2.countNonZero(masked_roi)
            
            if total_pixels == 0:
                return False, 0.0
            
            fill_ratio = filled_pixels / total_pixels
            
            # Determine if bubble is filled
            is_filled = fill_ratio >= self.bubble_threshold
            confidence = calculate_confidence_score(fill_ratio, self.bubble_threshold)
            
            return is_filled, confidence
            
        except Exception as e:
            self.logger.error(f"Error analyzing bubble filling: {e}")
            return False, 0.0
    
    def group_bubbles_by_question(self, bubbles: List[Dict]) -> List[List[Dict]]:
        """
        Group bubbles by question rows
        
        Args:
            bubbles: List of detected bubbles
            
        Returns:
            List of question groups, each containing 4 bubbles (a, b, c, d)
        """
        try:
            if not bubbles:
                return []
            
            # Group bubbles by vertical position (questions in rows)
            question_groups = []
            current_group = []
            
            # Sort bubbles by Y coordinate first
            sorted_bubbles = sorted(bubbles, key=lambda b: b['center'][1])
            
            tolerance = 20  # Pixel tolerance for same row
            current_y = sorted_bubbles[0]['center'][1]
            
            for bubble in sorted_bubbles:
                bubble_y = bubble['center'][1]
                
                # If bubble is in the same row
                if abs(bubble_y - current_y) <= tolerance:
                    current_group.append(bubble)
                else:
                    # Sort current group by X coordinate (left to right)
                    if current_group:
                        current_group.sort(key=lambda b: b['center'][0])
                        if len(current_group) >= 2:  # At least 2 options per question
                            question_groups.append(current_group)
                    
                    # Start new group
                    current_group = [bubble]
                    current_y = bubble_y
            
            # Don't forget the last group
            if current_group:
                current_group.sort(key=lambda b: b['center'][0])
                if len(current_group) >= 2:
                    question_groups.append(current_group)
            
            # Filter groups to have exactly 4 bubbles (a, b, c, d)
            filtered_groups = []
            for group in question_groups:
                if len(group) == 4:
                    filtered_groups.append(group)
                elif len(group) > 4:
                    # Take the first 4 bubbles if more than 4
                    filtered_groups.append(group[:4])
            
            self.logger.info(f"Grouped bubbles into {len(filtered_groups)} questions")
            return filtered_groups
            
        except Exception as e:
            self.logger.error(f"Error grouping bubbles: {e}")
            return []
    
    def extract_answers(self, image: np.ndarray) -> List[str]:
        """
        Extract answers from OMR sheet
        
        Args:
            image: Processed OMR image
            
        Returns:
            List of detected answers (a, b, c, d, or empty string for no answer)
        """
        try:
            # Detect all bubbles
            bubbles = self.detect_bubbles(image)
            
            if not bubbles:
                self.logger.warning("No bubbles detected in image")
                return [''] * self.omr_config.get('total_questions', 100)
            
            # Group bubbles by questions
            question_groups = self.group_bubbles_by_question(bubbles)
            
            answers = []
            answer_options = ['a', 'b', 'c', 'd']
            
            for group in question_groups:
                question_answer = ''
                max_confidence = 0.0
                best_option = ''
                
                # Analyze each bubble in the group
                for i, bubble in enumerate(group):
                    if i < len(answer_options):
                        is_filled, confidence = self.analyze_bubble_filling(image, bubble)
                        
                        if is_filled and confidence > max_confidence:
                            max_confidence = confidence
                            best_option = answer_options[i]
                
                # Set answer based on best confidence
                if max_confidence > 0.5:  # Minimum confidence threshold
                    question_answer = best_option
                
                answers.append(question_answer)
            
            # Pad with empty answers if we have fewer than expected questions
            total_questions = self.omr_config.get('total_questions', 100)
            while len(answers) < total_questions:
                answers.append('')
            
            # Truncate if we have more answers than expected
            answers = answers[:total_questions]
            
            self.logger.info(f"Extracted {len([a for a in answers if a])} answers out of {len(answers)} questions")
            return answers
            
        except Exception as e:
            self.logger.error(f"Error extracting answers: {e}")
            return [''] * self.omr_config.get('total_questions', 100)
    
    def create_detection_visualization(self, image: np.ndarray, bubbles: List[Dict], 
                                     answers: List[str]) -> np.ndarray:
        """
        Create visualization showing detected bubbles and answers
        
        Args:
            image: Original image
            bubbles: List of detected bubbles
            answers: List of detected answers
            
        Returns:
            Annotated image
        """
        try:
            # Create a copy for visualization
            if len(image.shape) == 3:
                vis_image = image.copy()
            else:
                vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Group bubbles by questions
            question_groups = self.group_bubbles_by_question(bubbles)
            answer_options = ['a', 'b', 'c', 'd']
            
            for q_idx, group in enumerate(question_groups):
                if q_idx < len(answers):
                    selected_answer = answers[q_idx]
                    
                    for i, bubble in enumerate(group):
                        if i < len(answer_options):
                            option = answer_options[i]
                            center = bubble['center']
                            
                            # Determine color based on selection
                            if option == selected_answer and selected_answer != '':
                                # Selected answer - green
                                color = (0, 255, 0)
                                thickness = 3
                            else:
                                # Non-selected - blue
                                color = (255, 0, 0)
                                thickness = 2
                            
                            # Draw circle around bubble
                            cv2.circle(vis_image, center, 15, color, thickness)
                            
                            # Add option label
                            label_pos = (center[0] - 5, center[1] - 20)
                            cv2.putText(vis_image, option.upper(), label_pos, 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Add question number
                    if group:
                        first_bubble_center = group[0]['center']
                        q_label_pos = (first_bubble_center[0] - 40, first_bubble_center[1] + 5)
                        cv2.putText(vis_image, f"Q{q_idx + 1}", q_label_pos,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            return vis_image
            
        except Exception as e:
            self.logger.error(f"Error creating detection visualization: {e}")
            return image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    def validate_bubble_detection(self, bubbles: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Validate bubble detection results
        
        Args:
            bubbles: List of detected bubbles
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if we have enough bubbles
        expected_bubbles = self.omr_config.get('total_questions', 100) * 4  # 4 options per question
        if len(bubbles) < expected_bubbles * 0.8:  # Allow 20% tolerance
            issues.append(f"Too few bubbles detected: {len(bubbles)} (expected ~{expected_bubbles})")
        
        # Group bubbles and check consistency
        question_groups = self.group_bubbles_by_question(bubbles)
        if len(question_groups) < self.omr_config.get('total_questions', 100) * 0.8:
            issues.append(f"Too few question groups: {len(question_groups)}")
        
        # Check for inconsistent group sizes
        group_sizes = [len(group) for group in question_groups]
        if group_sizes:
            most_common_size = max(set(group_sizes), key=group_sizes.count)
            inconsistent_groups = sum(1 for size in group_sizes if size != most_common_size)
            
            if inconsistent_groups > len(question_groups) * 0.2:  # More than 20% inconsistent
                issues.append(f"Inconsistent bubble grouping detected in {inconsistent_groups} questions")
        
        return len(issues) == 0, issues
    
    def get_detection_statistics(self, image: np.ndarray, answers: List[str]) -> Dict[str, any]:
        """
        Get statistics about the detection process
        
        Args:
            image: Processed image
            answers: Detected answers
            
        Returns:
            Dictionary with detection statistics
        """
        try:
            bubbles = self.detect_bubbles(image)
            question_groups = self.group_bubbles_by_question(bubbles)
            
            # Count filled vs empty answers
            filled_answers = len([a for a in answers if a != ''])
            empty_answers = len(answers) - filled_answers
            
            # Calculate answer distribution
            answer_distribution = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'empty': empty_answers}
            for answer in answers:
                if answer in answer_distribution:
                    answer_distribution[answer] += 1
            
            stats = {
                'total_bubbles_detected': len(bubbles),
                'question_groups_formed': len(question_groups),
                'total_questions': len(answers),
                'filled_answers': filled_answers,
                'empty_answers': empty_answers,
                'answer_distribution': answer_distribution,
                'detection_success_rate': (filled_answers / len(answers)) * 100 if answers else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting detection statistics: {e}")
            return {}
    
    def detect_multiple_markings(self, image: np.ndarray) -> List[int]:
        """
        Detect questions with multiple markings
        
        Args:
            image: Processed image
            
        Returns:
            List of question numbers with multiple markings
        """
        try:
            bubbles = self.detect_bubbles(image)
            question_groups = self.group_bubbles_by_question(bubbles)
            
            multiple_markings = []
            
            for q_idx, group in enumerate(question_groups):
                filled_count = 0
                
                for bubble in group:
                    is_filled, confidence = self.analyze_bubble_filling(image, bubble)
                    if is_filled and confidence > 0.5:
                        filled_count += 1
                
                if filled_count > 1:
                    multiple_markings.append(q_idx + 1)  # 1-indexed question numbers
            
            if multiple_markings:
                self.logger.warning(f"Multiple markings detected in questions: {multiple_markings}")
            
            return multiple_markings
            
        except Exception as e:
            self.logger.error(f"Error detecting multiple markings: {e}")
            return []
    
    def enhance_bubble_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Apply specific enhancements for better bubble detection
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image for bubble detection
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply morphological opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(opened, (3, 3), 0)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing image for bubble detection: {e}")
            return image