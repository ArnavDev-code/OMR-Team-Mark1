"""
OMR Image Processing Module
Handles image preprocessing, perspective correction, and OMR sheet detection
"""

import cv2
import numpy as np
from PIL import Image
import imutils
from skimage import measure
from typing import Tuple, List, Optional
import streamlit as st
from .utils import load_config, setup_logging, resize_image

class OMRImageProcessor:
    """
    Image processing class for OMR sheets
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the image processor
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging()
        self.processing_config = self.config.get('processing', {})
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for OMR detection
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Resize image if too large
            target_width = self.processing_config.get('image_resize_width', 800)
            if image.shape[1] > target_width:
                image = resize_image(image, target_width)
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blur_kernel = self.processing_config.get('gaussian_blur_kernel', 5)
            blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Remove noise using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            self.logger.info("Image preprocessing completed successfully")
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {e}")
            raise
    
    def detect_omr_sheet(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and extract OMR sheet from the image
        
        Args:
            image: Input image
            
        Returns:
            Cropped OMR sheet or None if not found
        """
        try:
            # Find contours
            contours = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Find the largest rectangular contour (likely the OMR sheet)
            for contour in contours:
                # Approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                # If we have a 4-point contour with sufficient area, it's likely the OMR sheet
                if len(approx) == 4 and cv2.contourArea(contour) > 10000:
                    return approx.reshape(4, 2)
            
            # If no perfect rectangle found, use the largest contour
            if contours:
                largest_contour = contours[0]
                if cv2.contourArea(largest_contour) > 5000:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
            
            self.logger.warning("Could not detect OMR sheet boundaries")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in OMR sheet detection: {e}")
            return None
    
    def correct_perspective(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Correct perspective distortion using four corner points
        
        Args:
            image: Input image
            corners: Four corner points of the OMR sheet
            
        Returns:
            Perspective-corrected image
        """
        try:
            # Order the corners: top-left, top-right, bottom-right, bottom-left
            def order_points(pts):
                rect = np.zeros((4, 2), dtype=np.float32)
                
                # Sum and difference to find corners
                s = pts.sum(axis=1)
                diff = np.diff(pts, axis=1)
                
                rect[0] = pts[np.argmin(s)]      # top-left
                rect[2] = pts[np.argmax(s)]      # bottom-right
                rect[1] = pts[np.argmin(diff)]   # top-right
                rect[3] = pts[np.argmax(diff)]   # bottom-left
                
                return rect
            
            rect = order_points(corners)
            (tl, tr, br, bl) = rect
            
            # Calculate dimensions of the new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            # Define destination points
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]
            ], dtype=np.float32)
            
            # Calculate perspective transformation matrix
            matrix = cv2.getPerspectiveTransform(rect, dst)
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
            
            self.logger.info("Perspective correction completed")
            return warped
            
        except Exception as e:
            self.logger.error(f"Error in perspective correction: {e}")
            return image
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast and remove noise
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Remove noise
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Sharpen the image
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"Error in image enhancement: {e}")
            return image
    
    def detect_and_correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew in the image
        
        Args:
            image: Input image
            
        Returns:
            Skew-corrected image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply threshold
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Find contours
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            
            # Find the largest contour (document boundary)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get minimum area rectangle
                rect = cv2.minAreaRect(largest_contour)
                angle = rect[2]
                
                # Correct angle
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                # Rotate image if skew is significant
                if abs(angle) > 0.5:
                    (h, w) = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # Calculate new dimensions
                    cos = np.abs(M[0, 0])
                    sin = np.abs(M[0, 1])
                    nW = int((h * sin) + (w * cos))
                    nH = int((h * cos) + (w * sin))
                    
                    # Adjust rotation matrix
                    M[0, 2] += (nW / 2) - center[0]
                    M[1, 2] += (nH / 2) - center[1]
                    
                    # Perform rotation
                    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    
                    self.logger.info(f"Corrected skew angle: {angle:.2f} degrees")
                    return rotated
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error in skew correction: {e}")
            return image
    
    def process_omr_image(self, image: np.ndarray, apply_perspective_correction: bool = True) -> np.ndarray:
        """
        Complete processing pipeline for OMR image
        
        Args:
            image: Input image
            apply_perspective_correction: Whether to apply perspective correction
            
        Returns:
            Fully processed OMR image
        """
        try:
            # Step 1: Initial preprocessing
            processed = self.preprocess_image(image)
            
            # Step 2: Detect and correct skew
            processed = self.detect_and_correct_skew(processed)
            
            # Step 3: Detect OMR sheet boundaries
            if apply_perspective_correction:
                corners = self.detect_omr_sheet(processed)
                if corners is not None:
                    # Apply perspective correction
                    original_color = image.copy()
                    processed = self.correct_perspective(original_color, corners)
                    
                    # Reprocess after perspective correction
                    processed = self.preprocess_image(processed)
            
            # Step 4: Final enhancement
            processed = self.enhance_image(processed)
            
            self.logger.info("OMR image processing completed successfully")
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in OMR image processing: {e}")
            raise
    
    def create_debug_image(self, original: np.ndarray, processed: np.ndarray, 
                          corners: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create debug image showing processing steps
        
        Args:
            original: Original image
            processed: Processed image
            corners: Detected corners (if any)
            
        Returns:
            Debug image with annotations
        """
        try:
            # Create side-by-side comparison
            if len(original.shape) == 3:
                original_display = original.copy()
            else:
                original_display = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            
            if len(processed.shape) == 3:
                processed_display = processed.copy()
            else:
                processed_display = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            
            # Draw corners if detected
            if corners is not None:
                cv2.drawContours(original_display, [corners.astype(int)], -1, (0, 255, 0), 3)
            
            # Resize images to same height for concatenation
            h1, w1 = original_display.shape[:2]
            h2, w2 = processed_display.shape[:2]
            
            target_height = min(h1, h2, 400)  # Limit height for display
            
            aspect1 = w1 / h1
            aspect2 = w2 / h2
            
            new_w1 = int(target_height * aspect1)
            new_w2 = int(target_height * aspect2)
            
            original_resized = cv2.resize(original_display, (new_w1, target_height))
            processed_resized = cv2.resize(processed_display, (new_w2, target_height))
            
            # Concatenate images
            debug_image = np.hstack([original_resized, processed_resized])
            
            # Add text labels
            cv2.putText(debug_image, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(debug_image, "Processed", (new_w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return debug_image
            
        except Exception as e:
            self.logger.error(f"Error creating debug image: {e}")
            return original if len(original.shape) == 3 else cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    def validate_processed_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate if processed image is suitable for answer detection
        
        Args:
            image: Processed image
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if image is None or image.size == 0:
                return False, "Image is empty"
            
            # Check minimum size
            min_height, min_width = 50 , 50
            if image.shape[0] < min_height or image.shape[1] < min_width:
                return False, f"Image too small. Minimum size: {min_width}x{min_height}"
            
            # Check if image has sufficient contrast
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            contrast = gray.std()
            if contrast < 20:
                return False, "Image has insufficient contrast"
            
            # Check for content (not just blank)
            binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            content_ratio = np.sum(binary == 0) / binary.size
            
            if content_ratio < 0.05:  # Less than 5% content
                return False, "Image appears to be mostly blank"
            
            if content_ratio > 0.95:  # More than 95% content
                return False, "Image appears to be too dark or corrupted"
            
            return True, "Image validation passed"
            
        except Exception as e:
            return False, f"Error during validation: {str(e)}"