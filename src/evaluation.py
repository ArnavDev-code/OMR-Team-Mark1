"""
OMR Evaluation Engine
Handles answer key loading, scoring, and result generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import streamlit as st
from .utils import load_config, setup_logging, validate_answer_key_format

class OMREvaluator:
    """
    Main evaluation class for OMR answer checking and scoring
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the evaluator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logging()
        self.answer_keys = {}
        self.subjects = self.config.get('subjects', [])
        self.questions_per_subject = self.config['omr_settings']['questions_per_subject']
        
    def load_answer_key(self, file_path: str) -> bool:
        """
        Load answer key from Excel file
        
        Args:
            file_path: Path to Excel file containing answer keys
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Read Excel file
            df = pd.read_excel(file_path)
            
            # Validate format
            if not validate_answer_key_format(df, self.config):
                return False
            
            # Process answer keys for each subject
            for subject in self.subjects:
                if subject in df.columns:
                    # Extract answers and convert to lowercase
                    answers = df[subject].head(self.questions_per_subject).astype(str).str.lower().str.strip()
                    self.answer_keys[subject] = answers.tolist()
                    
            self.logger.info(f"Successfully loaded answer keys for subjects: {list(self.answer_keys.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading answer key: {e}")
            st.error(f"Error loading answer key: {e}")
            return False
    
    def load_answer_key_from_uploaded_file(self, uploaded_file) -> bool:
        """
        Load answer key from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Read Excel file from uploaded file
            df = pd.read_excel(uploaded_file)
            
            # Validate format
            if not validate_answer_key_format(df, self.config):
                return False
            
            # Process answer keys for each subject
            for subject in self.subjects:
                if subject in df.columns:
                    # Extract answers and convert to lowercase, handle question numbering
                    subject_data = df[subject].head(self.questions_per_subject)
                    answers = []
                    
                    for answer in subject_data:
                        if pd.isna(answer):
                            answers.append('')
                        else:
                            # Extract letter after dash (e.g., "1 - a" -> "a")
                            answer_str = str(answer).lower().strip()
                            if ' - ' in answer_str:
                                letter = answer_str.split(' - ')[-1].strip()
                                answers.append(letter)
                            else:
                                # Handle formats like "41 - c" or just "c"
                                parts = answer_str.split()
                                if len(parts) >= 3 and parts[1] == '-':
                                    answers.append(parts[2])
                                elif len(parts) == 1 and parts[0] in ['a', 'b', 'c', 'd']:
                                    answers.append(parts[0])
                                else:
                                    answers.append('')
                    
                    self.answer_keys[subject] = answers
                    
            self.logger.info(f"Successfully loaded answer keys for subjects: {list(self.answer_keys.keys())}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading answer key from uploaded file: {e}")
            st.error(f"Error loading answer key: {e}")
            return False
    
    def evaluate_omr_sheet(self, detected_answers: List[str], sheet_set: str = 'A') -> Dict[str, Any]:
        """
        Evaluate a single OMR sheet
        
        Args:
            detected_answers: List of detected answers (100 answers)
            sheet_set: Sheet set identifier ('A' or 'B')
            
        Returns:
            Dictionary containing evaluation results
        """
        if not self.answer_keys:
            raise ValueError("No answer keys loaded. Please load answer keys first.")
        
        if len(detected_answers) != self.config['omr_settings']['total_questions']:
            raise ValueError(f"Expected {self.config['omr_settings']['total_questions']} answers, got {len(detected_answers)}")
        
        results = {
            'sheet_set': sheet_set,
            'total_score': 0,
            'subject_scores': {},
            'correct_answers': 0,
            'wrong_answers': 0,
            'unanswered': 0,
            'accuracy': 0.0,
            'detailed_results': [],
            'subject_accuracy': {}
        }
        
        question_index = 0
        
        # Evaluate each subject
        for subject_idx, subject in enumerate(self.subjects):
            subject_correct = 0
            subject_total = self.questions_per_subject
            subject_answers = []
            
            # Get answer key for this subject
            if subject not in self.answer_keys:
                self.logger.warning(f"No answer key found for subject: {subject}")
                continue
                
            correct_answers = self.answer_keys[subject]
            
            # Check answers for this subject
            for q in range(self.questions_per_subject):
                if question_index >= len(detected_answers):
                    break
                    
                student_answer = detected_answers[question_index].lower().strip()
                correct_answer = correct_answers[q] if q < len(correct_answers) else ''
                
                is_correct = student_answer == correct_answer and student_answer != ''
                
                question_result = {
                    'question_number': question_index + 1,
                    'subject': subject,
                    'student_answer': student_answer,
                    'correct_answer': correct_answer,
                    'is_correct': is_correct,
                    'is_unanswered': student_answer == ''
                }
                
                results['detailed_results'].append(question_result)
                subject_answers.append(question_result)
                
                if is_correct:
                    subject_correct += 1
                    results['correct_answers'] += 1
                elif student_answer == '':
                    results['unanswered'] += 1
                else:
                    results['wrong_answers'] += 1
                
                question_index += 1
            
            # Calculate subject score and accuracy
            results['subject_scores'][subject] = subject_correct
            if subject_total > 0:
                subject_accuracy = (subject_correct / subject_total) * 100
                results['subject_accuracy'][subject] = subject_accuracy
        
        # Calculate overall metrics
        results['total_score'] = results['correct_answers']
        total_questions = self.config['omr_settings']['total_questions']
        results['accuracy'] = (results['correct_answers'] / total_questions) * 100
        
        self.logger.info(f"Evaluation completed. Score: {results['total_score']}/{total_questions}")
        
        return results
    
    def batch_evaluate(self, batch_results: List[Tuple[str, List[str]]], sheet_set: str = 'A') -> List[Dict[str, Any]]:
        """
        Evaluate multiple OMR sheets
        
        Args:
            batch_results: List of tuples (filename, detected_answers)
            sheet_set: Sheet set identifier
            
        Returns:
            List of evaluation results
        """
        all_results = []
        
        for filename, detected_answers in batch_results:
            try:
                result = self.evaluate_omr_sheet(detected_answers, sheet_set)
                result['filename'] = filename
                all_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error evaluating {filename}: {e}")
                error_result = {
                    'filename': filename,
                    'error': str(e),
                    'total_score': 0,
                    'accuracy': 0.0
                }
                all_results.append(error_result)
        
        return all_results
    
    def generate_summary_report(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary report for batch evaluation
        
        Args:
            all_results: List of evaluation results
            
        Returns:
            Summary statistics
        """
        if not all_results:
            return {}
        
        valid_results = [r for r in all_results if 'error' not in r]
        
        if not valid_results:
            return {'error': 'No valid results found'}
        
        # Calculate summary statistics
        total_sheets = len(valid_results)
        scores = [r['total_score'] for r in valid_results]
        accuracies = [r['accuracy'] for r in valid_results]
        
        summary = {
            'total_sheets_evaluated': total_sheets,
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'std_score': np.std(scores),
            'average_accuracy': np.mean(accuracies),
            'pass_rate': len([s for s in scores if s >= 60]) / total_sheets * 100,  # Assuming 60% pass rate
        }
        
        # Subject-wise analysis
        subject_summary = {}
        for subject in self.subjects:
            subject_scores = []
            for result in valid_results:
                if 'subject_scores' in result and subject in result['subject_scores']:
                    subject_scores.append(result['subject_scores'][subject])
            
            if subject_scores:
                subject_summary[subject] = {
                    'average_score': np.mean(subject_scores),
                    'max_score': np.max(subject_scores),
                    'min_score': np.min(subject_scores),
                    'pass_rate': len([s for s in subject_scores if s >= 12]) / len(subject_scores) * 100  # 60% of 20
                }
        
        summary['subject_summary'] = subject_summary
        
        return summary
    
    def get_answer_key_preview(self) -> pd.DataFrame:
        """
        Get a preview of loaded answer keys
        
        Returns:
            DataFrame with answer key preview
        """
        if not self.answer_keys:
            return pd.DataFrame()
        
        # Create preview DataFrame
        preview_data = {}
        for subject, answers in self.answer_keys.items():
            preview_data[subject] = answers[:10]  # First 10 answers for preview
        
        preview_df = pd.DataFrame(preview_data)
        preview_df.index = [f"Q{i+1}" for i in range(len(preview_df))]
        
        return preview_df
    
    def validate_detected_answers(self, detected_answers: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate detected answers format
        
        Args:
            detected_answers: List of detected answers
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check length
        expected_length = self.config['omr_settings']['total_questions']
        if len(detected_answers) != expected_length:
            issues.append(f"Expected {expected_length} answers, got {len(detected_answers)}")
        
        # Check answer format
        valid_answers = set(['a', 'b', 'c', 'd', ''])
        for i, answer in enumerate(detected_answers):
            if answer.lower().strip() not in valid_answers:
                issues.append(f"Invalid answer '{answer}' at position {i+1}")
        
        return len(issues) == 0, issues