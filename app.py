"""
OMR Evaluation System - Streamlit Web Application
Main application for automated OMR sheet evaluation
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os
import zipfile
import io
from typing import List, Dict, Any

# Import our custom modules
from src.image_processing import OMRImageProcessor
from src.answer_detection import OMRAnswerDetector
from src.evaluation import OMREvaluator
from src.utils import (
    load_config, create_directories, convert_pil_to_cv2, 
    convert_cv2_to_pil, format_results_summary, save_results_to_excel,
    get_file_size_mb
)

# Configure Streamlit page
st.set_page_config(
    page_title="OMR Evaluation System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class OMRApp:
    """Main OMR Application Class"""
    
    def __init__(self):
        """Initialize the application"""
        # Create necessary directories
        create_directories()
        
        # Load configuration
        self.config = load_config()
        
        # Initialize processors
        self.image_processor = OMRImageProcessor()
        self.answer_detector = OMRAnswerDetector()
        self.evaluator = OMREvaluator()
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'processed_results' not in st.session_state:
            st.session_state.processed_results = []
        if 'answer_key_loaded' not in st.session_state:
            st.session_state.answer_key_loaded = False
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'batch_processing' not in st.session_state:
            st.session_state.batch_processing = False
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🎯 OMR Evaluation System</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Quick stats in header
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", "100")
        with col2:
            st.metric("Subjects", "5")
        with col3:
            st.metric("Questions per Subject", "20")
        with col4:
            total_processed = len(st.session_state.processed_results)
            st.metric("Sheets Processed", total_processed)
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.header("⚙️ Configuration")
        
        # Answer key section
        st.sidebar.subheader("📋 Answer Key")
        
        # Option to use default answer key or upload new one
        answer_key_option = st.sidebar.radio(
            "Answer Key Source:",
            ["Use Default Key", "Upload New Key"]
        )
        
        if answer_key_option == "Use Default Key":
            if st.sidebar.button("Load Default Answer Key"):
                if self.load_default_answer_key():
                    st.sidebar.success("✅ Default answer key loaded!")
                    st.session_state.answer_key_loaded = True
                else:
                    st.sidebar.error("❌ Failed to load default answer key")
        else:
            uploaded_key = st.sidebar.file_uploader(
                "Upload Answer Key (Excel)",
                type=['xlsx', 'xls'],
                help="Upload Excel file with answer keys"
            )
            
            if uploaded_key is not None:
                if self.evaluator.load_answer_key_from_uploaded_file(uploaded_key):
                    st.sidebar.success("✅ Answer key loaded successfully!")
                    st.session_state.answer_key_loaded = True
                    
                    # Show preview
                    preview_df = self.evaluator.get_answer_key_preview()
                    if not preview_df.empty:
                        st.sidebar.subheader("📋 Answer Key Preview")
                        st.sidebar.dataframe(preview_df)
                else:
                    st.sidebar.error("❌ Failed to load answer key")
        
        # Processing options
        st.sidebar.subheader("🔧 Processing Options")
        
        apply_perspective = st.sidebar.checkbox(
            "Apply Perspective Correction", 
            value=True,
            help="Automatically correct perspective distortion"
        )
        
        show_debug_images = st.sidebar.checkbox(
            "Show Debug Images",
            value=False,
            help="Display image processing steps"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence for answer detection"
        )
        
        return {
            'apply_perspective': apply_perspective,
            'show_debug_images': show_debug_images,
            'confidence_threshold': confidence_threshold
        }
    
    def load_default_answer_key(self) -> bool:
        """Load the default answer key file"""
        default_key_path = self.config.get('paths', {}).get('answer_keys', 'Key (Set A and B).xlsx')
        
        if os.path.exists(default_key_path):
            return self.evaluator.load_answer_key(default_key_path)
        else:
            st.error(f"Default answer key file not found: {default_key_path}")
            return False
    
    def process_single_image(self, uploaded_file, processing_options: Dict) -> Dict[str, Any]:
        """Process a single OMR image"""
        try:
            # Load and convert image
            image = Image.open(uploaded_file)
            cv_image = convert_pil_to_cv2(image)
            
            # Store original for display
            st.session_state.current_image = cv_image.copy()
            
            # Process image
            with st.spinner("Processing image..."):
                processed_image = self.image_processor.process_omr_image(
                    cv_image, 
                    apply_perspective_correction=processing_options['apply_perspective']
                )
            
            # Validate processed image
            is_valid, validation_message = self.image_processor.validate_processed_image(processed_image)
            if not is_valid:
                st.error(f"Image validation failed: {validation_message}")
                return None
            
            # Extract answers
            with st.spinner("Detecting answers..."):
                detected_answers = self.answer_detector.extract_answers(processed_image)
            
            # Validate answers
            is_valid_answers, answer_issues = self.evaluator.validate_detected_answers(detected_answers)
            if not is_valid_answers:
                st.warning(f"Answer validation issues: {', '.join(answer_issues)}")
            
            # Evaluate if answer key is loaded
            result = {
                'filename': uploaded_file.name,
                'detected_answers': detected_answers,
                'processed_image': processed_image,
                'original_image': cv_image,
                'processing_success': True
            }
            
            if st.session_state.answer_key_loaded:
                with st.spinner("Evaluating answers..."):
                    evaluation_result = self.evaluator.evaluate_omr_sheet(detected_answers)
                    result.update(evaluation_result)
            
            return result
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    def render_single_processing_tab(self, processing_options: Dict):
        """Render single image processing tab"""
        st.header("📄 Single Image Processing")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload OMR Sheet Image",
            type=['jpg', 'jpeg', 'png', 'pdf'],
            help="Upload a clear image of the OMR sheet"
        )
        
        if uploaded_file is not None:
            # Check file size
            file_size_mb = get_file_size_mb(uploaded_file)
            max_size = self.config.get('ui_settings', {}).get('max_file_size_mb', 10)
            
            if file_size_mb > max_size:
                st.error(f"File size ({file_size_mb:.1f}MB) exceeds maximum limit ({max_size}MB)")
                return
            
            # Show file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**File:** {uploaded_file.name}")
            with col2:
                st.info(f"**Size:** {file_size_mb:.2f} MB")
            with col3:
                st.info(f"**Type:** {uploaded_file.type}")
            
            # Process button
            if st.button("🚀 Process Image", type="primary"):
                if not st.session_state.answer_key_loaded:
                    st.warning("⚠️ Answer key not loaded. Only answer detection will be performed.")
                
                result = self.process_single_image(uploaded_file, processing_options)
                
                if result:
                    # Add to processed results
                    st.session_state.processed_results.append(result)
                    
                    # Display results
                    self.display_processing_results(result, processing_options)
    
    def render_batch_processing_tab(self, processing_options: Dict):
        """Render batch processing tab"""
        st.header("📁 Batch Processing")
        
        # File uploader for multiple files
        uploaded_files = st.file_uploader(
            "Upload Multiple OMR Sheet Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple OMR sheet images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for processing")
            
            # Show file list
            with st.expander("📋 File List", expanded=False):
                for i, file in enumerate(uploaded_files, 1):
                    st.write(f"{i}. {file.name} ({get_file_size_mb(file):.2f} MB)")
            
            # Process all button
            if st.button("🚀 Process All Images", type="primary"):
                if not st.session_state.answer_key_loaded:
                    st.warning("⚠️ Answer key not loaded. Only answer detection will be performed.")
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                    progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    result = self.process_single_image(uploaded_file, processing_options)
                    if result:
                        batch_results.append(result)
                
                # Store batch results
                st.session_state.processed_results.extend(batch_results)
                st.session_state.batch_processing = True
                
                status_text.text("✅ Batch processing completed!")
                
                # Display batch summary
                if batch_results:
                    self.display_batch_results(batch_results)
    
    def display_processing_results(self, result: Dict[str, Any], processing_options: Dict):
        """Display results of single image processing"""
        st.subheader("📊 Processing Results")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Results", "🔍 Detection", "📈 Analysis", "🖼️ Images"])
        
        with tab1:
            if 'total_score' in result:
                # Evaluation results available
                col1, col2 = st.columns(2)
                
                with col1:
                    # Score metrics
                    st.metric(
                        label="Total Score",
                        value=f"{result['total_score']}/100",
                        delta=f"{result['accuracy']:.1f}%"
                    )
                    
                    # Subject scores
                    st.subheader("📚 Subject Scores")
                    subject_df = pd.DataFrame([
                        {"Subject": subject, "Score": f"{score}/20", "Percentage": f"{(score/20)*100:.1f}%"}
                        for subject, score in result.get('subject_scores', {}).items()
                    ])
                    st.dataframe(subject_df, use_container_width=True)
                
                with col2:
                    # Score distribution pie chart
                    fig = px.pie(
                        values=list(result.get('subject_scores', {}).values()),
                        names=list(result.get('subject_scores', {}).keys()),
                        title="Score Distribution by Subject"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                with st.expander("📝 Detailed Question Analysis"):
                    if 'detailed_results' in result:
                        detailed_df = pd.DataFrame(result['detailed_results'])
                        st.dataframe(detailed_df, use_container_width=True)
                        
            else:
                # Only detection results
                st.info("Answer key not loaded. Showing detection results only.")
                st.subheader("🔍 Detected Answers")
                
                # Display answers in a grid
                answers = result.get('detected_answers', [])
                self.display_answer_grid(answers)
        
        with tab2:
            # Detection statistics
            if 'processed_image' in result:
                stats = self.answer_detector.get_detection_statistics(
                    result['processed_image'], 
                    result.get('detected_answers', [])
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Bubbles Detected", stats.get('total_bubbles_detected', 0))
                with col2:
                    st.metric("Questions Formed", stats.get('question_groups_formed', 0))
                with col3:
                    st.metric("Filled Answers", stats.get('filled_answers', 0))
                
                # Answer distribution chart
                if 'answer_distribution' in stats:
                    dist_data = stats['answer_distribution']
                    fig = px.bar(
                        x=list(dist_data.keys()),
                        y=list(dist_data.values()),
                        title="Answer Distribution",
                        labels={'x': 'Answer Option', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Multiple markings detection
            if 'processed_image' in result:
                multiple_markings = self.answer_detector.detect_multiple_markings(result['processed_image'])
                
                if multiple_markings:
                    st.warning(f"⚠️ Multiple markings detected in questions: {', '.join(map(str, multiple_markings))}")
                else:
                    st.success("✅ No multiple markings detected")
                
                # Quality metrics
                st.subheader("📊 Quality Metrics")
                if 'total_score' in result:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correct Answers", result.get('correct_answers', 0))
                    with col2:
                        st.metric("Wrong Answers", result.get('wrong_answers', 0))
                    with col3:
                        st.metric("Unanswered", result.get('unanswered', 0))
        
        with tab4:
            # Image display
            if processing_options['show_debug_images'] and 'original_image' in result and 'processed_image' in result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(convert_cv2_to_pil(result['original_image']), use_column_width=True)
                
                with col2:
                    st.subheader("Processed Image")
                    st.image(convert_cv2_to_pil(result['processed_image']), use_column_width=True)
                
                # Detection visualization
                if 'processed_image' in result:
                    bubbles = self.answer_detector.detect_bubbles(result['processed_image'])
                    if bubbles:
                        vis_image = self.answer_detector.create_detection_visualization(
                            result['processed_image'], 
                            bubbles, 
                            result.get('detected_answers', [])
                        )
                        st.subheader("Detection Visualization")
                        st.image(convert_cv2_to_pil(vis_image), use_column_width=True)
    
    def display_answer_grid(self, answers: List[str]):
        """Display answers in a grid format"""
        # Display answers in rows of 10
        for row in range(0, len(answers), 10):
            cols = st.columns(10)
            for i, col in enumerate(cols):
                if row + i < len(answers):
                    answer = answers[row + i] or "—"
                    col.metric(f"Q{row + i + 1}", answer)
    
    def display_batch_results(self, batch_results: List[Dict[str, Any]]):
        """Display batch processing results"""
        st.subheader("📊 Batch Processing Summary")
        
        # Summary statistics
        if st.session_state.answer_key_loaded:
            summary = self.evaluator.generate_summary_report(batch_results)
            
            if summary and 'error' not in summary:
                # Overall metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sheets Processed", summary['total_sheets_evaluated'])
                with col2:
                    st.metric("Average Score", f"{summary['average_score']:.1f}/100")
                with col3:
                    st.metric("Pass Rate", f"{summary['pass_rate']:.1f}%")
                with col4:
                    st.metric("Avg Accuracy", f"{summary['average_accuracy']:.1f}%")
                
                # Subject-wise analysis
                if 'subject_summary' in summary:
                    st.subheader("📚 Subject-wise Performance")
                    
                    subject_data = []
                    for subject, stats in summary['subject_summary'].items():
                        subject_data.append({
                            'Subject': subject,
                            'Avg Score': f"{stats['average_score']:.1f}/20",
                            'Max Score': f"{stats['max_score']}/20",
                            'Min Score': f"{stats['min_score']}/20",
                            'Pass Rate': f"{stats['pass_rate']:.1f}%"
                        })
                    
                    subject_df = pd.DataFrame(subject_data)
                    st.dataframe(subject_df, use_container_width=True)
                    
                    # Subject performance chart
                    fig = px.bar(
                        subject_df,
                        x='Subject',
                        y='Avg Score',
                        title="Average Score by Subject",
                        text='Avg Score'
                    )
                    fig.update_traces(texttemplate='%{text}', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
        
        # Individual results table
        st.subheader("📋 Individual Results")
        
        results_data = []
        for result in batch_results:
            if 'total_score' in result:
                results_data.append({
                    'Filename': result['filename'],
                    'Total Score': f"{result['total_score']}/100",
                    'Accuracy': f"{result['accuracy']:.1f}%",
                    'Correct': result.get('correct_answers', 0),
                    'Wrong': result.get('wrong_answers', 0),
                    'Unanswered': result.get('unanswered', 0)
                })
            else:
                results_data.append({
                    'Filename': result['filename'],
                    'Status': 'Processed (No Evaluation)',
                    'Detected Answers': len([a for a in result.get('detected_answers', []) if a])
                })
        
        if results_data:
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Export button
            if st.button("📥 Export Results to Excel"):
                output_path = save_results_to_excel(results_data, f"batch_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                if output_path:
                    st.success(f"✅ Results exported to {output_path}")
    
    def render_results_tab(self):
        """Render results and analytics tab"""
        st.header("📈 Results & Analytics")
        
        if not st.session_state.processed_results:
            st.info("No results available. Please process some OMR sheets first.")
            return
        
        # Filter results with evaluation data
        evaluated_results = [r for r in st.session_state.processed_results if 'total_score' in r]
        
        if not evaluated_results:
            st.info("No evaluation results available. Please load an answer key and process OMR sheets.")
            return
        
        # Overall analytics
        self.render_analytics_dashboard(evaluated_results)
    
    def render_analytics_dashboard(self, results: List[Dict[str, Any]]):
        """Render analytics dashboard"""
        # Summary metrics
        total_sheets = len(results)
        scores = [r['total_score'] for r in results]
        avg_score = np.mean(scores)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sheets", total_sheets)
        with col2:
            st.metric("Average Score", f"{avg_score:.1f}/100")
        with col3:
            st.metric("Highest Score", f"{max(scores)}/100")
        with col4:
            st.metric("Lowest Score", f"{min(scores)}/100")
        
        # Score distribution histogram
        st.subheader("📊 Score Distribution")
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="Score Distribution",
            labels={'x': 'Total Score', 'y': 'Number of Students'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Subject-wise performance
        st.subheader("📚 Subject-wise Performance Analysis")
        
        # Collect subject data
        subject_data = {}
        subjects = self.config.get('subjects', [])
        
        for subject in subjects:
            scores = [r.get('subject_scores', {}).get(subject, 0) for r in results]
            subject_data[subject] = {
                'scores': scores,
                'avg': np.mean(scores),
                'max': max(scores),
                'min': min(scores)
            }
        
        # Subject performance comparison
        subject_names = list(subject_data.keys())
        avg_scores = [subject_data[s]['avg'] for s in subject_names]
        
        fig = px.bar(
            x=subject_names,
            y=avg_scores,
            title="Average Performance by Subject",
            labels={'x': 'Subject', 'y': 'Average Score'},
            text=[f"{score:.1f}" for score in avg_scores]
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed subject analysis
        with st.expander("📈 Detailed Subject Analysis"):
            for subject in subjects:
                st.subheader(f"{subject}")
                data = subject_data[subject]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average", f"{data['avg']:.1f}/20")
                with col2:
                    st.metric("Maximum", f"{data['max']}/20")
                with col3:
                    st.metric("Minimum", f"{data['min']}/20")
                with col4:
                    pass_rate = len([s for s in data['scores'] if s >= 12]) / len(data['scores']) * 100
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")
    
    def run(self):
        """Run the main application"""
        # Render header
        self.render_header()
        
        # Render sidebar
        processing_options = self.render_sidebar()
        
        # Main content with tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📄 Single Processing", 
            "📁 Batch Processing", 
            "📊 Results & Analytics",
            "ℹ️ About"
        ])
        
        with tab1:
            self.render_single_processing_tab(processing_options)
        
        with tab2:
            self.render_batch_processing_tab(processing_options)
        
        with tab3:
            self.render_results_tab()
        
        with tab4:
            self.render_about_tab()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "🏢 **Innomatics Research Labs** - OMR Evaluation System v1.0 | "
            "Built with ❤️ using Streamlit"
        )
    
    def render_about_tab(self):
        """Render about tab with system information"""
        st.header("ℹ️ About OMR Evaluation System")
        
        st.markdown("""
        ### 🎯 Overview
        This **Automated OMR Evaluation System** is designed for **Innomatics Research Labs** 
        to efficiently evaluate placement readiness assessments across multiple subjects.
        
        ### 📋 Features
        - **📱 Mobile Camera Support**: Process OMR sheets captured via mobile phone
        - **🔄 Automatic Processing**: Perspective correction, skew correction, and noise removal
        - **🎯 High Accuracy**: <0.5% error tolerance with advanced bubble detection
        - **📊 Multiple Formats**: Support for different OMR sheet versions (Sets A & B)
        - **📈 Detailed Analytics**: Subject-wise scoring and performance analysis
        - **⚡ Fast Processing**: Batch processing capabilities for multiple sheets
        - **📤 Export Options**: Results export to Excel format
        
        ### 🔧 System Specifications
        """)
        
        # System specs table
        specs_data = {
            "Parameter": [
                "Total Questions",
                "Number of Subjects", 
                "Questions per Subject",
                "Answer Options",
                "Supported Formats",
                "Max File Size",
                "Error Tolerance",
                "Processing Speed"
            ],
            "Value": [
                "100",
                "5 (Python, EDA, SQL, Power BI, Statistics)",
                "20",
                "A, B, C, D",
                "JPG, JPEG, PNG, PDF",
                "10 MB",
                "<0.5%",
                "~30 seconds per sheet"
            ]
        }
        
        specs_df = pd.DataFrame(specs_data)
        st.table(specs_df)
        
        st.markdown("""
        ### 🚀 How to Use
        
        1. **Load Answer Key**: Use the default key or upload your own Excel file
        2. **Upload OMR Sheets**: Single image or batch upload supported
        3. **Process & Review**: Automatic processing with quality validation
        4. **Analyze Results**: View detailed scores and performance analytics
        5. **Export Data**: Download results in Excel format
        
        ### 🔍 Processing Pipeline
        
        1. **Image Preprocessing**
           - Perspective correction
           - Skew detection and correction
           - Noise removal and enhancement
        
        2. **Answer Detection**
           - Bubble detection using contour analysis
           - Fill ratio calculation
           - Confidence scoring
        
        3. **Evaluation**
           - Answer key matching
           - Subject-wise scoring
           - Detailed result generation
        
        ### ⚙️ Technical Stack
        - **Frontend**: Streamlit
        - **Image Processing**: OpenCV, scikit-image
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly
        - **File Handling**: openpyxl, Pillow
        
        ### 📞 Support
        For technical support or feature requests, please contact the development team.
        """)
        
        # Sample images section
        st.subheader("📸 Sample OMR Sheets")
        
        # Check if sample images exist
        sample_path_a = "samples/Set A"
        sample_path_b = "samples/Set B"
        
        if os.path.exists(sample_path_a):
            sample_files_a = [f for f in os.listdir(sample_path_a) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if sample_files_a:
                st.write("**Set A Samples:**")
                sample_file = os.path.join(sample_path_a, sample_files_a[0])
                try:
                    sample_image = Image.open(sample_file)
                    st.image(sample_image, caption=f"Sample from Set A: {sample_files_a[0]}", width=400)
                except:
                    st.info("Sample image could not be loaded")
        
        if os.path.exists(sample_path_b):
            sample_files_b = [f for f in os.listdir(sample_path_b) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if sample_files_b:
                st.write("**Set B Samples:**")
                sample_file = os.path.join(sample_path_b, sample_files_b[0])
                try:
                    sample_image = Image.open(sample_file)
                    st.image(sample_image, caption=f"Sample from Set B: {sample_files_b[0]}", width=400)
                except:
                    st.info("Sample image could not be loaded")

# Main execution
def main():
    """Main function to run the application"""
    try:
        app = OMRApp()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please check your configuration and try again.")

if __name__ == "__main__":
    main()