import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
from dotenv import load_dotenv
import pathlib
import base64
import cv2
from ultralytics import YOLO
import tempfile
import http.client
import json
import requests
import urllib3

# Disable SSL warning messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set matplotlib style for better looking plots
plt.style.use('dark_background')

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key="AIzaSyCCKYQT6Hrx_b8IrSBmPy6a82DYlIz5pE4")

# Set up the model
model = genai.GenerativeModel('gemini-1.5-pro')

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

# Set page config
st.set_page_config(
    page_title="Video Red Flag Analyzer",
    page_icon="🚨",
    layout="wide"
)

# Configure plot settings for dark theme
plt.rcParams.update({
    'figure.facecolor': '#0E1117',
    'axes.facecolor': '#0E1117',
    'axes.edgecolor': '#FFFFFF',
    'axes.labelcolor': '#FFFFFF',
    'text.color': '#FFFFFF',
    'xtick.color': '#FFFFFF',
    'ytick.color': '#FFFFFF',
    'grid.color': '#FFFFFF',
    'grid.alpha': 0.1
})

# Initialize session state
if 'video_file' not in st.session_state:
    st.session_state.video_file = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'annotated_video_path' not in st.session_state:
    st.session_state.annotated_video_path = None

def get_mime_type(file_path):
    """Get MIME type based on file extension"""
    extension = pathlib.Path(file_path).suffix.lower()
    mime_types = {
        '.mp4': 'video/mp4',
        '.avi': 'video/x-msvideo',
        '.mov': 'video/quicktime'
    }
    return mime_types.get(extension, 'video/mp4')

def process_video_with_yolo(video_path):
    """Process video with YOLOv8 and return annotated video path"""
    try:
        # Create a temporary file for the annotated video
        temp_dir = tempfile.gettempdir()
        output_path = os.path.join(temp_dir, 'annotated_video.mp4')
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open the video file")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Try different codecs in order of preference
        codecs = [
            ('avc1', '.mp4'),  # H.264 codec
            ('mp4v', '.mp4'),  # MPEG-4 codec
            ('XVID', '.avi'),  # XVID codec
            ('MJPG', '.avi'),  # Motion JPEG codec
        ]
        
        out = None
        for codec, ext in codecs:
            try:
                output_path = os.path.join(temp_dir, f'annotated_video{ext}')
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    break
            except Exception as e:
                continue
        
        if out is None or not out.isOpened():
            st.error("Error: Could not initialize video writer with any available codec")
            cap.release()
            return None
        
        # Process each frame
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8 inference
            results = yolo_model(frame)
            
            # Draw the results on the frame
            annotated_frame = results[0].plot()
            
            # Write the frame
            out.write(annotated_frame)
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            progress_text.text(f"Processing video: {int(progress * 100)}%")
        
        # Release everything
        cap.release()
        out.release()
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            progress_text.empty()
            progress_bar.empty()
            return output_path
        else:
            st.error("Error: Failed to save the annotated video")
            return None
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        return None

def analyze_video(video_file):
    """Analyze video using Gemini API"""
    try:
        # Save the uploaded file temporarily
        temp_path = "temp_video" + pathlib.Path(video_file.name).suffix
        with open(temp_path, "wb") as f:
            f.write(video_file.getvalue())
        
        # Process video with YOLOv8
        annotated_video_path = process_video_with_yolo(temp_path)
        if annotated_video_path:
            st.session_state.annotated_video_path = annotated_video_path
        
        # Prepare video data
        with st.spinner("Analyzing video content..."):
            with open(temp_path, 'rb') as f:
                video_bytes = f.read()
            
            # Create the content parts
            content = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": """
                            Analyze this video and identify any red flags such as:
                            - Accidents
                            - Weapons
                            - Theft
                            - Crimes
                            - Any suspicious activities
                            
                            For each red flag found, provide:
                            1. A detailed description
                            2. The exact timestamp when it occurs
                            3. The severity level (Low, Medium, High)
                            
                            Also provide a comprehensive summary of the video.
                            Format the response in a structured way with clear sections.
                            """
                        },
                        {
                            "inline_data": {
                                "mime_type": get_mime_type(temp_path),
                                "data": base64.b64encode(video_bytes).decode('utf-8')
                            }
                        }
                    ]
                }
            ]
            
            # Generate content with Gemini
            response = model.generate_content(content)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            return response.text
    
    except Exception as e:
        st.error(f"Error analyzing video: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return None

def create_timeline_graph(red_flags):
    """Create timeline graphs of red flags"""
    if not red_flags:
        return None
    
    # Create figure with a specific size
    fig = plt.figure(figsize=(15, 10))
    
    # Create subplots with specific heights
    gs = plt.GridSpec(3, 1, height_ratios=[1.5, 1, 1.5], hspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Extract data
    timestamps = []
    descriptions = []
    severities = []
    
    for flag in red_flags:
        try:
            time_str = flag.split("Timestamp: ")[1].split("\n")[0].strip()
            if '-' in time_str:
                time_str = time_str.split('-')[0]
            time_parts = time_str.split(":")
            if len(time_parts) == 3:
                h, m, s = map(int, time_parts)
                total_seconds = h * 3600 + m * 60 + s
                timestamps.append(total_seconds)
                desc = flag.split("Description: ")[1].split("\n")[0].strip()
                descriptions.append(desc)
                severity = flag.split("Severity: ")[1].split("\n")[0].strip()
                severities.append(severity)
        except (IndexError, ValueError) as e:
            continue
    
    if not timestamps:
        return None
    
    # Colors for severity levels
    severity_colors = {
        'High': '#FF4B4B',
        'Medium': '#FFA500',
        'Low': '#FFD700'
    }
    
    # Clear any existing plots
    plt.clf()
    
    # Plot 1: Timeline
    ax1.set_facecolor('#1E1E1E')
    ax1.set_xlim(0, max(timestamps) + 10)
    ax1.set_ylim(0, 1)
    
    for i, (ts, desc, sev) in enumerate(zip(timestamps, descriptions, severities)):
        color = severity_colors.get(sev, 'gray')
        ax1.axvline(x=ts, color=color, linewidth=3, alpha=0.8)
        ax1.text(ts, 0.5, f"Flag {i+1}\n{desc[:30]}...", 
                rotation=90, va='center', ha='right',
                bbox=dict(facecolor='#2E2E2E', alpha=0.9, edgecolor='none'),
                color='white', fontsize=8)
    
    ax1.set_title('Red Flags Timeline', fontsize=14, pad=20, color='white')
    ax1.grid(True, linestyle='--', alpha=0.2)
    ax1.set_yticks([])
    
    # Plot 2: Severity Distribution
    ax2.set_facecolor('#1E1E1E')
    severity_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for sev in severities:
        if sev in severity_counts:
            severity_counts[sev] += 1
    
    bars = ax2.bar(severity_counts.keys(), severity_counts.values())
    for bar, (severity, _) in zip(bars, severity_counts.items()):
        bar.set_color(severity_colors[severity])
        bar.set_alpha(0.8)
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', color='white')
    
    ax2.set_title('Red Flags by Severity', fontsize=14, color='white')
    ax2.set_ylabel('Count', color='white')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.2)
    
    # Plot 3: Time Distribution
    ax3.set_facecolor('#1E1E1E')
    time_periods = [ts // 60 for ts in timestamps]
    from collections import Counter
    period_counts = Counter(time_periods)
    
    x = sorted(period_counts.keys())
    y = [period_counts[p] for p in x]
    
    ax3.plot(x, y, color='#4A90E2', linewidth=2, marker='o')
    ax3.fill_between(x, y, alpha=0.2, color='#4A90E2')
    
    for i, v in zip(x, y):
        ax3.text(i, v, str(v), ha='center', va='bottom', color='white')
    
    ax3.set_title('Red Flags per Minute', fontsize=14, color='white')
    ax3.set_xlabel('Time (minutes)', color='white')
    ax3.set_ylabel('Number of Red Flags', color='white')
    ax3.grid(True, linestyle='--', alpha=0.2)
    ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig

def get_safety_tips(red_flags):
    """Get safety tips and prevention strategies based on red flags using Serper API with Gemini-generated queries"""
    if not red_flags:
        return None
    
    try:
        # Use Gemini to generate a relevant search query
        prompt = f"""
        Based on the following incident(s), generate a specific search query to find safety tips and prevention strategies.
        Focus on the most critical aspects that need safety guidance.
        
        Incidents:
        {red_flags}
        
        Generate a search query that would help find relevant safety information. 
        The query should be specific but concise (maximum 10 words).
        Return ONLY the search query, nothing else.
        """
        
        # Generate search query using Gemini
        query_response = model.generate_content(prompt)
        search_query = query_response.text.strip()
        
        # Use requests instead of http.client for better SSL handling
        
        # Make request to Serper API
        url = "https://google.serper.dev/search"
        payload = {
            "q": f"{search_query} safety prevention guidelines tips",
            "num": 5
        }
        headers = {
            'X-API-KEY': '8e0248bcce65ba8a469abc9ae4670a2247d20936',
            'Content-Type': 'application/json'
        }
        
        # Make request with SSL verification disabled (only for development)
        response = requests.post(url, json=payload, headers=headers, verify=False)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        
        # Extract relevant information from results
        safety_tips = []
        if 'organic' in data:
            # Add the search query as context
            safety_tips.append(f"🔍 Based on: '{search_query}'")
            
            for result in data['organic']:
                if 'snippet' in result:
                    # Clean and format the snippet
                    snippet = result['snippet'].strip()
                    if snippet:
                        safety_tips.append(snippet)
                if 'title' in result:
                    # Include titles of relevant resources
                    title = result['title'].strip()
                    if title and "..." not in title:
                        safety_tips.append(f"📚 Resource: {title}")
        
        return safety_tips if len(safety_tips) > 1 else None  # Return None if only search query is present
    
    except Exception as e:
        st.error(f"Error fetching safety tips: {str(e)}")
        return None

def main():
    st.title("🚨 Video Red Flag Analyzer")
    st.write("Upload a video to analyze for potential red flags and suspicious activities.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        st.session_state.video_file = uploaded_file
        st.video(uploaded_file)
        
        if st.button("Analyze Video"):
            with st.spinner("Analyzing video..."):
                analysis_results = analyze_video(uploaded_file)
                st.session_state.analysis_results = analysis_results
    
    # Display annotated video if available
    if st.session_state.annotated_video_path and os.path.exists(st.session_state.annotated_video_path):
        st.header("Object Detection Results")
        st.video(st.session_state.annotated_video_path)
    
    # Display analysis results
    if st.session_state.analysis_results:
        st.header("Analysis Results")
        
        # Split results into sections
        sections = st.session_state.analysis_results.split("\n\n")
        
        # Display summary
        st.subheader("Video Summary")
        st.write(sections[0])
        
        # Display red flags
        st.subheader("Red Flags Detected")
        red_flags = [section for section in sections if "Timestamp:" in section]
        
        if red_flags:
            # Create metrics for quick statistics
            total_flags = len(red_flags)
            high_severity = len([flag for flag in red_flags if "Severity: High" in flag])
            medium_severity = len([flag for flag in red_flags if "Severity: Medium" in flag])
            low_severity = len([flag for flag in red_flags if "Severity: Low" in flag])
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Red Flags", total_flags)
            with col2:
                st.metric("High Severity", high_severity)
            with col3:
                st.metric("Medium Severity", medium_severity)
            with col4:
                st.metric("Low Severity", low_severity)
            
            # Display individual red flags
            for i, flag in enumerate(red_flags, 1):
                with st.expander(f"Red Flag {i}"):
                    st.write(flag)
            
            # Create and display visualizations
            st.subheader("Red Flags Analysis")
            
            # Create the visualization
            timeline_fig = create_timeline_graph(red_flags)
            
            # Display the visualization with explicit size
            if timeline_fig:
                st.write("### Timeline and Distribution Analysis")
                st.pyplot(timeline_fig, use_container_width=True)
            
            # Get and display safety tips
            st.subheader("🛡️ Safety Tips & Prevention Strategies")
            with st.spinner("Generating safety recommendations..."):
                safety_tips = get_safety_tips(red_flags)
                if safety_tips:
                    # Display search context first
                    st.write(safety_tips[0])  # Display the search query used
                    
                    # Display tips in expandable sections
                    for i, tip in enumerate(safety_tips[1:], 1):  # Skip the first item (search query)
                        if tip.startswith("📚 Resource:"):
                            # Display resources differently
                            st.write(tip)
                        else:
                            with st.expander(f"Safety Tip {i}"):
                                st.write(tip)
                else:
                    st.info("No specific safety tips found for the detected incidents.")
            
            # Generate report
            st.subheader("Detailed Report")
            for flag in red_flags:
                st.write(flag)
                st.write("---")
        else:
            st.info("No red flags were detected in the video.")

if __name__ == "__main__":
    main() 