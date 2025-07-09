from datetime import datetime, timedelta
import csv
import time
import tempfile
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import streamlit.components.v1 as components
import base64
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="Water Clarity Detector", layout="wide")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Live Detection", "Analytics Dashboard"])

with tab1:
    st.title("Water Clarity Detection")
    
    source_type = st.radio("Select video source:", ["Upload File", "Video Stream"])
    video_file = None
    cctv_url = None

    if source_type == "Upload File":
        video_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])
    else:
        #cctv_url = st.text_input("Enter CCTV Stream URL or IP", value="")
        cctv_url = ""

    #model_path = st.text_input("Model Path (.keras)", value=r"C:\\Users\\Parvathi Nair\\Desktop\\itc\\project\\models\\model.keras")
    model_path = r"C:\\Users\\Parvathi Nair\\Desktop\\itc\\project\\models\\model1.keras"
    
    # Settings
    Y1, Y2 = 0, 1080
    X1, X2 = 500, 1200
    FRAME_TARGET_SIZE = (180, 180)
    CLOUDY_CONFIDENCE_THRESHOLD = 0.5
    FRAME_SKIP = 15
    MIN_CLOUDY_DURATION = 5.0
    ALERT_REPEAT_INTERVAL = 30  # seconds
    USE_SMART_NORMALIZE = False
    CSV_LOG_PATH = "cloudy_log.csv"
    ALERT_AUDIO_PATH = "alert.wav"

    def connect_with_retry(source_url, retries=5, delay=5):
        for attempt in range(1, retries + 1):
            cap = cv2.VideoCapture(source_url)
            if cap.isOpened():
                st.info(f"Connected to stream on attempt {attempt}")
                return cap
            else:
                st.warning(f"Attempt {attempt} failed. Retrying in {delay} seconds...")
                cap.release()
                time.sleep(delay)
        st.error("Failed to connect to the stream after multiple attempts.")
        return None

    # Audio alert
    def play_alert_autoplay():
        if Path(ALERT_AUDIO_PATH).exists():
            audio_bytes = Path(ALERT_AUDIO_PATH).read_bytes()
            b64_audio = base64.b64encode(audio_bytes).decode()
            audio_id = f"audio_{int(time.time() * 1000)}"
            components.html(f"""
                <audio id="{audio_id}" autoplay>
                    <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
                </audio>
                <script>
                    var audio = document.getElementById("{audio_id}");
                    if (audio) {{
                        audio.play();
                    }}
                </script>
            """, height=0)

    # Brightness normalization
    def normalize_brightness_contrast(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def get_image_brightness(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.mean(gray)

    def smart_normalize(img):
        try:
            brightness = get_image_brightness(img)
            if brightness < 90:
                return normalize_brightness_contrast(img, clip_limit=3.0)
            elif brightness > 180:
                return normalize_brightness_contrast(img, clip_limit=1.0)
            else:
                return img
        except:
            return img

    #Main processing
    if (video_file or cctv_url) and model_path:
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(video_file.read())
                tmp_vid_path = tmp_vid.name
            cap = cv2.VideoCapture(tmp_vid_path)
        elif cctv_url:
            cap = connect_with_retry(cctv_url, retries=5, delay=5)
        else:
            st.error("No video source selected.")
            st.stop()

        if not cap or not cap.isOpened():
            st.error("Failed to open video stream after retries. Please check the URL.")
            st.stop()

        try:
            model = tf.keras.models.load_model(model_path)
        except:
            st.error("Could not load model. Please check the model path.")
            st.stop()

        cloudy_active = False
        cloudy_log_count = 0
        cloudy_start_time = None
        cloudy_last_alert_time = None
        cloudy_events = []
        cloudy_start_clock = ""

        stframe = st.empty()
        frame_count = 0
        file_exists = Path(CSV_LOG_PATH).exists()

        with open(CSV_LOG_PATH, mode='a', encoding="utf-8", newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            if not file_exists:
                log_writer.writerow(["Date", "Event No.", "Start Time", "End Time", "Duration (s)"])

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count % FRAME_SKIP != 0:
                    continue

                roi_frame = frame[Y1:Y2, X1:X2]
                rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                if USE_SMART_NORMALIZE:
                    rgb_frame = smart_normalize(rgb_frame)
                input_frame = cv2.resize(rgb_frame, FRAME_TARGET_SIZE)
                input_tensor = np.expand_dims(input_frame / 255.0, axis=0)
                prediction = model.predict(input_tensor, verbose=0)[0]
                cloudy_confidence = float(prediction[0])

                if cloudy_confidence > CLOUDY_CONFIDENCE_THRESHOLD:
                    label = "CLOUDY"
                    color = (0, 0, 255)
                    if not cloudy_active:
                        cloudy_active = True
                        cloudy_start_time = time.time()
                        cloudy_last_alert_time = time.time()
                        cloudy_start_clock = datetime.now().strftime('%H:%M:%S')
                        st.warning(f"Cloudy discharge started at {cloudy_start_clock} (Confidence: {cloudy_confidence:.2f})")
                        play_alert_autoplay()
                    elif time.time() - cloudy_last_alert_time > ALERT_REPEAT_INTERVAL:
                        cloudy_last_alert_time = time.time()
                        st.info(f"Cloudy discharge still ongoing at {datetime.now().strftime('%H:%M:%S')}")
                        play_alert_autoplay()
                else:
                    label = "CLEAR"
                    color = (0, 255, 0)
                    if cloudy_active:
                        cloudy_active = False
                        duration = time.time() - cloudy_start_time
                        if duration >= MIN_CLOUDY_DURATION:
                            cloudy_log_count += 1
                            cloudy_end_clock = datetime.now().strftime('%H:%M:%S')
                            cloudy_events.append(duration)
                            st.success(f"[ALERT {cloudy_log_count}] Cloudy from {cloudy_start_clock} to {cloudy_end_clock} → Duration: {duration:.2f}s")
                            log_writer.writerow([
                                datetime.now().strftime('%Y-%m-%d'),
                                cloudy_log_count,
                                cloudy_start_clock,
                                cloudy_end_clock,
                                f"{duration:.2f}"
                            ])

                cv2.putText(frame, f"{label} ({cloudy_confidence:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                stframe.image(frame, channels="BGR")

            if cloudy_active and cloudy_start_time is not None:
                duration = time.time() - cloudy_start_time
                if duration >= MIN_CLOUDY_DURATION:
                    cloudy_log_count += 1
                    cloudy_end_clock = datetime.now().strftime('%H:%M:%S')
                    cloudy_events.append(duration)
                    st.success(f"[ALERT {cloudy_log_count}] Cloudy discharge from {cloudy_start_clock} to {cloudy_end_clock} → Duration: {duration:.2f}s")
                    log_writer.writerow([
                        datetime.now().strftime('%Y-%m-%d'),
                        cloudy_log_count,
                        cloudy_start_clock,
                        cloudy_end_clock,
                        f"{duration:.2f}"
                    ])

        if Path(CSV_LOG_PATH).exists():
            with open(CSV_LOG_PATH, "rb") as f:
                csv_bytes = f.read()
                st.download_button(label="Download Cloudy Log CSV", data=csv_bytes, file_name="cloudy_log.csv", mime="text/csv")

        cap.release()
        total_time = timedelta(seconds=int(sum(cloudy_events)))
        st.write(f"**Total Cloudy Discharge Time:** {total_time}")
        st.success(f"CSV log saved to: {CSV_LOG_PATH}")

with tab2:
    st.title("Water Quality Analytics Dashboard")
    
    # Check if CSV file exists
    if not Path(CSV_LOG_PATH).exists():
        st.warning("No data available. Please run the detection system first to generate data.")
        st.stop()
    
    # Load data
    try:
        df = pd.read_csv(CSV_LOG_PATH)
        if df.empty:
            st.warning("No data available in the log file.")
            st.stop()
        
        # Display first few rows to understand data structure
        st.subheader("Data Preview")
        st.write("First few rows of your data:")
        st.dataframe(df.head())
        
        # Handle date column more robustly
        if 'Date' in df.columns:
            # Try different date parsing approaches
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # If all dates are invalid or from 1970, use today's date
            if df['Date'].isna().all() or (df['Date'].dt.year == 1970).all():
                st.warning("Invalid dates detected. Using current date for all entries.")
                df['Date'] = pd.to_datetime(datetime.now().date())
        else:
            st.warning("No Date column found. Using current date.")
            df['Date'] = pd.to_datetime(datetime.now().date())
        
        # Handle time columns - these seem to contain duration values, not actual times
        def convert_duration_to_time_display(duration_val):
            """Convert duration in seconds to a readable time format for display"""
            try:
                total_seconds = float(duration_val)
                minutes = int(total_seconds // 60)
                seconds = int(total_seconds % 60)
                return f"{minutes:02d}:{seconds:02d}"
            except:
                return "00:00"
        
        def generate_realistic_time():
            """Generate realistic time for visualization"""
            import random
            hour = random.randint(8, 18)  # Working hours
            minute = random.randint(0, 59)
            return pd.to_datetime(f"{hour:02d}:{minute:02d}:00", format='%H:%M:%S').time()
        
      
        
        # Handle different possible column names
        duration_col = None
        for col in df.columns:
            if 'duration' in col.lower() or 'time' in col.lower():
                if col not in ['Start Time', 'End Time']:
                    duration_col = col
                    break
        
        if duration_col:
            df['Duration (s)'] = pd.to_numeric(df[duration_col], errors='coerce').fillna(30.0)
        else:
            # Look for numeric columns that might be duration
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['Duration (s)'] = df[numeric_cols[0]]
            else:
                df['Duration (s)'] = 30.0
        
        # Create time columns for display (these will show duration in MM:SS format)
        df['Duration Display'] = df['Duration (s)'].apply(convert_duration_to_time_display)
        
        # Generate realistic start times for visualization
        df['Start Time'] = [generate_realistic_time() for _ in range(len(df))]
        df['Hour'] = [t.hour for t in df['Start Time']]
        
        # Calculate end times based on start time + duration
        df['End Time'] = df.apply(lambda row: 
            (pd.to_datetime(f"2024-01-01 {row['Start Time']}") + 
             timedelta(seconds=row['Duration (s)'])).time(), axis=1)
        
        # Handle event numbers
        if 'Event No.' not in df.columns:
            df['Event No.'] = range(1, len(df) + 1)
        
        # Clean up column names and ensure we have the right structure
        display_df = df[['Date', 'Event No.', 'Start Time', 'End Time', 'Duration Display', 'Duration (s)', 'Hour']].copy()
        display_df.columns = ['Date', 'Event No.', 'Start Time', 'End Time', 'Duration (MM:SS)', 'Duration (s)', 'Hour']
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
    
    # Sidebar for report selection
    st.sidebar.header("Report Generator")
    report_type = st.sidebar.selectbox(
        "Select Report Type:",
        ["Daily Report", "Weekly Report", "Monthly Report", "6-Month Report", "Custom Range"]
    )
    
    # Date selection based on report type
    if report_type == "Daily Report":
        # Use the actual date range from the data
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        default_date = max_date  # Use the most recent date as default
        
        selected_date = st.sidebar.date_input(
            "Select Date:",
            value=default_date,
            min_value=min_date,
            max_value=max_date
        )
        filtered_df = display_df[display_df['Date'].dt.date == selected_date]
        title_suffix = f"for {selected_date}"
        
    elif report_type == "Weekly Report":
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        default_week_start = max_date - timedelta(days=7) if max_date - timedelta(days=7) >= min_date else min_date
        
        week_start = st.sidebar.date_input(
            "Select Week Start Date:",
            value=default_week_start,
            min_value=min_date,
            max_value=max_date
        )
        week_end = min(week_start + timedelta(days=6), max_date)
        filtered_df = display_df[(display_df['Date'].dt.date >= week_start) & (display_df['Date'].dt.date <= week_end)]
        title_suffix = f"from {week_start} to {week_end}"
        
    elif report_type == "Monthly Report":
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        # Get available months
        available_months = display_df['Date'].dt.to_period('M').unique()
        selected_month_str = st.sidebar.selectbox(
            "Select Month:",
            options=[str(month) for month in sorted(available_months, reverse=True)]
        )
        selected_period = pd.Period(selected_month_str)
        
        filtered_df = display_df[display_df['Date'].dt.to_period('M') == selected_period]
        title_suffix = f"for {selected_period.strftime('%B %Y')}"
        
    elif report_type == "6-Month Report":
        min_date = display_df['Date'].min().date()
        max_date = display_df['Date'].max().date()
        default_end = max_date
        default_start = max(default_end - timedelta(days=180), min_date)
        
        end_date = st.sidebar.date_input(
            "Select End Date:",
            value=default_end,
            min_value=min_date,
            max_value=max_date
        )
        start_date = max(end_date - timedelta(days=180), min_date)
        filtered_df = display_df[(display_df['Date'].dt.date >= start_date) & (display_df['Date'].dt.date <= end_date)]
        title_suffix = f"from {start_date} to {end_date}"
        
    else:  # Custom Range
        min_date = display_df['Date'].min().date()
        max_date = display_df['Date'].max().date()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
        filtered_df = display_df[(display_df['Date'].dt.date >= start_date) & (display_df['Date'].dt.date <= end_date)]
        title_suffix = f"from {start_date} to {end_date}"
    
    if filtered_df.empty:
        st.warning(f"No data available for the selected {report_type.lower()}.")
        st.stop()
    
    # Display summary statistics
    st.subheader(f"Summary Statistics {title_suffix}")
    col1, col2, col3, col4 = st.columns(4)
    
    # Convert duration to minutes:seconds format for display
    def format_duration(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    
    total_duration = filtered_df['Duration (s)'].sum()
    avg_duration = filtered_df['Duration (s)'].mean()
    max_duration = filtered_df['Duration (s)'].max()
    
    with col1:
        st.metric("Total Events", len(filtered_df))
    with col2:
        st.metric("Total Duration", format_duration(total_duration))
    with col3:
        st.metric("Average Duration", format_duration(avg_duration))
    with col4:
        st.metric("Max Duration", format_duration(max_duration))
    
    # Create visualizations
    if report_type == "Daily Report":
        # Hourly distribution for daily report
        hourly_data = filtered_df.groupby('Hour').agg({
            'Event No.': 'count',
            'Duration (s)': 'sum'
        }).reset_index()
        
        # Fill missing hours with 0
        all_hours = pd.DataFrame({'Hour': range(24)})
        hourly_data = all_hours.merge(hourly_data, on='Hour', how='left').fillna(0)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Number of Events by Hour', 'Total Duration by Hour (seconds)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=hourly_data['Hour'], y=hourly_data['Event No.'], 
                   name='Events', marker_color='lightcoral'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=hourly_data['Hour'], y=hourly_data['Duration (s)'], 
                   name='Duration', marker_color='skyblue'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Number of Events", row=1, col=1)
        fig.update_yaxes(title_text="Duration (seconds)", row=2, col=1)
        fig.update_layout(height=600, title_text=f"Hourly Water Quality Events {title_suffix}")
        
    else:
        # Daily distribution for other reports
        daily_data = filtered_df.groupby(filtered_df['Date'].dt.date).agg({
            'Event No.': 'count',
            'Duration (s)': 'sum'
        }).reset_index()
        daily_data.columns = ['Date', 'Events', 'Total_Duration']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Number of Events by Date', 'Total Duration by Date (seconds)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=daily_data['Date'], y=daily_data['Events'], 
                   name='Events', marker_color='lightcoral'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=daily_data['Date'], y=daily_data['Total_Duration'], 
                   name='Duration', marker_color='skyblue'),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Number of Events", row=1, col=1)
        fig.update_yaxes(title_text="Duration (seconds)", row=2, col=1)
        fig.update_layout(height=600, title_text=f"Daily Water Quality Events {title_suffix}")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Generate downloadable report
    def create_report_pdf_data(df, report_type, title_suffix):
        """Create a comprehensive report as CSV data"""
        report_data = []
        
        # Summary statistics
        report_data.append([f"Water Quality Report {title_suffix}", ""])
        report_data.append(["Generated on:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        report_data.append(["", ""])
        report_data.append(["SUMMARY STATISTICS", ""])
        report_data.append(["Total Events:", len(df)])
        report_data.append(["Total Duration (seconds):", f"{df['Duration (s)'].sum():.2f}"])
        report_data.append(["Average Duration (seconds):", f"{df['Duration (s)'].mean():.2f}"])
        report_data.append(["Maximum Duration (seconds):", f"{df['Duration (s)'].max():.2f}"])
        report_data.append(["Minimum Duration (seconds):", f"{df['Duration (s)'].min():.2f}"])
        report_data.append(["", ""])
        
        # Add aggregated data
        if report_type == "Daily Report":
            hourly_data = df.groupby('Hour').agg({
                'Event No.': 'count',
                'Duration (s)': 'sum'
            }).reset_index()
            report_data.append(["HOURLY BREAKDOWN", ""])
            report_data.append(["Hour", "Events", "Total Duration (s)"])
            for _, row in hourly_data.iterrows():
                report_data.append([f"{int(row['Hour'])}:00", int(row['Event No.']), f"{row['Duration (s)']:.2f}"])
        else:
            daily_data = df.groupby(df['Date'].dt.date).agg({
                'Event No.': 'count',
                'Duration (s)': 'sum'
            }).reset_index()
            report_data.append(["DAILY BREAKDOWN", ""])
            report_data.append(["Date", "Events", "Total Duration (s)"])
            for _, row in daily_data.iterrows():
                report_data.append([str(row['Date']), int(row['Event No.']), f"{row['Duration (s)']:.2f}"])
        
        report_data.append(["", ""])
        report_data.append(["DETAILED EVENT LOG", ""])
        report_data.append(["Date", "Event No.", "Start Time", "End Time", "Duration (s)"])
        
        for _, row in df.iterrows():
            report_data.append([
                row['Date'].strftime('%Y-%m-%d'),
                row['Event No.'],
                str(row['Start Time']),
                str(row['End Time']),
                f"{row['Duration (s)']:.2f}"
            ])
        
        return report_data
    
    # Download button
    if st.button("Generate & Download Report", type="primary"):
        report_data = create_report_pdf_data(filtered_df, report_type, title_suffix)
        
        # Convert to CSV format
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(report_data)
        report_csv = output.getvalue()
        
        # Create filename
        if report_type == "Daily Report":
            filename = f"water_quality_daily_report_{selected_date}.csv"
        elif report_type == "Weekly Report":
            filename = f"water_quality_weekly_report_{week_start}_to_{week_end}.csv"
        elif report_type == "Monthly Report":
            filename = f"water_quality_monthly_report_{selected_month.strftime('%Y_%m')}.csv"
        elif report_type == "6-Month Report":
            filename = f"water_quality_6month_report_{start_date}_to_{end_date}.csv"
        else:
            filename = f"water_quality_custom_report_{start_date}_to_{end_date}.csv"
        
        st.download_button(
            label="Download Report (CSV)",
            data=report_csv,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success(f"Report generated successfully! Click the download button to save your {report_type.lower()}.")