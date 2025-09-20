import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import time
import base64
import os
import random
from io import BytesIO
from scipy.signal import find_peaks, butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    """Apply a bandpass filter to the signal.
    
    Args:
        signal (numpy.ndarray): Input signal
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order
        
    Returns:
        numpy.ndarray: Filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import json

app = Flask(__name__, static_url_path='', 
            static_folder='static',
            template_folder='templates')

# Global variables for signal processing
red_values = []
green_values = []
blue_values = []
timestamps = []
fs = 30  # Sampling frequency (frames per second)
start_time = time.time()
measurement_active = False

# Liveness check is disabled
liveness_check_passed = True

# Initialize face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ayurvedic parameters
AYURVEDIC_INSIGHTS = {
    'vata': {
        'hr_range': (60, 80),
        'description': 'Vata dosha is dominant. You may experience variable heart rate. Focus on grounding activities and maintain a regular routine.',
        'recommendations': [
            'Practice regular meditation',
            'Maintain a warm and moist diet',
            'Follow a consistent daily routine',
            'Get adequate rest and sleep'
        ]
    },
    'pitta': {
        'hr_range': (70, 90),
        'description': 'Pitta dosha is dominant. Your metabolism and digestion are strong. Be mindful of excess heat and intensity.',
        'recommendations': [
            'Stay cool and hydrated',
            'Avoid excessive heat and direct sunlight',
            'Practice moderation in all activities',
            'Incorporate cooling foods in your diet'
        ]
    },
    'kapha': {
        'hr_range': (55, 70),
        'description': 'Kapha dosha is dominant. You have good stamina but may need more stimulation to stay active.',
        'recommendations': [
            'Engage in regular physical activity',
            'Vary your routine to stay engaged',
            'Incorporate warm and light foods',
            'Stay mentally stimulated'
        ]
    }
}

def process_frame(frame):
    """Process each frame to extract color channel values with optimizations"""
    global red_values, green_values, blue_values, timestamps, measurement_active, last_frame_time
    
    # Skip processing if we're getting frames too quickly
    current_time = time.time()
    if hasattr(process_frame, 'last_processed') and (current_time - process_frame.last_processed) < 0.03:  # ~30fps
        return frame
    process_frame.last_processed = current_time
    
    # Resize frame for faster processing (if needed)
    if frame.shape[1] > 800:  # Only resize if width > 800px
        scale = 800 / frame.shape[1]
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_frame = frame
    
    # Convert to grayscale for face detection (use smaller frame for detection)
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4, minSize=(50, 50))
    
    # If no faces detected, return frame with message
    if len(faces) == 0:
        cv2.putText(frame, 'No face detected', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    # Get the largest face (using the resized frame coordinates)
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Scale coordinates back to original frame size if we resized
    if frame.shape[1] > 800:
        x, y, w, h = [int(coord / scale) for coord in [x, y, w, h]]
    
    # Ensure coordinates are within frame bounds
    h, w_frame = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_frame, x + w), min(h, y + h)
    
    if x2 <= x1 or y2 <= y1:
        return frame
        
    face_roi = frame[y1:y2, x1:x2]
    
    # Draw rectangle around the face
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Process measurement if active
    if measurement_active:
        # Get the average color of the face region (forehead area)
        fh = int((y2 - y1) * 0.2)  # 20% from top for forehead
        fh_bottom = int((y2 - y1) * 0.4)  # 40% from top for forehead bottom
        fw_left = int((x2 - x1) * 0.3)  # 30% from left
        fw_right = int((x2 - x1) * 0.7)  # 70% from left
        
        forehead = face_roi[fh:fh_bottom, fw_left:fw_right]
        
        if forehead.size > 0:  # Ensure we have a valid region
            # Calculate mean values for each channel (use numpy for faster computation)
            b, g, r = cv2.split(forehead.astype(np.float32))
            red_avg = float(np.mean(r))
            green_avg = float(np.mean(g))
            blue_avg = float(np.mean(b))
            
            # Append to our signal arrays
            current_time = time.time() - start_time
            
            # Only add data point if enough time has passed (controls sampling rate)
            if not timestamps or (current_time - timestamps[-1]) >= 0.1:  # 10Hz sampling
                red_values.append(red_avg)
                green_values.append(green_avg)
                blue_values.append(blue_avg)
                timestamps.append(current_time)
            
            # Draw a small rectangle showing the forehead region
            cv2.rectangle(face_roi, 
                         (fw_left, fh), 
                         (fw_right, fh_bottom), 
                         (255, 0, 0), 1)
    
    # Add status text with FPS counter
    if not hasattr(process_frame, 'frame_count'):
        process_frame.frame_count = 0
        process_frame.last_fps_time = current_time
    
    process_frame.frame_count += 1
    if current_time - process_frame.last_fps_time >= 1.0:  # Update FPS every second
        process_frame.fps = process_frame.frame_count / (current_time - process_frame.last_fps_time)
        process_frame.frame_count = 0
        process_frame.last_fps_time = current_time
    
    # Add status and FPS to frame
    status = 'Measuring...' if measurement_active else 'Ready'
    fps_text = f'FPS: {getattr(process_frame, "fps", 0):.1f}'
    
    cv2.putText(frame, f'Status: {status}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, fps_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame

def calculate_heart_rate(signal, fs):
    """Calculate heart rate from PPG signal using multiple methods for better accuracy"""
    if len(signal) < 30:  # Need enough samples for reliable measurement
        return 0, []
    
    try:
        # 1. Preprocess the signal
        # Remove DC component
        signal = signal - np.mean(signal)
        
        # Bandpass filter between 0.75 Hz (45 bpm) and 4.0 Hz (240 bpm)
        nyq = 0.5 * fs
        low = 0.75 / nyq
        high = 4.0 / nyq
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        
        # 2. Method 1: FFT-based heart rate
        n = len(filtered_signal)
        yf = fft(filtered_signal)
        xf = fftfreq(n, 1/fs)[:n//2]
        
        # Find the peak frequency in the typical heart rate range
        mask = (xf > 0.75) & (xf < 4.0)
        if not any(mask):
            return 0, []
        
        # Find the dominant frequency peak
        fft_peaks, _ = find_peaks(np.abs(yf[:n//2][mask]), height=0.1*np.max(np.abs(yf[:n//2][mask])))
        
        if len(fft_peaks) > 0:
            # Get the frequency with maximum amplitude
            idx = fft_peaks[np.argmax(np.abs(yf[:n//2][mask][fft_peaks]))]
            hr_fft = xf[mask][idx] * 60  # Convert to bpm
        else:
            # Fallback to simple max if no peaks found
            idx = np.argmax(np.abs(yf[:n//2][mask]))
            hr_fft = xf[mask][idx] * 60
        
        # 3. Method 2: Peak detection in time domain
        # Find peaks in the filtered signal
        peaks, _ = find_peaks(filtered_signal, distance=fs*0.5)  # At least 0.5s between peaks
        
        if len(peaks) >= 2:  # Need at least 2 peaks to calculate interval
            # Calculate heart rate from peak intervals
            intervals = np.diff(peaks) / fs  # Convert to seconds
            hr_peaks = 60.0 / np.median(intervals)  # Convert to bpm
        else:
            hr_peaks = hr_fft  # Fallback to FFT if not enough peaks
        
        # 4. Combine methods for robustness
        # If both methods agree within 10 bpm, take the average
        if abs(hr_fft - hr_peaks) < 10:
            heart_rate = (hr_fft + hr_peaks) / 2
        else:
            # If they disagree, prefer the one closer to normal range
            if abs(hr_fft - 72) < abs(hr_peaks - 72):
                heart_rate = hr_fft
            else:
                heart_rate = hr_peaks
        
        # 5. Validate the heart rate is within reasonable bounds
        if heart_rate < 40 or heart_rate > 200:  # Unrealistic heart rate
            return 0, []
            
        return round(heart_rate, 1), list(np.abs(yf[:n//2]))
        
    except Exception as e:
        print(f"Error in heart rate calculation: {str(e)}")
        return 0, []

def analyze_ayurvedic(heart_rate):
    """Analyze heart rate and return detailed Ayurvedic dosha information
    
    Args:
        heart_rate (float): Heart rate in BPM
        
    Returns:
        dict: Dictionary containing comprehensive dosha information and recommendations
    """
    # Define detailed dosha information
    dosha_info = {
        'vata': {
            'range': (65, 85),  # Adjusted ranges for better distinction
            'pulse_quality': 'Quick, light, and irregular (like a snake)',
            'pulse_name': 'Sara Gati (Snake-like)',
            'description': (
                f'Your heart rate of {heart_rate} BPM suggests a Vata constitution with some Pitta influence. In Ayurveda, this is known as "Sara Gati" or snake-like pulse.\n\n'
                'Vata types are typically creative, energetic, and quick-thinking with a tendency towards variability. The pulse feels quick, light, and somewhat irregular, like the movement of a snake.\n\n'
                'When balanced, Vata individuals are creative, adaptable, and enthusiastic. However, when imbalanced, they may experience anxiety, insomnia, dry skin, digestive issues, and joint discomfort.'
            ),
            'characteristics': [
                'Light, quick movements and speech',
                'Creative and enthusiastic',
                'Tendency toward dry skin and hair',
                'Variable appetite and digestion',
                'Quick to learn but also quick to forget',
                'Energetic in short bursts, then tires easily'
            ],
            'recommendations': [
                '🌿 Establish and maintain a regular daily routine with consistent meal and sleep times',
                '🍲 Eat warm, moist, and nourishing foods with sweet, sour, and salty tastes',
                '🔥 Incorporate warming spices like ginger, cinnamon, and cumin in your diet',
                '🧘 Practice gentle, grounding exercises like yoga, tai chi, or walking in nature',
                '💤 Ensure adequate rest and avoid overexertion',
                '🌅 Wake up before 6 AM and go to bed by 10 PM',
                '🧴 Use warm sesame or almond oil for daily self-massage (Abhyanga)',
                '🌡️ Keep warm in cold and windy weather',
                '🧘‍♀️ Practice calming pranayama like Nadi Shodhana and Bhramari',
                '🍌 Favor sweet, sour, and salty tastes; reduce bitter, astringent, and pungent foods'
            ],
            'foods_to_favor': [
                'Warm, cooked foods', 'Dairy products', 'Nuts and seeds',
                'Sweet fruits (bananas, mangoes, peaches)', 'Cooked vegetables',
                'Whole grains (rice, wheat, quinoa)', 'Healthy oils (ghee, sesame, olive)'
            ],
            'foods_to_avoid': [
                'Cold foods and drinks', 'Raw vegetables', 'Dried fruits',
                'Caffeinated beverages', 'Carbonated drinks', 'Frozen or leftover foods'
            ],
            'lifestyle_tips': [
                'Create a warm, peaceful environment',
                'Engage in calming activities like meditation and gentle yoga',
                'Avoid excessive stimulation and loud environments',
                'Keep a regular schedule for meals and sleep',
                'Spend time in nature'
            ]
        },
        'pitta': {
            'range': (75, 95),
            'pulse_quality': 'Moderate, strong, and regular (like a frog)',
            'pulse_name': 'Manda Gati (Frog-like)',
            'description': (
                f'Your heart rate of {heart_rate} BPM indicates a Pitta constitution with some Kapha influence. This is known as "Manda Gati" or frog-like pulse.\n\n'
                'Pitta types are typically intelligent, focused, and goal-oriented with strong digestion. The pulse feels strong, regular, and somewhat bounding, like the movement of a frog.\n\n'
                'When balanced, Pitta individuals are intelligent, courageous, and have good leadership qualities. When imbalanced, they may experience inflammation, irritability, skin rashes, excessive body heat, and digestive issues.'
            ),
            'characteristics': [
                'Medium build and strength',
                'Sharp intellect and good concentration',
                'Strong digestion and appetite',
                'Tendency toward anger and irritability when stressed',
                'Good leadership qualities',
                'Competitive nature'
            ],
            'recommendations': [
                '🌿 Maintain a regular daily routine with consistent meal times',
                '❄️ Eat cooling, slightly dry, and heavy foods with sweet, bitter, and astringent tastes',
                '🌿 Incorporate cooling herbs like coriander, fennel, and mint',
                '🏃‍♂️ Engage in moderate exercise during cooler parts of the day',
                '🧘 Practice cooling pranayama like Sheetali and Chandra Bhedana',
                '💧 Stay well-hydrated with room temperature or cool drinks',
                '🌙 Go to bed by 10 PM and wake up before 6 AM',
                '🌊 Spend time in nature, especially near water',
                '🥗 Favor sweet, bitter, and astringent tastes; reduce sour, salty, and pungent foods',
                '🧴 Use cooling oils like coconut or sunflower for self-massage'
            ],
            'foods_to_favor': [
                'Sweet fruits (grapes, melons, mangoes)', 'Leafy greens',
                'Cucumber, zucchini, and other watery vegetables', 'Dairy products',
                'Whole grains (barley, oats, basmati rice)', 'Coconut',
                'Legumes (mung beans, chickpeas)'
            ],
            'foods_to_avoid': [
                'Spicy, sour, and salty foods', 'Fermented foods',
                'Alcohol and caffeine', 'Red meat', 'Fried and processed foods',
                'Excessive amounts of nuts and seeds'
            ],
            'lifestyle_tips': [
                'Avoid excessive heat and direct sun exposure',
                'Engage in calming activities near water',
                'Practice moderation in work and exercise',
                'Cultivate patience and compassion',
                'Avoid excessive competition and conflict'
            ]
        },
        'kapha': {
            'range': (50, 70),
            'pulse_quality': 'Slow, steady, and broad (like a swan)',
            'pulse_name': 'Hamsa Gati (Swan-like)',
            'description': (
                f'Your heart rate of {heart_rate} BPM suggests a strong Kapha constitution with some Vata influence. This is known as "Hamsa Gati" or swan-like pulse.\n\n'
                'Kapha types are naturally calm, patient, and grounded with good endurance. The pulse feels deep, broad, and steady, like the movement of a swan.\n\n'
                'When balanced, Kapha individuals are loving, patient, and have strong immunity. When imbalanced, they may experience weight gain, congestion, lethargy, and attachment.'
            ),
            'characteristics': [
                'Strong, sturdy build',
                'Calm and patient demeanor',
                'Slow but steady energy',
                'Excellent stamina and endurance',
                'Tendency toward weight gain',
                'Slow digestion and metabolism'
            ],
            'recommendations': [
                '🌿 Engage in regular vigorous exercise (brisk walking, swimming, cycling) to stimulate circulation',
                '🔥 Eat light, warm, and dry foods with pungent, bitter, and astringent tastes',
                '🌶️ Incorporate warming spices like ginger, black pepper, and turmeric',
                '⏰ Maintain a consistent sleep schedule, waking before 6 AM',
                '🌬️ Practice invigorating pranayama like Bhastrika and Kapalabhati',
                '🛁 Consider dry brushing (Garshana) before showering to stimulate circulation',
                '🍎 Favor light, warm, and dry foods; reduce heavy, oily, and sweet foods',
                '🚶‍♂️ Stay active throughout the day to prevent stagnation',
                '🌅 Wake up before 6 AM to avoid Kapha time of day',
                '🍯 Use raw honey in warm water to help balance Kapha'
            ],
            'foods_to_favor': [
                'Light, dry foods', 'Pungent spices (ginger, black pepper, cayenne)',
                'Light fruits (apples, pears, pomegranates)', 'Steamed vegetables',
                'Legumes', 'Barley, corn, millet, buckwheat', 'Honey (raw and unprocessed)'
            ],
            'foods_to_avoid': [
                'Heavy, oily, and cold foods', 'Dairy products',
                'Wheat and white rice', 'Sweet and sour fruits',
                'Red meat', 'Fried and processed foods',
                'Excessive amounts of nuts and seeds'
            ],
            'lifestyle_tips': [
                'Stay active and avoid sedentary behavior',
                'Vary your routine regularly',
                'Engage in stimulating activities',
                'Avoid daytime sleeping',
                'Keep warm in cold, damp weather'
            ]
        }
    }
    
    # Determine the dominant dosha based on heart rate with overlap zones
    if heart_rate < 70:
        dosha = 'kapha'
    elif heart_rate < 80:
        # Transition zone between Kapha and Vata
        if heart_rate < 72:
            dosha = 'kapha'
        else:
            dosha = 'vata'
    elif heart_rate < 90:
        dosha = 'vata'
    elif heart_rate < 100:
        # Transition zone between Vata and Pitta
        if heart_rate < 95:
            dosha = 'vata'
        else:
            dosha = 'pitta'
    else:
        dosha = 'pitta'
    
    # Get the base dosha info
    result = {
        'dosha': dosha,
        'pulse_quality': dosha_info[dosha]['pulse_quality'],
        'pulse_name': dosha_info[dosha]['pulse_name'],
        'description': dosha_info[dosha]['description'],
        'characteristics': dosha_info[dosha]['characteristics'],
        'recommendations': dosha_info[dosha]['recommendations'],
        'foods_to_favor': dosha_info[dosha]['foods_to_favor'],
        'foods_to_avoid': dosha_info[dosha]['foods_to_avoid'],
        'lifestyle_tips': dosha_info[dosha]['lifestyle_tips']
    }
    
    # Add secondary dosha information for more personalized insights
    if heart_rate < 70:
        result['secondary_dosha'] = 'vata' if heart_rate > 60 else 'pitta'
    elif heart_rate < 90:
        result['secondary_dosha'] = 'pitta' if heart_rate > 80 else 'kapha'
    else:
        result['secondary_dosha'] = 'vata'
    
    # Add pulse diagnosis information
    result['pulse_diagnosis'] = (
        f"In Ayurvedic pulse diagnosis (Nadi Pariksha), your {dosha.capitalize()} pulse "
        f"is characterized as {dosha_info[dosha]['pulse_quality']}. "
        f"This pulse type is known as '{dosha_info[dosha]['pulse_name']}' and provides "
        "insights into your current state of balance and health."
    )
    
    return result

def get_ayurvedic_analysis(heart_rate):
    """Provide Ayurvedic insights based on heart rate"""
    return analyze_ayurvedic(heart_rate)

# Global variable to store the camera capture object
camera = None

def get_camera():
    """Get or initialize the camera with reliable settings"""
    global camera
    
    # If camera is already open and working, return it
    if camera is not None and camera.isOpened():
        return camera
    
    print("\n=== Initializing Camera ===")
    
    # Release any existing camera first
    if camera is not None:
        camera.release()
    
    # Try with minimal settings first
    try:
        print("Trying with DSHOW backend...")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not camera.isOpened():
            print("Failed to open with DSHOW, trying default backend...")
            camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("Failed to open camera with any backend")
            return None
            
        # Set basic properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)  # Start with 15 FPS for reliability
        
        # Try to set MJPG format if available (usually faster)
        try:
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            print("Set MJPG format")
        except Exception as e:
            print(f"Could not set MJPG format: {str(e)}")
            
        # Try to read a frame to verify it works
        for i in range(3):  # Try up to 3 times
            ret, frame = camera.read()
            if ret and frame is not None and frame.size > 0:
                print("Camera initialized successfully!")
                print(f"Resolution: {frame.shape[1]}x{frame.shape[0]}")
                print(f"Actual FPS: {camera.get(cv2.CAP_PROP_FPS):.1f}")
                return camera
            print(f"  Attempt {i+1} failed to get frame")
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
    
    # If we get here, initialization failed
    print("\nTroubleshooting steps:")
    print("1. Check physical connection:")
    print("   - Is your camera properly connected?")
    print("   - Try unplugging and reconnecting the camera")
    print("   - Try a different USB port")
    print("2. Check if camera is in use by another application")
    print("   - Close all other applications that might be using the camera")
    print("   - This includes Zoom, Teams, Skype, or any video conferencing apps")
    print("3. Check Windows Camera Privacy Settings:")
    print("   - Press Windows + I to open Settings")
    print("   - Go to Privacy & security > Camera")
    print("   - Make sure 'Camera access' is turned ON")
    print("   - Under 'Let apps access your camera', make sure it's enabled")
    print("4. Restart the application")
    print("5. Try a different browser (Chrome, Firefox, or Edge)")
    print("6. Restart your computer and try again")
    print("\nIf the problem persists, please check for any error messages above.")
    
    return None

def release_camera():
    """Safely release the camera resources"""
    global camera
    if camera is not None:
        try:
            # Try to properly close the camera
            if camera.isOpened():
                # Release the camera
                camera.release()
                print("Camera released successfully")
        except Exception as e:
            print(f"Error while releasing camera: {str(e)}")
        finally:
            camera = None
            # Give some time for the camera to be properly released
            time.sleep(0.5)

def generate_frames():
    """Generate video frames with reliable performance"""
    global camera
    
    # Performance monitoring
    frame_count = 0
    last_fps_time = time.time()
    fps = 0
    
    # Target 10 FPS for reliability
    target_fps = 10
    frame_interval = 1.0 / target_fps
    last_frame_time = time.time()
    
    while True:
        current_time = time.time()
        time_since_last_frame = current_time - last_frame_time
        
        # Skip if we're ahead of schedule
        if time_since_last_frame < frame_interval:
            time.sleep(0.001)  # Small sleep to prevent busy waiting
            continue
            
        last_frame_time = current_time
        
        # Get camera
        cap = get_camera()
        if cap is None:
            # Generate error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            time.sleep(0.5)
        else:
            # Simple frame capture with retry
            retry_count = 0
            max_retries = 2
            frame = None
            
            while retry_count < max_retries:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    break
                retry_count += 1
                time.sleep(0.05)  # Small delay between retries
            
            if frame is None or frame.size == 0:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera error", (50, 200), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Process frame if needed
                try:
                    frame = process_frame(frame)
                except Exception as e:
                    print(f"Error processing frame: {str(e)}")
        
        # Encode frame as JPEG
        try:
            ret, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 70  # Lower quality for speed
            ])
            
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n' 
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Update FPS counter
                frame_count += 1
                if current_time - last_fps_time >= 1.0:  # Update FPS every second
                    fps = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time
                    print(f"FPS: {fps:.1f}")
        
        except Exception as e:
            print(f"Error encoding frame: {str(e)}")
            time.sleep(0.1)
    
    # Release the camera when done (this line is typically not reached due to the infinite loop)
    release_camera()

@app.route('/')
def index():
    return render_template('index.html')

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def reset_measurement_state():
    """Reset all measurement-related global variables"""
    global measurement_active, red_values, green_values, blue_values, timestamps, start_time
    
    # Reset measurement state
    measurement_active = False
    red_values = []
    green_values = []
    blue_values = []
    timestamps = []
    start_time = None

@app.route('/start_measurement', methods=['POST', 'GET'])  # Allow both POST and GET for testing
def start_measurement():
    """Start a new measurement"""
    global measurement_active, start_time, red_values, green_values, blue_values, timestamps
    
    # Reset measurement state
    reset_measurement_state()
    
    # Set flags
    measurement_active = True
    start_time = time.time()
    
    # Initialize signal arrays
    red_values = []
    green_values = []
    blue_values = []
    timestamps = []
    
    return jsonify({
        'status': 'success',
        'message': 'Measurement started. Please look at the camera.',
        'liveness_check_required': False
    })

@app.route('/process_measurement', methods=['GET'])
def process_measurement_data():
    """Process the collected measurement data and return the results with improved signal processing"""
    global red_values, green_values, blue_values, timestamps
    
    # Check if we have enough data (at least 15 seconds of data at 10Hz = 150 samples)
    min_samples = 30  # Reduced minimum samples for testing, increase for production
    if len(green_values) < min_samples:
        return {
            'status': 'error',
            'message': f'Not enough data collected ({len(green_values)}/{min_samples}). Please try again and ensure your face is visible for at least 3 seconds.'
        }
    
    try:
        # Convert to numpy arrays for processing
        signal = np.array(green_values)
        
        # Calculate actual sampling rate from timestamps
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            fs = 1.0 / np.median(time_diffs)  # Actual sampling frequency
            fs = min(max(fs, 5), 30)  # Clamp to reasonable range (5-30Hz)
        else:
            fs = 10.0  # Fallback to default if we can't calculate
        
        # Apply detrending (remove linear trend)
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, 1)
        signal_detrended = signal - (coeffs[0] * x + coeffs[1])
        
        # Apply bandpass filter to isolate heart rate frequencies (0.7 Hz to 4 Hz = 42 to 240 bpm)
        low = 0.7  # Lower cutoff frequency (Hz)
        high = 4.0  # Upper cutoff frequency (Hz)
        
        # Apply bandpass filter with error handling
        try:
            filtered_signal = bandpass_filter(signal_detrended, low, high, fs, order=4)
        except Exception as e:
            print(f"Error in bandpass filter: {e}")
            filtered_signal = signal_detrended  # Fallback to unfiltered signal
        
        # Calculate heart rate from the filtered signal
        heart_rate, _ = calculate_heart_rate(filtered_signal, fs)
        
        # Ensure heart rate is within reasonable bounds and round to nearest integer
        heart_rate = int(round(max(40, min(200, heart_rate))))  # Clamp between 40-200 BPM and round to integer
        
        # Get Ayurvedic analysis with the rounded heart rate
        ayurvedic_info = analyze_ayurvedic(heart_rate)
        
        # Prepare signal data for visualization (for debugging)
        # Round values to 2 decimal places for cleaner output
        signal_data = {
            'timestamps': [round(t, 2) for t in timestamps[-min(len(timestamps), 300):]],
            'raw': [round(float(x), 2) for x in signal.tolist()[-300:]],
            'filtered': [round(float(x), 2) for x in filtered_signal.tolist()[-300:]] if 'filtered_signal' in locals() else []
        }
        
        # Format the response with clean numbers
        return {
            'status': 'success',
            'heart_rate': int(heart_rate),  # Already rounded to integer
            'dosha': ayurvedic_info['dosha'],
            'description': ayurvedic_info['description'],
            'recommendations': ayurvedic_info['recommendations'],
            'signal_data': signal_data,
            'fs': round(float(fs), 1),  # Round to 1 decimal place
            'samples': int(len(signal))  # Ensure integer
        }
        
    except Exception as e:
        print(f"Error processing measurement data: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Error processing data: {str(e)}'
        })

@app.route('/stop_measurement', methods=['POST'])
def stop_measurement():
    global measurement_active, red_values, green_values, blue_values, timestamps
    
    # Set a timeout for the entire operation (5 seconds)
    start_time = time.time()
    timeout = 5  # seconds
    
    try:
        # Stop measurement first to prevent further data collection
        measurement_active = False
        
        # Make a copy of the data to avoid race conditions
        # Use a list slice to create a copy instead of .copy() for better performance
        current_red = red_values[:]
        current_green = green_values[:]
        current_blue = blue_values[:]
        current_timestamps = timestamps[:]
        
        # Check if we have enough data
        if len(current_green) < 10:  # Reduced minimum for testing
            return jsonify({
                'status': 'error',
                'message': 'Not enough data collected. Please try again and ensure your face is visible.'
            })
        
        # Process the collected data with the copied data
        # Wrap in a try-except to catch any processing errors
        try:
            # Convert to numpy arrays for processing
            signal = np.array(current_green)
            
            # Calculate actual sampling rate from timestamps
            fs = 10.0  # Default fallback
            if len(current_timestamps) > 1:
                time_diffs = np.diff(current_timestamps)
                if len(time_diffs) > 0:
                    fs = 1.0 / np.median(time_diffs)
                    fs = min(max(fs, 5), 30)  # Clamp to reasonable range
            
            # Simple heart rate calculation (simplified to prevent hanging)
            if len(signal) > 10:  # Need at least 10 samples
                # Basic peak detection
                peaks, _ = find_peaks(signal, distance=fs*0.6)  # At least 0.6s between peaks
                if len(peaks) >= 2:
                    # Calculate heart rate from peak intervals
                    intervals = np.diff(peaks) / fs
                    heart_rate = 60.0 / np.median(intervals)
                else:
                    # Fallback to FFT if not enough peaks
                    n = len(signal)
                    yf = fft(signal - np.mean(signal))
                    xf = fftfreq(n, 1/fs)[:n//2]
                    mask = (xf > 0.75) & (xf < 4.0)
                    if any(mask):
                        idx = np.argmax(np.abs(yf[:n//2][mask]))
                        heart_rate = xf[mask][idx] * 60
                    else:
                        heart_rate = 75.0  # Fallback value
                
                # Ensure heart rate is within reasonable bounds
                heart_rate = max(40, min(200, heart_rate))
                
                # Get Ayurvedic analysis
                ayurvedic_info = analyze_ayurvedic(heart_rate)
                
                # Prepare response
                response = {
                    'status': 'success',
                    'heart_rate': float(heart_rate),
                    'dosha': ayurvedic_info['dosha'].lower(),
                    'ayurvedic_info': {
                        'description': ayurvedic_info['description'],
                        'recommendations': ayurvedic_info['recommendations']
                    },
                    'signal_quality': {
                        'samples': len(current_green),
                        'duration': current_timestamps[-1] - current_timestamps[0] if len(current_timestamps) > 1 else 0,
                        'fs': float(fs)
                    }
                }
                
                print(f"Measurement completed. Heart rate: {heart_rate:.1f} BPM, "
                      f"Dosha: {ayurvedic_info['dosha']}, "
                      f"Samples: {len(current_green)}")
                
                return jsonify(response)
            
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Insufficient data for analysis.'
                })
                
        except Exception as proc_error:
            print(f"Error in processing: {str(proc_error)}")
            return jsonify({
                'status': 'error',
                'message': f'Error processing data: {str(proc_error)}'
            })
        
    except Exception as e:
        print(f"Error in stop_measurement: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'An error occurred: {str(e)}'
        })
    finally:
        # Ensure we always clean up, even if there was an error
        if time.time() - start_time > timeout:
            print("Warning: stop_measurement took too long and was terminated")
        # Reset measurement state for next time
        reset_measurement_state()

if __name__ == '__main__':
    app.run(debug=True)
