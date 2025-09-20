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

# Liveness detection variables
liveness_check_active = False
liveness_check_passed = False
blink_detected = False
blink_start_time = 0
blink_duration = 0
prev_gray = None

# Initialize face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Fallback eye cascade if the default one fails
eye_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

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

def check_blink(eye_region, gray):
    """Check if a blink is detected in the eye region"""
    try:
        # Expand the eye region slightly to ensure we capture the entire eye
        padding = 5
        x, y, w, h = eye_region
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        eye_roi = gray[y:y+h, x:x+w]
        if eye_roi.size == 0:
            return False
            
        # Apply adaptive thresholding which works better in different lighting
        eye_roi = cv2.equalizeHist(eye_roi)
        _, threshold = cv2.threshold(eye_roi, 35, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, eye is likely closed
        if not contours:
            return True
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea, default=None)
        if largest_contour is not None:
            area = cv2.contourArea(largest_contour)
            # If the area of the largest contour is small, eye is likely closed
            if area < (w * h * 0.3):  # 30% of eye region
                return True
                
    except Exception as e:
        print(f"Error in check_blink: {e}")
        
    return False

def detect_eyes(face_roi, gray_frame):
    """Detect eyes in the face region using multiple cascade classifiers"""
    # Try the primary eye detector first
    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # If no eyes found, try the alternative detector
    if len(eyes) < 2 and eye_cascade_alt:
        eyes = eye_cascade_alt.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
    
    return eyes

def detect_liveness(face_roi, gray_frame):
    """Detect liveness by checking for eye blinks"""
    global blink_detected, blink_start_time, blink_duration
    
    try:
        # Detect eyes in the face region
        eyes = detect_eyes(face_roi, gray_frame)
        
        # If we can't detect eyes, the person might be blinking
        if len(eyes) < 2:
            if not blink_detected:
                blink_detected = True
                blink_start_time = time.time()
                print("Blink started")
            else:
                blink_duration = time.time() - blink_start_time
                print(f"Blink duration: {blink_duration:.2f}s")
                if 0.1 < blink_duration < 2.0:  # Extended max duration to 2 seconds
                    print("Blink detected!")
                    return True
            return False
            
        # Reset blink detection if eyes are visible
        if blink_detected:
            blink_detected = False
            print("Eyes detected, resetting blink detection")
        
        # Check each detected eye
        eyes_closed = 0
        for (ex, ey, ew, eh) in eyes[:2]:  # Only check first two eyes
            eye_region = (ex, ey, ew, eh)
            if check_blink(eye_region, gray_frame):
                eyes_closed += 1
                print(f"Eye closed detected ({eyes_closed}/2)")
        
        # If both eyes are closed
        if eyes_closed >= 2:
            if not blink_detected:
                blink_detected = True
                blink_start_time = time.time()
                print("Both eyes closed, starting blink timer")
            else:
                blink_duration = time.time() - blink_start_time
                print(f"Blink in progress: {blink_duration:.2f}s")
                if 0.1 < blink_duration < 2.0:  # Extended max duration to 2 seconds
                    print("Successful blink detected!")
                    return True
        else:
            if blink_detected:
                print("Eyes opened, resetting blink detection")
                blink_detected = False
            
    except Exception as e:
        print(f"Error in detect_liveness: {e}")
        import traceback
        traceback.print_exc()
        
    return False

def process_frame(frame):
    """Process each frame to extract color channel values with liveness detection"""
    global liveness_check_active, liveness_check_passed, prev_gray, blink_detected, blink_start_time
    
    # Convert frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        color = (0, 255, 0) if liveness_check_passed else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add liveness check status text
        status_text = "Liveness: Verified" if liveness_check_passed else "Liveness: Checking..."
        cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Extract ROI (Region of Interest) - forehead region
        roi = frame[y:y+h//2, x:x+w]
        
        if roi.size > 0 and measurement_active:
            # Perform liveness check if not already passed
            if not liveness_check_passed and liveness_check_active:
                face_roi = gray[y:y+h, x:x+w]
                if face_roi.size > 0:  # Ensure we have a valid face ROI
                    if detect_liveness(face_roi, gray):
                        liveness_check_passed = True
                        liveness_check_active = False
                        blink_detected = False
            
            # Only process signal if liveness check passed
            if liveness_check_passed or not liveness_check_active:
                # Calculate mean values for each channel in the ROI
                b, g, r = cv2.split(roi)
                red_mean = np.mean(r)
                green_mean = np.mean(g)
                blue_mean = np.mean(b)
                
                # Append values to the lists
                red_values.append(red_mean)
                green_values.append(green_mean)
                blue_values.append(blue_mean)
                timestamps.append(time.time() - start_time)
    
    # Add instructions for liveness check
    if liveness_check_active and not liveness_check_passed:
        cv2.putText(frame, "Please blink your eyes", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
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
    """Get or initialize the camera capture object with enhanced error handling"""
    global camera
    
    # If camera is already open and working, return it
    if camera is not None and camera.isOpened():
        return camera
    
    print("\n=== Initializing Camera ===")
    
    # Try different camera indices (0 is usually the default)
    for camera_index in [0, 1, 2]:
        print(f"\nTrying camera index {camera_index}...")
        
        # Try different backends
        for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                print(f"Attempting with backend: {api}")
                
                # Try to open camera with minimal settings first
                camera = cv2.VideoCapture(camera_index, api)
                
                if not camera.isOpened():
                    print(f"Failed to open camera {camera_index} with backend {api}")
                    continue
                
                # Try to read a frame
                ret, frame = camera.read()
                if not ret or frame is None:
                    print(f"Could not read frame from camera {camera_index}")
                    camera.release()
                    continue
                
                print(f"Success! Camera opened at index {camera_index} with backend {api}")
                print(f"Default resolution: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                
                # Try to set preferred settings (but don't fail if they don't work)
                try:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    print(f"Set resolution to: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                except Exception as e:
                    print(f"Warning: Could not set camera properties: {e}")
                
                return camera
                
            except Exception as e:
                print(f"Error with camera {camera_index}, backend {api}: {str(e)[:200]}")
                if camera is not None:
                    camera.release()
                    camera = None
    
    # If we get here, all attempts failed
    error_msg = """
    ====================================================
    CRITICAL: Could not initialize any camera
    ====================================================
    
    TROUBLESHOOTING STEPS:
    1. Check physical connection:
       - Is your camera properly connected?
       - Try unplugging and reconnecting the camera
    
    2. Check if camera is in use:
       - Close all other applications that might be using the camera
       - This includes Zoom, Teams, Skype, or any video conferencing apps
    
    3. Check Windows Camera Privacy Settings:
       - Press Windows + I to open Settings
       - Go to Privacy & security > Camera
       - Make sure "Camera access" is turned ON
       - Under "Let apps access your camera", make sure it's enabled
       - Scroll down and make sure your browser has camera access
    
    4. Try a different browser:
       - Some browsers handle camera access differently
       - Try Chrome, Firefox, or Edge
    
    5. Test with Windows Camera app:
       - Open the built-in Camera app
       - If it doesn't work there, the issue is with your camera/drivers
    
    6. Restart your computer and try again
    
    If the problem persists after trying these steps, please share any error messages.
    """
    print(error_msg)
    raise RuntimeError("Failed to initialize camera. Please check the troubleshooting steps above.")

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
    """Generate video frames from webcam with robust error handling"""
    global camera
    last_error_time = 0
    error_cooldown = 5  # seconds between error messages to avoid spamming
    
    while True:
        current_time = time.time()
        try:
            # Get or initialize camera
            cap = get_camera()
            if cap is None:
                # Only show error occasionally to avoid log spam
                if current_time - last_error_time > error_cooldown:
                    print("Camera not available, retrying...")
                    last_error_time = current_time
                
                # Generate error frame if camera can't be opened
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Camera not detected", (50, 200), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Please check your camera connection", (50, 250), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "and refresh the page", (50, 300), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Read frame from camera with retry logic
                max_retries = 3
                success = False
                frame = None
                
                for attempt in range(max_retries):
                    success, frame = cap.read()
                    if success and frame is not None and frame.size > 0:
                        break
                    print(f"Failed to grab frame (attempt {attempt + 1}/{max_retries})")
                    time.sleep(0.1)
                
                if not success or frame is None or frame.size == 0:
                    print("Failed to grab frame from camera after multiple attempts")
                    # Generate error frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "Camera error", (50, 200), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "Reinitializing camera...", (50, 250), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Try to reinitialize camera on error
                    release_camera()
                    time.sleep(1)  # Give some time before retrying
                    continue
                else:
                    # Process the frame
                    try:
                        frame = process_frame(frame)
                    except Exception as e:
                        print(f"Error processing frame: {str(e)}")
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Processing error", (50, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n' + 
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
        except Exception as e:
            print(f"Critical error in frame generation: {str(e)}")
            # Generate error frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Error: {str(e)[:50]}...", (50, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n' + 
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except:
                pass
                
            # Wait a bit before retrying
            time.sleep(1)
    
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
    global measurement_active, start_time, red_values, green_values, blue_values, timestamps
    global liveness_check_active, liveness_check_passed, blink_detected, blink_start_time
    
    # Reset measurement data
    red_values = []
    green_values = []
    blue_values = []
    timestamps = []
    start_time = time.time()
    
    # Reset liveness check state
    liveness_check_active = True
    liveness_check_passed = False
    blink_detected = False
    blink_start_time = 0
    
    # Start measurement
    measurement_active = True

@app.route('/start_measurement', methods=['POST', 'GET'])  # Allow both POST and GET for testing
def start_measurement():
    """Start a new measurement with liveness check"""
    global measurement_active, start_time, red_values, green_values, blue_values, timestamps
    global liveness_check_active, liveness_check_passed, blink_detected, blink_start_time
    
    try:
        # Reset all measurement state
        reset_measurement_state()
        
        print("Measurement started with liveness check")
        response = {
            'status': 'success',
            'message': 'Measurement started with liveness check',
            'requires_liveness': True,
            'duration': 25  # Let frontend know the expected duration in seconds
        }
        print(f"Sending response: {response}")
        return jsonify(response)
    except Exception as e:
        error_msg = f'Failed to start measurement: {str(e)}'
        print(f"Error in start_measurement: {error_msg}")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

def process_measurement_data():
    """Process the collected measurement data and return the results"""
    global red_values, green_values, blue_values, timestamps
    
    # Check if we have enough data
    if len(green_values) < 10:  # Need at least 10 samples (about 1 second)
        return {
            'status': 'error',
            'message': 'Not enough data collected. Please try again and ensure your face is visible.'
        }
    
    try:
        # Convert to numpy arrays for processing
        green_signal = np.array(green_values)
        
        # Apply bandpass filter to isolate heart rate frequencies (0.7 Hz to 4 Hz = 42 to 240 bpm)
        fs = 10  # Sampling frequency (10 Hz)
        low = 0.7  # Lower cutoff frequency (Hz)
        high = 4.0  # Upper cutoff frequency (Hz)
        
        # Apply bandpass filter
        filtered_signal = bandpass_filter(green_signal, low, high, fs, order=5)
        
        # Calculate heart rate from the filtered signal
        heart_rate, _ = calculate_heart_rate(filtered_signal, fs)
        
        # Get Ayurvedic analysis
        ayurvedic_info = analyze_ayurvedic(heart_rate)
        
        return {
            'status': 'success',
            'heart_rate': heart_rate,
            'dosha': ayurvedic_info['dosha'],
            'description': ayurvedic_info['description'],
            'recommendations': ayurvedic_info['recommendations']
        }
        
    except Exception as e:
        print(f"Error processing measurement data: {e}")
        return {
            'status': 'error',
            'message': f'Error processing data: {str(e)}'
        }

@app.route('/stop_measurement', methods=['POST'])
def stop_measurement():
    global measurement_active, liveness_check_active, liveness_check_passed, blink_detected
    
    try:
        # Stop measurement
        measurement_active = False
        
        # Reset liveness check state
        liveness_check_active = False
        liveness_check_passed = False
        blink_detected = False
        
        # Process the collected data
        result = process_measurement_data()
        
        # If there was an error, return it
        if result['status'] != 'success':
            return jsonify(result)
        
        # Format response to match frontend expectations
        response = {
            'status': 'success',
            'heart_rate': result['heart_rate'],
            'plot': None,  # Add plot data if available
            'dosha': result['dosha'].lower(),  # Ensure lowercase to match frontend CSS
            'ayurvedic_info': {
                'description': result['description'],
                'recommendations': result['recommendations']
            }
        }
            
        print(f"Measurement completed. Heart rate: {result['heart_rate']:.1f} BPM, Dosha: {result['dosha']}")
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in stop_measurement: {e}")
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)
