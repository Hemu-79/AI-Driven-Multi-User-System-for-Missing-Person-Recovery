"""
CCTV AI - Missing Person Detection System (Streamlit)
Using YOLO for person detection + Face Recognition for identification
Fast, accurate, and scalable
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import time
# import face_recognition  # Removed due to dlib dependency
from ultralytics import YOLO
from loguru import logger
from insightface.app import FaceAnalysis
from enhanced_detector import MissingPersonDetector

# Configure logger
if not globals().get("_STREAMLIT_LOGGER_CONFIGURED", False):
    logger.add("logs/streamlit_app.log", rotation="10 MB")
    _STREAMLIT_LOGGER_CONFIGURED = True

# Page configuration
st.set_page_config(
    page_title="CCTV AI - Missing Person Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize directories
os.makedirs("Database", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("found", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Database initialization
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect("Database/data.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS missing_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            missing_state TEXT,
            missing_city TEXT,
            pincode INTEGER,
            missing_date TEXT,
            description TEXT,
            image_f TEXT,
            complaint_name TEXT,
            complaint_phone TEXT,
            complaint_address TEXT,
            footage_path TEXT,
            status INTEGER DEFAULT 0,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn, cur


def _enhance_low_light_bgr(image_bgr):
    """Apply a lightweight contrast enhancement for low-light reference images."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _uploaded_to_bgr(uploaded, enhance=False):
    img = Image.open(uploaded)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if enhance:
        bgr = _enhance_low_light_bgr(bgr)
    return bgr


def _contains_face_opencv(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


def _get_person_reference_counts(db_path="data"):
    conn, cur = init_database()
    cur.execute("SELECT id FROM missing_people WHERE status IN (0, 1)")
    ids = [str(r["id"]) for r in cur.fetchall()]
    conn.close()

    import glob
    counts = {}
    for pid in ids:
        counts[pid] = len(glob.glob(f"{db_path}/{pid}_*.jpg"))
    return counts

# Initialize YOLO model for person detection
@st.cache_resource
def load_yolo_model():
    """Load YOLO model for person detection"""
    try:
        model = YOLO('yolov8n.pt')  # Nano model for speed
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO: {e}")
        return None

class FaceDetector:
    """Face detection and recognition using InsightFace embeddings"""
    
    def __init__(self):
        self.known_person_ids = {}
        self.known_embeddings = {}
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.load_known_faces()

    @staticmethod
    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return vec
        return vec / norm
    
    def load_known_faces(self):
        """Load all known faces from database"""
        try:
            conn, cur = init_database()
            # Only load faces for people who are currently registered and not yet found/verified
            cur.execute("SELECT id, name, image_f FROM missing_people WHERE status IN (0, 1)")
            records = cur.fetchall()
            
            import glob
            
            for record in records:
                person_id = record['id']
                self.known_embeddings[person_id] = []
                
                # Find all images for this person
                image_paths = glob.glob(f"./data/{person_id}_*.jpg")
                if not image_paths:
                    # Fallback to single image logic if no batch images found
                    image_path = record['image_f']
                    if image_path and os.path.exists(image_path):
                        image_paths = [image_path]
                
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        if image is None:
                            continue

                        faces = self.face_app.get(image)
                        if not faces:
                            continue

                        best_face = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        )
                        emb = self._normalize(best_face.embedding.astype(np.float32))
                        self.known_embeddings[person_id].append(emb)
                
                if self.known_embeddings[person_id]:
                    self.known_person_ids[person_id] = {
                        'name': record['name'],
                        'id': person_id
                    }
                    logger.info(f"Loaded {len(self.known_embeddings[person_id])} embedding(s) for person {person_id}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
    
    def detect_faces_in_frame(self, frame):
        """Detect all faces in a frame"""
        try:
            return self.face_app.get(frame)
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def match_face(self, embedding, threshold=0.45):
        """Match a face embedding against known embeddings"""
        if not self.known_embeddings:
            return None, 0.0
        
        try:
            emb = self._normalize(embedding.astype(np.float32))
            best_pid = None
            best_score = -1.0

            for pid, refs in self.known_embeddings.items():
                if not refs:
                    continue
                score = max(float(np.dot(emb, ref)) for ref in refs)
                if score > best_score:
                    best_score = score
                    best_pid = pid

            if best_pid is not None and best_score >= threshold:
                return best_pid, best_score
            
        except Exception as e:
            logger.error(f"Error matching face: {e}")
        
        return None, 0.0

class PersonDetector:
    """YOLO-based person detection"""
    
    def __init__(self):
        self.yolo_model = load_yolo_model()
        self.face_detector = FaceDetector()
    
    def detect_persons_in_frame(self, frame):
        """Detect persons using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, classes=[0], verbose=False)  # Class 0 is 'person'
            persons = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:  # Confidence threshold
                        persons.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                        })
            
            return persons
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def process_frame(self, frame):
        """Process frame: detect persons, then faces"""
        detections = []
        
        # First detect persons with YOLO
        persons = self.detect_persons_in_frame(frame)
        
        # For each person, detect and match faces
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            
            # Extract person region
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            
            # Detect faces in person region
            face_locations = self.face_detector.detect_faces_in_frame(person_crop)
            
            for face in face_locations:
                fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
                fw = fx2 - fx1
                fh = fy2 - fy1
                if fw < 80 or fh < 80:
                    continue

                # Adjust coordinates to full frame
                face_left = x1 + fx1
                face_top = y1 + fy1
                face_right = x1 + fx2
                face_bottom = y1 + fy2
                
                # Match face
                person_id, confidence = self.face_detector.match_face(face.embedding)
                
                if person_id:
                    detections.append({
                        'person_id': person_id,
                        'person_name': self.face_detector.known_person_ids[person_id]['name'],
                        'confidence': confidence,
                        'face_bbox': (face_left, face_top, face_right, face_bottom),
                        'person_bbox': person['bbox']
                    })
        
        return detections

def save_detection(person_id, frame, detection, source_name):
    """Save detection to database and disk"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"found/{person_id}/{source_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full frame
        frame_path = f"{save_dir}/{timestamp}_full.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Save face crop
        x1, y1, x2, y2 = detection['face_bbox']
        face_crop = frame[y1:y2, x1:x2]
        face_path = f"{save_dir}/{timestamp}_face.jpg"
        cv2.imwrite(face_path, face_crop)
        
        # Update database
        conn, cur = init_database()
        cur.execute("UPDATE missing_people SET status=1 WHERE id=? AND status=0", (person_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Detection saved for person {person_id}")
        return frame_path, face_path
        
    except Exception as e:
        logger.error(f"Error saving detection: {e}")
        return None, None


def infer_footage_config(video_path):
    """Auto-tune footage detection settings based on video duration, resolution, and quality."""
    profile_defaults = {
        'Fast': {
            'profile': 'Fast',
            'surity': 2,
            'frame_time_gap': 6,
            'similarity_threshold': 0.54,
            'process_every_n_frames': 1,
            'min_face_size': 72,
            'blur_threshold': 65.0,
            'vote_window_seconds': 5,
            'det_size': (512, 512),
            'enable_multiscale_retry': False,
            'multiscale_retry_every_n_frames': 14,
        },
        'Balanced': {
            'profile': 'Balanced',
            'surity': 3,
            'frame_time_gap': 5,
            'similarity_threshold': 0.50,
            'process_every_n_frames': 1,
            'min_face_size': 64,
            'blur_threshold': 55.0,
            'vote_window_seconds': 6,
            'det_size': (640, 640),
            'enable_multiscale_retry': True,
            'multiscale_retry_every_n_frames': 10,
        },
        'Accurate': {
            'profile': 'Accurate',
            'surity': 3,
            'frame_time_gap': 5,
            'similarity_threshold': 0.46,
            'process_every_n_frames': 1,
            'min_face_size': 56,
            'blur_threshold': 45.0,
            'vote_window_seconds': 8,
            'det_size': (800, 800),
            'enable_multiscale_retry': True,
            'multiscale_retry_every_n_frames': 6,
        },
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cfg = dict(profile_defaults['Balanced'])
        return cfg, "Balanced fallback (unable to read video metadata)"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (total_frames / fps) if fps > 1e-6 and total_frames > 0 else 0.0

    sample_stats = []
    sample_limit = 12
    # Sequential sampling avoids expensive random-seek overhead on many CCTV codecs.
    sequential_step = max(1, (total_frames // sample_limit) if total_frames > 0 else 10)
    frame_idx = 0
    while len(sample_stats) < sample_limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % sequential_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sample_stats.append((brightness, sharpness))
        frame_idx += 1

    cap.release()
    if sample_stats:
        avg_brightness = float(np.mean([x[0] for x in sample_stats]))
        avg_sharpness = float(np.mean([x[1] for x in sample_stats]))
    else:
        avg_brightness = 110.0
        avg_sharpness = 70.0

    high_res = (width * height) >= (1280 * 720)
    long_video = duration_sec >= 120.0
    low_light_or_blur = (avg_brightness < 60.0) or (avg_sharpness < 50.0)

    if low_light_or_blur:
        chosen = 'Accurate'
        reason = "Auto-selected Accurate (low-light/blurred footage)"
    elif high_res or long_video:
        chosen = 'Fast'
        reason = "Auto-selected Fast (high-resolution/long footage)"
    else:
        chosen = 'Balanced'
        reason = "Auto-selected Balanced (normal footage quality)"

    cfg = dict(profile_defaults[chosen])
    return cfg, reason

def process_video_file(video_path, progress_callback=None, footage_config=None):
    """Process uploaded video file using enhanced detector tuned for CCTV footage."""
    detections_log = []
    if progress_callback:
        progress_callback(1)

    auto_cfg, auto_reason = infer_footage_config(video_path)
    cfg = dict(auto_cfg)
    if footage_config:
        cfg.update({k: v for k, v in footage_config.items() if k in cfg})

    logger.info("Footage auto-config: %s | reason=%s", cfg.get('profile', 'Balanced'), auto_reason)
    if progress_callback:
        progress_callback(5)
    
    # Snapshot files before run so we can list only new detections from this processing job.
    before_files = set(str(p) for p in Path("found").rglob("*.jpg"))

    provider_info = "unknown"

    progress_state = {"last": 0.0}

    def monotonic_progress(pct):
        pct = float(max(0.0, min(100.0, pct)))
        if pct < progress_state["last"]:
            pct = progress_state["last"]
        progress_state["last"] = pct
        if progress_callback:
            progress_callback(pct)

    def stage_callback(stage_start, stage_end):
        def _cb(local_pct):
            local = float(max(0.0, min(100.0, local_pct)))
            mapped = stage_start + (stage_end - stage_start) * (local / 100.0)
            monotonic_progress(mapped)
        return _cb

    def run_once(local_cfg, stage_start=5.0, stage_end=90.0):
        nonlocal provider_info
        detector = None
        try:
            detector = MissingPersonDetector(
                db_path='data',
                surity=local_cfg['surity'],
                frame_time_gap=local_cfg['frame_time_gap'],
                similarity_threshold=local_cfg['similarity_threshold'],
                process_every_n_frames=local_cfg['process_every_n_frames'],
                min_face_size=local_cfg['min_face_size'],
                blur_threshold=local_cfg['blur_threshold'],
                vote_window_seconds=local_cfg['vote_window_seconds'],
                det_size=local_cfg['det_size'],
                enable_multiscale_retry=local_cfg['enable_multiscale_retry'],
                multiscale_retry_every_n_frames=local_cfg['multiscale_retry_every_n_frames'],
                prefer_gpu=True,
                confidence_margin=0.04,
                enable_focus_refine=True,
            )
            provider_info = ", ".join(getattr(detector, "active_providers", ["unknown"]))
            detector.process_video(
                video_path,
                source_name=Path(video_path).name,
                progress_callback=stage_callback(stage_start, stage_end),
            )
        finally:
            if detector is not None:
                detector.close()

    run_once(cfg, stage_start=5.0, stage_end=88.0)

    if progress_callback:
        progress_callback(100)

    after_files = set(str(p) for p in Path("found").rglob("*.jpg"))
    new_files = sorted(after_files - before_files)
    face_files = [p for p in new_files if p.endswith("_face.jpg")]
    if face_files:
        new_files = face_files

    conn, cur = init_database()
    cur.execute("SELECT id, name FROM missing_people")
    pid_to_name = {str(row['id']): row['name'] for row in cur.fetchall()}
    conn.close()

    for file_path in new_files:
        p = Path(file_path)
        # Expected layout: found/<pid>/<source>/<timestamp>.jpg
        if "found" not in p.parts:
            continue
        found_idx = p.parts.index("found")
        if len(p.parts) <= found_idx + 2:
            continue
        person_id = p.parts[found_idx + 1]
        person_name = pid_to_name.get(str(person_id), f"PID {person_id}")
        full_path = str(p).replace("_face.jpg", "_full.jpg")
        detections_log.append({
            'person_id': int(person_id) if str(person_id).isdigit() else person_id,
            'person_name': person_name,
            'confidence': None,
            'frame_number': None,
            'timestamp': datetime.fromtimestamp(p.stat().st_mtime),
            'frame_path': full_path if os.path.exists(full_path) else str(p),
            'face_path': str(p),
        })

    if not detections_log:
        fallback_cfg = dict(cfg)
        fallback_cfg.update({
            'profile': 'AccurateFallback',
            'similarity_threshold': min(float(cfg['similarity_threshold']), 0.45),
            'process_every_n_frames': 1,
            'min_face_size': min(int(cfg['min_face_size']), 56),
            'blur_threshold': min(float(cfg['blur_threshold']), 45.0),
            'det_size': (800, 800),
            'enable_multiscale_retry': True,
            'multiscale_retry_every_n_frames': 6,
        })
        logger.info("No detections in first pass, running accurate fallback pass")
        run_once(fallback_cfg, stage_start=88.0, stage_end=96.0)
        cfg = fallback_cfg

        after_files_fb = set(str(p) for p in Path("found").rglob("*.jpg"))
        new_files_fb = sorted(after_files_fb - before_files)
        face_files_fb = [p for p in new_files_fb if p.endswith("_face.jpg")]
        if face_files_fb:
            new_files_fb = face_files_fb

        for file_path in new_files_fb:
            p = Path(file_path)
            if "found" not in p.parts:
                continue
            found_idx = p.parts.index("found")
            if len(p.parts) <= found_idx + 2:
                continue
            person_id = p.parts[found_idx + 1]
            person_name = pid_to_name.get(str(person_id), f"PID {person_id}")
            full_path = str(p).replace("_face.jpg", "_full.jpg")
            detections_log.append({
                'person_id': int(person_id) if str(person_id).isdigit() else person_id,
                'person_name': person_name,
                'confidence': None,
                'frame_number': None,
                'timestamp': datetime.fromtimestamp(p.stat().st_mtime),
                'frame_path': full_path if os.path.exists(full_path) else str(p),
                'face_path': str(p),
            })

    cfg["provider_info"] = provider_info
    monotonic_progress(100.0)
    return detections_log, cfg, auto_reason

def main():
    """Main Streamlit app"""
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #ff7f0e;
            margin-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🔍 CCTV AI - Missing Person Detection</div>', unsafe_allow_html=True)
    
    # Initialize database
    conn, cur = init_database()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/cctv.png", width=100)
        st.title("Navigation")
        
        menu = st.radio(
            "Select Option",
            ["📊 Dashboard", "➕ Register Missing Person", "📹 Live Detection", 
             "🎬 Process Footage", "✅ Verify Matches", "🔍 Search Records"]
        )
    
    # Dashboard
    if menu == "📊 Dashboard":
        st.markdown('<div class="sub-header">System Dashboard</div>', unsafe_allow_html=True)
        
        # Statistics
        cur.execute("SELECT COUNT(*) as total FROM missing_people")
        total = cur.fetchone()['total']
        
        cur.execute("SELECT COUNT(*) as pending FROM missing_people WHERE status=1")
        pending = cur.fetchone()['pending']
        
        cur.execute("SELECT COUNT(*) as verified FROM missing_people WHERE status=2")
        verified = cur.fetchone()['verified']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cases", total)
        with col2:
            st.metric("Pending Verification", pending, delta="+new")
        with col3:
            st.metric("Verified Matches", verified)
        
        # Recent activity
        st.subheader("Recent Cases")
        cur.execute("SELECT * FROM missing_people ORDER BY created_at DESC LIMIT 10")
        records = cur.fetchall()
        
        if records:
            df = pd.DataFrame([dict(r) for r in records])
            df = df[['id', 'name', 'age', 'gender', 'missing_city', 'missing_state', 'status']]
            df['status'] = df['status'].map({0: 'Not Detected', 1: 'Pending', 2: 'Verified', 3: 'Failed'})
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No cases registered yet")
    
    # Register Missing Person
    elif menu == "➕ Register Missing Person":
        st.markdown('<div class="sub-header">Register New Missing Person</div>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Missing Person Details")
                name = st.text_input("Full Name *")
                gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
                age = st.number_input("Age *", min_value=1, max_value=120, value=25)
                
                missing_state = st.text_input("Missing State *")
                missing_city = st.text_input("Missing City *")
                pincode = st.text_input("Pincode *")
                missing_date = st.date_input("Missing Date *")
                description = st.text_area("Description")
                
                photo = st.file_uploader("Upload Recent Photo *", type=['jpg', 'jpeg', 'png'])
                additional_photos = st.file_uploader(
                    "Optional: Additional Reference Photos (0-8)",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                )
            
            with col2:
                st.subheader("Complainant Details")
                complaint_name = st.text_input("Complainant Name *")
                complaint_phone = st.text_input("Phone Number *")
                complaint_address = st.text_area("Address *")
                
                st.subheader("Optional")
                footage = st.file_uploader("CCTV Footage (Optional)", type=['mp4', 'avi', 'mkv', 'mov'])
                enhance_refs = st.checkbox("Enhance low-light reference photos", value=True)
            
            submitted = st.form_submit_button("Register", use_container_width=True, type="primary")
            
            if submitted:
                if not all([name, age, missing_state, missing_city, pincode, missing_date, 
                           complaint_name, complaint_phone, complaint_address, photo]):
                    st.error("Please fill all required fields marked with *")
                else:
                    try:
                        if additional_photos and len(additional_photos) > 8:
                            st.error("Please upload at most 8 additional reference photos.")
                            additional_photos = additional_photos[:8]

                        primary_bgr = _uploaded_to_bgr(photo, enhance=enhance_refs)
                        if not _contains_face_opencv(primary_bgr):
                            st.error("No face detected in the photo. Please upload a clear face photo.")
                        else:
                            # Save footage if provided
                            footage_path = None
                            if footage:
                                footage_path = f"temp/{footage.name}"
                                with open(footage_path, 'wb') as f:
                                    f.write(footage.getbuffer())
                            
                            # Insert into database
                            cur.execute('''
                                INSERT INTO missing_people 
                                (name, gender, age, missing_state, missing_city, pincode, 
                                 missing_date, description, image_f, complaint_name, 
                                 complaint_phone, complaint_address, footage_path, face_encoding, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                            ''', (name, gender, age, missing_state, missing_city, pincode,
                                 str(missing_date), description, "", complaint_name,
                                 complaint_phone, complaint_address, footage_path, "opencv_detected", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                            
                            conn.commit()
                            person_id = cur.lastrowid

                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            saved_paths = []

                            primary_path = f"data/{person_id}_primary_{ts}.jpg"
                            cv2.imwrite(primary_path, primary_bgr)
                            saved_paths.append(primary_path)

                            if additional_photos:
                                for idx, extra in enumerate(additional_photos, start=1):
                                    try:
                                        ref_bgr = _uploaded_to_bgr(extra, enhance=enhance_refs)
                                        if not _contains_face_opencv(ref_bgr):
                                            continue
                                        ref_path = f"data/{person_id}_ref{idx}_{ts}.jpg"
                                        cv2.imwrite(ref_path, ref_bgr)
                                        saved_paths.append(ref_path)
                                    except Exception as ref_err:
                                        logger.warning(f"Skipped one reference photo for {name}: {ref_err}")

                            cur.execute(
                                "UPDATE missing_people SET image_f=?, face_encoding=? WHERE id=?",
                                (primary_path, f"opencv_detected:{len(saved_paths)}_refs", person_id),
                            )
                            conn.commit()

                            st.success(f"✅ Successfully registered {name} with {len(saved_paths)} reference photo(s)")
                            
                            # Reload face detector
                            if 'detector' in st.session_state:
                                st.session_state.detector.face_detector.load_known_faces()
                            
                            logger.info(f"Registered new person: {name}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Registration error: {e}")
    
    # Live Detection
    elif menu == "📹 Live Detection":
        st.markdown('<div class="sub-header">Live CCTV Detection</div>', unsafe_allow_html=True)
        
        st.info("💡 Live camera detection using YOLO + Face Recognition")
        
        camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0)
        
        col1, col2 = st.columns(2)
        start_btn = col1.button("🟢 Start Detection", use_container_width=True)
        stop_btn = col2.button("🔴 Stop Detection", use_container_width=True)
        
        if start_btn:
            st.session_state.detection_running = True
        if stop_btn:
            st.session_state.detection_running = False
        
        if st.session_state.get('detection_running', False):
            try:
                cap = cv2.VideoCapture(camera_id)
                detector = PersonDetector()
                
                frame_placeholder = st.empty()
                detection_log = st.empty()
                
                detections_list = []
                
                while st.session_state.get('detection_running', False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Process frame
                    detections = detector.process_frame(frame)
                    
                    # Draw detections
                    for detection in detections:
                        x1, y1, x2, y2 = detection['face_bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{detection['person_name']} ({detection['confidence']:.2%})"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Add time below the person
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, current_time, (x1, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save detection
                        save_detection(detection['person_id'], frame, detection, f"cam_{camera_id}")
                        detections_list.append(detection)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Updated to fix deprecation warning: use_container_width=True -> width='stretch'
                    frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                    
                    # Show detection log
                    if detections_list:
                        detection_log.write(f"✅ Detected: {len(detections_list)} matches")
                    
                    time.sleep(0.1)
                
                cap.release()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Live detection error: {e}")
    
    # Process Footage
    elif menu == "🎬 Process Footage":
        st.markdown('<div class="sub-header">Process CCTV Footage</div>', unsafe_allow_html=True)

        st.caption("Uploaded footage is now auto-optimized for speed and accuracy. No manual threshold setup required.")

        ref_counts = _get_person_reference_counts(db_path="data")
        low_ref_pids = [pid for pid, cnt in ref_counts.items() if cnt < 3]
        if low_ref_pids:
            st.warning(
                "Active Mode "
                
            )
        
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mkv', 'mov'])
        
        if uploaded_file:
            # Save temporarily
            temp_path = f"temp/{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🎬 Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value):
                    progress_bar.progress(int(value))
                    status_text.text(f"Processing: {value:.1f}%")
                
                try:
                    detections, used_cfg, auto_reason = process_video_file(temp_path, update_progress)

                    st.info(
                        f"Auto profile: {used_cfg.get('profile', 'Balanced')} | "
                        f"Frame stride: {used_cfg.get('process_every_n_frames')} (1 = every frame) | "
                        f"Threshold: {used_cfg.get('similarity_threshold'):.2f} | "
                        f"Det size: {used_cfg.get('det_size')[0]} | "
                        f"Provider: {used_cfg.get('provider_info', 'unknown')} | {auto_reason}"
                    )
                    
                    st.success(f"✅ Processing complete! Found {len(detections)} matches")
                    
                    if detections:
                        df = pd.DataFrame(detections)
                        st.dataframe(df[['person_name', 'confidence', 'frame_number', 'timestamp']], 
                                   use_container_width=True)
                        
                        # Show sample detections
                        st.subheader("Sample Detections")
                        cols = st.columns(3)
                        for idx, detection in enumerate(detections[:6]):
                            with cols[idx % 3]:
                                if os.path.exists(detection['face_path']):
                                    st.image(detection['face_path'], caption=f"Face: {detection['person_name']}")
                                if os.path.exists(detection['frame_path']):
                                    st.image(detection['frame_path'], caption="Detection Frame")
                
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Video processing error: {e}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Verify Matches
    elif menu == "✅ Verify Matches":
        st.markdown('<div class="sub-header">Verify Detection Matches</div>', unsafe_allow_html=True)
        
        cur.execute("SELECT * FROM missing_people WHERE status=1")
        pending = cur.fetchall()
        
        if not pending:
            st.info("No pending verifications")
        else:
            for record in pending:
                person_id = record['id']
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 2])
                    
                    with col1:
                        st.subheader(record['name'])
                        if os.path.exists(record['image_f']):
                            st.image(record['image_f'], caption="Registered Photo")
                    
                    with col2:
                        st.write(f"**Age:** {record['age']} | **Gender:** {record['gender']}")
                        st.write(f"**Location:** {record['missing_city']}, {record['missing_state']}")
                        st.write(f"**Contact:** {record['complaint_phone']}")
                        
                        # Show detected images
                        found_dir = f"found/{person_id}"
                        if os.path.exists(found_dir):
                            images = list(Path(found_dir).rglob("*_face.jpg"))
                            if images:
                                st.write("**Detected Images:**")
                                for img_path in images[:3]:
                                    st.image(str(img_path), width=150)
                    
                    with col3:
                        if st.button("✅ Confirm Match", key=f"match_{person_id}", 
                                   use_container_width=True):
                            cur.execute("UPDATE missing_people SET status=2 WHERE id=?", (person_id,))
                            conn.commit()
                            st.success("Match confirmed!")
                            st.rerun()
                        
                        if st.button("❌ Not a Match", key=f"no_match_{person_id}", 
                                   use_container_width=True):
                            cur.execute("UPDATE missing_people SET status=3 WHERE id=?", (person_id,))
                            conn.commit()
                            st.warning("Marked as not a match")
                            st.rerun()
                    
                    st.divider()
    
    # Search Records
    elif menu == "🔍 Search Records":
        st.markdown('<div class="sub-header">Search Records</div>', unsafe_allow_html=True)
        
        search_term = st.text_input("Search by name, city, or state")
        status_filter = st.multiselect("Status", ["Not Detected", "Pending", "Verified", "Failed"])
        
        query = "SELECT * FROM missing_people WHERE 1=1"
        params = []
        
        if search_term:
            query += " AND (name LIKE ? OR missing_city LIKE ? OR missing_state LIKE ?)"
            params.extend([f"%{search_term}%"] * 3)
        
        if status_filter:
            status_map = {"Not Detected": 0, "Pending": 1, "Verified": 2, "Failed": 3}
            status_values = [status_map[s] for s in status_filter]
            query += f" AND status IN ({','.join('?' * len(status_values))})"
            params.extend(status_values)
        
        cur.execute(query, params)
        results = cur.fetchall()
        
        if results:
            st.write(f"Found {len(results)} records")
            
            for record in results:
                with st.expander(f"{record['name']} - {record['missing_city']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(record['image_f']):
                            st.image(record['image_f'])
                    
                    with col2:
                        st.write(f"**ID:** {record['id']}")
                        st.write(f"**Age/Gender:** {record['age']}/{record['gender']}")
                        st.write(f"**Location:** {record['missing_city']}, {record['missing_state']}")
                        st.write(f"**Missing Since:** {record['missing_date']}")
                        st.write(f"**Description:** {record['description']}")
                        st.write(f"**Contact:** {record['complaint_phone']}")
                        
                        status_text = {0: "Not Detected", 1: "Pending", 2: "Verified", 3: "Failed"}
                        st.write(f"**Status:** {status_text[record['status']]}")
                        
                        # Delete option for verified records
                        if record['status'] == 2:
                            if st.button("🗑️ Delete Record", key=f"del_{record['id']}", type="primary"):
                                try:
                                    # Delete associated files (optional but recommended)
                                    if os.path.exists(record['image_f']):
                                        os.remove(record['image_f'])
                                    
                                    # Delete from database
                                    cur.execute("DELETE FROM missing_people WHERE id=?", (record['id'],))
                                    conn.commit()
                                    st.success("Record deleted successfully")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting record: {e}")
        else:
            st.info("No records found")
    
    conn.close()

if __name__ == "__main__":
    main()
