import cv2
import numpy as np
import uuid
import os
import argparse
import logging
from typing import Tuple, Dict, List, Optional
from datetime import datetime
from pathlib import Path


class ModelConfig:
    """Configuration for the neural network models used in detection."""
    
    def _init_(self):
        """Initialize model configuration with paths to model files."""
        self.age_prototxt = "age_deploy.prototxt"
        self.age_model = "age_net.caffemodel"
        self.gender_prototxt = "gender_deploy.prototxt"
        self.gender_model = "gender_net.caffemodel"
        self.face_prototxt = "deploy.prototxt"
        self.face_model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.mean_values = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
    
    def validate(self) -> bool:
        """Validate that all required model files exist.
        
        Returns:
            bool: True if all files exist, False otherwise
        """
        required_files = [
            self.age_prototxt, self.age_model,
            self.gender_prototxt, self.gender_model,
            self.face_prototxt, self.face_model
        ]
        
        missing_files = [f for f in required_files if not os.path.isfile(f)]
        
        if missing_files:
            for file in missing_files:
                logging.error(f"Missing model file: {file} (should be in the same directory as this script)")
            return False
        return True


class FaceInfo:
    """Stores information about a detected face."""
    
    def _init_(self, bbox: Tuple[int, int, int, int]):
        """Initialize face information.
        
        Args:
            bbox: Bounding box of the face (x, y, w, h)
        """
        self.bbox = bbox
        self.age = None
        self.gender = None
        self.age_conf = 0.0
        self.gender_conf = 0.0
        self.frame_count = 0
        self.last_seen = datetime.now()
        self.stable = False
    
    def update(self, bbox: Tuple[int, int, int, int], age: str, age_conf: float, 
               gender: str, gender_conf: float) -> None:
        """Update face information with new detection data.
        
        Args:
            bbox: New bounding box
            age: Detected age range
            age_conf: Confidence of age prediction
            gender: Detected gender
            gender_conf: Confidence of gender prediction
        """
        self.bbox = bbox
        self.last_seen = datetime.now()
        
        # Update age and gender if confidence is higher
        if age_conf > self.age_conf:
            self.age = age
            self.age_conf = age_conf
        
        if gender_conf > self.gender_conf:
            self.gender = gender
            self.gender_conf = gender_conf
        
        self.frame_count += 1
        
        # Mark as stable after seeing the face for several frames
        if self.frame_count >= 5:
            self.stable = True


class FaceMemory:
    """Manages detected faces across video frames."""
    
    def _init_(self, max_faces: int = 10, ttl_seconds: int = 3):
        """Initialize face memory tracker.
        
        Args:
            max_faces: Maximum number of faces to track simultaneously
            ttl_seconds: Time-to-live in seconds for faces that disappeared
        """
        self.faces: Dict[str, FaceInfo] = {}
        self.max_faces = max_faces
        self.ttl_seconds = ttl_seconds
    
    def cleanup(self) -> None:
        """Remove faces that haven't been seen recently."""
        current_time = datetime.now()
        expired_ids = []
        
        for face_id, face_info in self.faces.items():
            time_diff = (current_time - face_info.last_seen).total_seconds()
            if time_diff > self.ttl_seconds:
                expired_ids.append(face_id)
        
        for face_id in expired_ids:
            del self.faces[face_id]
    
    def assign_id(self, face_box: Tuple[int, int, int, int]) -> str:
        """Assign an ID to a face, either matching existing one or creating new.
        
        Args:
            face_box: Bounding box of the face (x, y, w, h)
            
        Returns:
            str: Face ID
        """
        # Clean up old faces first
        self.cleanup()
        
        x, y, w, h = face_box
        face_center = (x + w // 2, y + h // 2)
        
        # Try to match with existing faces based on position
        best_id = None
        min_distance = float('inf')
        
        for face_id, face_info in self.faces.items():
            fx, fy, fw, fh = face_info.bbox
            center = (fx + fw // 2, fy + fh // 2)
            
            # Calculate Euclidean distance between centers
            distance = np.sqrt((face_center[0] - center[0])*2 + (face_center[1] - center[1])*2)
            
            # Calculate size similarity (ratio of areas)
            area1 = w * h
            area2 = fw * fh
            size_ratio = min(area1, area2) / max(area1, area2)
            
            # Combined metric (lower is better)
            combined_metric = distance * (1 + (1 - size_ratio))
            
            if combined_metric < min_distance and combined_metric < 100:  # Threshold for matching
                min_distance = combined_metric
                best_id = face_id
        
        if best_id:
            return best_id
        
        # If we have too many faces, remove the oldest one
        if len(self.faces) >= self.max_faces:
            oldest_id = min(self.faces, key=lambda fid: self.faces[fid].last_seen)
            del self.faces[oldest_id]
        
        # Create new face
        new_id = str(uuid.uuid4())[:8]
        self.faces[new_id] = FaceInfo(face_box)
        return new_id
    
    def update(self, face_id: str, bbox: Tuple[int, int, int, int], 
               age: str, age_conf: float, gender: str, gender_conf: float) -> None:
        """Update information for a specific face.
        
        Args:
            face_id: Face identifier
            bbox: Bounding box
            age: Detected age range
            age_conf: Confidence of age prediction
            gender: Detected gender
            gender_conf: Confidence of gender prediction
        """
        if face_id in self.faces:
            self.faces[face_id].update(bbox, age, age_conf, gender, gender_conf)
    
    def get_info(self, face_id: str) -> Tuple[Optional[str], Optional[str], bool]:
        """Get age and gender information for a face if available.
        
        Args:
            face_id: Face identifier
            
        Returns:
            Tuple: (age, gender, is_stable)
        """
        if face_id not in self.faces:
            return None, None, False
        
        face = self.faces[face_id]
        return face.age, face.gender, face.stable


class AgeGenderDetector:
    """Detects faces and predicts age and gender from video frames."""
    
    def _init_(self, face_confidence: float = 0.7, display_unstable: bool = False):
        """Initialize the detector with the specified models.
        
        Args:
            face_confidence: Minimum confidence for face detection
            display_unstable: Whether to display unstable detections
        """
        self.config = ModelConfig()
        if not self.config.validate():
            raise ValueError("Invalid model configuration - missing files")
        
        self.face_confidence = face_confidence
        self.display_unstable = display_unstable
        
        # Initialize DNN models
        logging.info("Loading neural network models...")
        try:
            self.age_net = cv2.dnn.readNetFromCaffe(self.config.age_prototxt, self.config.age_model)
            self.gender_net = cv2.dnn.readNetFromCaffe(self.config.gender_prototxt, self.config.gender_model)
            self.face_net = cv2.dnn.readNetFromCaffe(self.config.face_prototxt, self.config.face_model)
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise
        
        # Initialize face tracking
        self.memory = FaceMemory()
        
        # Statistics
        self.frame_count = 0
        self.faces_detected = 0
        self.processing_times = []
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """Detect faces in the frame.
        
        Args:
            frame: Input image
            
        Returns:
            List of tuples containing face bounding boxes and face images
        """
        if frame is None or frame.size == 0:
            return []
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                    self.config.mean_values, swapRB=False)
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.face_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                
                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Skip tiny faces
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                
                face_img = frame[y1:y2, x1:x2]
                faces.append(((x1, y1, x2 - x1, y2 - y1), face_img))
        
        return faces
    
    def predict_age_gender(self, face_img: np.ndarray) -> Tuple[str, float, str, float]:
        """Predict age and gender from a face image.
        
        Args:
            face_img: Image of the face
            
        Returns:
            Tuple: (age, age_confidence, gender, gender_confidence)
        """
        # Resize the face image to the expected input size
        try:
            face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                             self.config.mean_values, swapRB=False)
            
            # Gender prediction
            self.gender_net.setInput(face_blob)
            gender_preds = self.gender_net.forward()
            gender_idx = gender_preds[0].argmax()
            gender = self.config.gender_list[gender_idx]
            gender_conf = float(gender_preds[0][gender_idx])
            
            # Age prediction
            self.age_net.setInput(face_blob)
            age_preds = self.age_net.forward()
            age_idx = age_preds[0].argmax()
            age = self.config.age_list[age_idx]
            age_conf = float(age_preds[0][age_idx])
            
            return age, age_conf, gender, gender_conf
            
        except Exception as e:
            logging.warning(f"Error predicting age/gender: {e}")
            return "Unknown", 0.0, "Unknown", 0.0
    
    def draw_results(self, frame: np.ndarray, face_id: str, bbox: Tuple[int, int, int, int], 
                     age: str, gender: str, stable: bool) -> None:
        """Draw detection results on the frame.
        
        Args:
            frame: Input image
            face_id: Face identifier
            bbox: Bounding box (x, y, w, h)
            age: Detected age range
            gender: Detected gender
            stable: Whether detection is stable
        """
        x, y, w, h = bbox
        
        if stable:
            color = (0, 255, 0)  # Green for stable detections
            label = f"{gender}, {age}"
        elif self.display_unstable:
            color = (0, 165, 255)  # Orange for unstable detections
            label = f"Analyzing... ({gender}, {age})"
        else:
            color = (0, 165, 255)  # Orange for unstable detections
            label = "Analyzing..."
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Create background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 10, y), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x + 5, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw face ID (optional)
        if self.display_unstable:
            cv2.putText(frame, f"ID: {face_id}", (x, y + h + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with detection results
        """
        if frame is None or frame.size == 0:
            return frame
        
        start_time = datetime.now()
        self.frame_count += 1
        
        # Make a copy of the frame for drawing
        output_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        self.faces_detected += len(faces)
        
        # Process each face
        for face_box, face_img in faces:
            # Skip problematic faces
            if face_img.size == 0 or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue
                
            # Get or assign face ID
            face_id = self.memory.assign_id(face_box)
            
            # Predict age and gender
            age, age_conf, gender, gender_conf = self.predict_age_gender(face_img)
            
            # Update face information
            self.memory.update(face_id, face_box, age, age_conf, gender, gender_conf)
            
            # Get current information
            age_txt, gender_txt, stable = self.memory.get_info(face_id)
            
            # Draw results on frame
            if age_txt and gender_txt:
                self.draw_results(output_frame, face_id, face_box, 
                                 age_txt, gender_txt, stable)
        
        # Calculate processing time
        elapsed = (datetime.now() - start_time).total_seconds()
        self.processing_times.append(elapsed)
        
        # Show FPS if we have enough data
        if len(self.processing_times) >= 10:
            avg_time = sum(self.processing_times[-30:]) / min(30, len(self.processing_times))
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return output_frame
    
    def get_statistics(self) -> Dict:
        """Get detection statistics.
        
        Returns:
            Dictionary containing detection statistics
        """
        avg_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        return {
            "frames_processed": self.frame_count,
            "faces_detected": self.faces_detected,
            "avg_processing_time": avg_time,
            "fps": 1.0 / avg_time if avg_time > 0 else 0,
            "active_faces": len(self.memory.faces)
        }


class VideoProcessor:
    """Handles video capture and processing."""
    
    def _init_(self, detector: AgeGenderDetector, 
                 source: int = 0, 
                 output_path: Optional[str] = None,
                 display: bool = True,
                 width: int = 640,
                 height: int = 480):
        """Initialize video processor.
        
        Args:
            detector: Age and gender detector instance
            source: Video source (camera index or file path)
            output_path: Path to save output video
            display: Whether to display output
            width: Frame width
            height: Frame height
        """
        self.detector = detector
        self.source = source
        self.output_path = output_path
        self.display = display
        self.width = width
        self.height = height
        
        self.cap = None
        self.writer = None
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize capture and writer.
        
        Returns:
            bool: Success status
        """
        try:
            # Initialize capture
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logging.error(f"Failed to open video source: {self.source}")
                return False
            
            # Set resolution if specified
            if self.width and self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Initialize video writer if output path is specified
            if self.output_path:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps > 0 else 30
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(
                    self.output_path, fourcc, fps, (frame_width, frame_height)
                )
                
                if not self.writer.isOpened():
                    logging.error(f"Failed to create video writer: {self.output_path}")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error initializing video processor: {e}")
            return False
    
    def release(self) -> None:
        """Release resources."""
        if self.cap:
            self.cap.release()
        
        if self.writer:
            self.writer.release()
        
        if self.display:
            cv2.destroyAllWindows()
    
    def run(self) -> bool:
        """Run the video processor.
        
        Returns:
            bool: Success status
        """
        if not self.initialize():
            return False
        
        self.running = True
        logging.info("Video processing started")
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of video stream")
                    break
                
                # Process frame
                output_frame = self.detector.process(frame)
                
                # Write frame if writer is initialized
                if self.writer:
                    self.writer.write(output_frame)
                
                # Display frame if enabled
                if self.display:
                    cv2.imshow("Age & Gender Detection", output_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logging.info("Processing stopped by user")
                        break
            
            # Print statistics
            stats = self.detector.get_statistics()
            logging.info(f"Statistics: {stats}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error during video processing: {e}")
            return False
            
        finally:
            self.release()


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Set up logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=level,
            format=log_format
        )


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Age and Gender Detection System")
    
    # Input source
    parser.add_argument("-s", "--source", type=str, default="0",
                        help="Video source (camera index or file path, default: 0)")
    
    # Output
    parser.add_argument("-o", "--output", type=str,
                        help="Output video file path")
    
    # Display
    parser.add_argument("--no-display", action="store_true",
                        help="Disable display")
    
    # Resolution
    parser.add_argument("--width", type=int, default=640,
                        help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Frame height (default: 480)")
    
    # Detection parameters
    parser.add_argument("--face-confidence", type=float, default=0.7,
                        help="Face detection confidence threshold (default: 0.7)")
    parser.add_argument("--show-unstable", action="store_true",
                        help="Show unstable detections")
    
    # Logging
    parser.add_argument("--log-file", type=str,
                        help="Log file path")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_file, logging.DEBUG if args.debug else logging.INFO)
    
    try:
        # Convert source to int if it's a number (camera index)
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
        
        # Initialize detector
        detector = AgeGenderDetector(
            face_confidence=args.face_confidence,
            display_unstable=args.show_unstable
        )
        
        # Initialize video processor
        processor = VideoProcessor(
            detector=detector,
            source=source,
            output_path=args.output,
            display=not args.no_display,
            width=args.width,
            height=args.height
        )
        
        # Run processor
        processor.run()
        
    except Exception as e:
        logging.error(f"Unhandled exception: {e}", exc_info=True)
        return 1
    
    return 0


if _name_ == "_main_":
    exit(main())
