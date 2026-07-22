import math

# MediaPipe FaceMesh key indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_CORNERS = [61, 291]
MOUTH_INNER_UP_DOWN = [13, 14]

def euclidean_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def calculate_ear(eye_points, landmarks):
    """Calculate Eye Aspect Ratio"""
    # Vertical distances
    v1 = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    v2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Horizontal distance
    h = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    if h == 0:
        return 0
    return (v1 + v2) / (2.0 * h)

def extract_face_emotion(landmarks):
    """
    Computes a simple emotional valence score (-1.0 to 1.0) based on facial geometry.
    This replaces DeepFace server-side inference for real-time.
    """
    if not landmarks or len(landmarks) < 468:
        return 0.0

    left_ear = calculate_ear(LEFT_EYE, landmarks)
    right_ear = calculate_ear(RIGHT_EYE, landmarks)
    avg_ear = (left_ear + right_ear) / 2.0

    mouth_width = euclidean_distance(landmarks[MOUTH_CORNERS[0]], landmarks[MOUTH_CORNERS[1]])
    mouth_height = euclidean_distance(landmarks[MOUTH_INNER_UP_DOWN[0]], landmarks[MOUTH_INNER_UP_DOWN[1]])

    # Simple heuristic logic
    # Higher mouth_width generally correlates to smiling
    # High mouth_height = surprise/talking
    # Low EAR = squinting / tired / sad
    
    score = 0.0
    
    # EAR contribution: continuous mapping
    # Normal EAR ~0.25-0.35, low → negative, high → slightly positive
    ear_score = (avg_ear - 0.25) * 2.0  # maps 0.15→-0.2, 0.25→0, 0.35→0.2
    score += max(-0.4, min(0.3, ear_score))
    
    # Mouth contribution: continuous mapping based on width and ratio
    mouth_ratio = mouth_width / (mouth_height + 1e-6)
    
    # Wider mouth → more positive (smile), narrow → negative (frown/pursed)
    mouth_score = (mouth_width - 0.06) * 6.0  # maps 0.04→-0.12, 0.06→0, 0.10→0.24
    score += max(-0.3, min(0.5, mouth_score))
    
    # Open mouth (surprise/talking) adds slight positive
    if mouth_height > 0.02:
        score += min(0.15, mouth_height * 3.0)

    # Clamp between -1.0 and 1.0
    return max(-1.0, min(1.0, score))
