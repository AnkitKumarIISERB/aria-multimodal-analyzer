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
    
    # Very crude baseline assumptions for normalization
    # Normal EAR is ~0.25 to 0.35
    if avg_ear < 0.20:
        score -= 0.3  # Squinting or closed eyes
    
    # Normal mouth width vs height
    # If mouth is wide and slightly open -> Smile
    mouth_ratio = mouth_width / (mouth_height + 1e-6)
    
    if mouth_width > 0.08: # Arbitrary threshold, depends on face distance
        score += 0.4 # Smiling
    elif mouth_width < 0.05:
        score -= 0.2 # Pursed lips

    # Clamp between -1.0 and 1.0
    return max(-1.0, min(1.0, score))
