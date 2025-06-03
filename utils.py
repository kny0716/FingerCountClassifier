import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

def extract_features(landmarks):
    angles = []
    joint_sets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),    
        (0, 5, 6), (5, 6, 7), (6, 7, 8),    
        (0, 9, 10), (9, 10, 11), (10, 11, 12), 
        (0, 13, 14), (13, 14, 15), (14, 15, 16), 
        (0, 17, 18), (17, 18, 19), (18, 19, 20)  
    ]
    for j in joint_sets:
        a = [landmarks[j[0]].x, landmarks[j[0]].y]
        b = [landmarks[j[1]].x, landmarks[j[1]].y]
        c = [landmarks[j[2]].x, landmarks[j[2]].y]
        angles.append(calculate_angle(a, b, c))
    return angles

def extract_features_india(landmarks):
    angles = []
    joint_sets = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    for j in joint_sets:
        a = [landmarks[j[0]].x, landmarks[j[0]].y]
        b = [landmarks[j[1]].x, landmarks[j[1]].y]
        c = [landmarks[j[2]].x, landmarks[j[2]].y]
        angles.append(calculate_angle(a, b, c))

    base_x, base_y = landmarks[0].x, landmarks[0].y
    tip_indices = [4, 8, 12, 16, 20]
    relative_coords = []
    distances = []

    for i in tip_indices:
        dx = landmarks[i].x - base_x
        dy = landmarks[i].y - base_y
        relative_coords.extend([dx, dy])
        distances.append(np.sqrt(dx**2 + dy**2))

    return angles + relative_coords + distances


