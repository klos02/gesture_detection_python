import cv2
import mediapipe as mp
import math
import numpy as np
 
# Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
 
# Wczytanie strumienia wideo z kamery
video_capture = cv2.VideoCapture(0)
 
def calculate_angle(a, b, c):
    a = np.array(a)  # Punkt A
    b = np.array(b)  # Punkt B (środek)
    c = np.array(c)  # Punkt C
 
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
 
    if angle > 180.0:
        angle = 360.0 - angle
 
    return angle
 
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
 
 
while True:
    # Pobranie klatki z wideo
    ret, frame = video_capture.read()
    if not ret:
        break
 
    # Konwersja koloru klatki z BGR (OpenCV) na RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Przetworzenie klatki w celu wykrycia dłoni
    result = hands.process(rgb_frame)
 
    # Jeśli wykryto dłonie
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Rysowanie linii i punktów dłoni na oryginalnej klatce wideo
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
            # Pobranie współrzędnych odpowiednich punktów
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            ring_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            pinky_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
 
            # Konwersja współrzędnych na piksele
            thumb_tip_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            thumb_mcp_coords = (int(thumb_mcp.x * frame.shape[1]), int(thumb_mcp.y * frame.shape[0]))
            index_finger_tip_coords = (int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0]))
            index_finger_mcp_coords = (int(index_finger_mcp.x * frame.shape[1]), int(index_finger_mcp.y * frame.shape[0]))
            middle_finger_tip_coords = (int(middle_finger_tip.x * frame.shape[1]), int(middle_finger_tip.y * frame.shape[0]))
            middle_finger_mcp_coords = (int(middle_finger_mcp.x * frame.shape[1]), int(middle_finger_mcp.y * frame.shape[0]))
            ring_finger_tip_coords = (int(ring_finger_tip.x * frame.shape[1]), int(ring_finger_tip.y * frame.shape[0]))
            ring_finger_mcp_coords = (int(ring_finger_mcp.x * frame.shape[1]), int(ring_finger_mcp.y * frame.shape[0]))
            pinky_finger_tip_coords = (int(pinky_finger_tip.x * frame.shape[1]), int(pinky_finger_tip.y * frame.shape[0]))
            pinky_finger_mcp_coords = (int(pinky_finger_mcp.x * frame.shape[1]), int(pinky_finger_mcp.y * frame.shape[0]))
 
            # Obliczenie kąta między kciukiem a palcem wskazującym
            angle = calculate_angle(thumb_tip_coords, thumb_mcp_coords, index_finger_tip_coords)
            isCloseToMcp = True
            distance_index = calculate_distance(index_finger_mcp_coords, index_finger_tip_coords)
            distance_middle = calculate_distance(middle_finger_mcp_coords, middle_finger_tip_coords)
            distance_ring = calculate_distance(ring_finger_mcp_coords, ring_finger_tip_coords)
            distance_pinky = calculate_distance(pinky_finger_mcp_coords, pinky_finger_tip_coords)
            
            distances = [distance_index, distance_middle, distance_ring, distance_pinky]
            
            for i in distances:
                if i > 70:
                    isCloseToMcp = False
                    break
            
            # Rozpoznanie gestu na podstawie kąta
            #print(distance_tip_mcp)
            if angle > 60 and thumb_tip.y < index_finger_tip.y and isCloseToMcp:
                cv2.putText(frame, 'Thumbs Up', (thumb_tip_coords[0], 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif thumb_tip.y > index_finger_tip.y and isCloseToMcp:
                cv2.putText(frame, 'Thumbs Down', (thumb_tip_coords[0], 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Wyświetlanie obrazu z wykrytymi dłońmi i gestami
    cv2.imshow('Video', frame)
 
    # Naciśnięcie 'q', aby wyjść z pętli
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Zwolnienie uchwytu kamery i zamknięcie okna
video_capture.release()
cv2.destroyAllWindows()