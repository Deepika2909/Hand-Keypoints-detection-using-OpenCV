import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Read input image
image = cv2.imread("C:\\Users\\deepika\\Downloads\\OIP.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to find hand landmarks
results = hands.process(image_rgb)

# Draw hand landmarks on the image
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw landmarks and connections
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        )

# Display the annotated image
cv2.imshow('Hand Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
