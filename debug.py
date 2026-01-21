import mediapipe
import sys

print("Python Version:", sys.version)
print("MediaPipe Location:", mediapipe.__file__)
print("Dir(mediapipe):", dir(mediapipe))

try:
    import mediapipe.solutions
    print("SUCCESS: mediapipe.solutions found!")
except ImportError as e:
    print("FAILURE: Could not import solutions.", e)