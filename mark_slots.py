import cv2
import numpy as np
import os

# Dictionary of videos to choose from
video_files = {
    "1": r"G:\Car-Parking-Detection\video.mp4\easy1.mp4",
    "2": r"G:\Car-Parking-Detection\video.mp4\easy2.mp4",
    "3": r"G:\Car-Parking-Detection\video.mp4\easy3.mp4"
}

print("🎥 Select a video:")
for key, path in video_files.items():
    print(f"{key}. {os.path.basename(path)}")

choice = input("Enter choice (1/2/3): ").strip()

if choice not in video_files:
    print("❌ Invalid selection.")
    exit()

video_path = video_files[choice]
video_name = os.path.splitext(os.path.basename(video_path))[0]  # e.g. 'easy1'

print(f"📂 Loading video: {video_path}")

# Create parking_slots directory if it doesn't exist
slot_dir = "parking_slots"
os.makedirs(slot_dir, exist_ok=True)

# Slot file path based on video name
slot_file = os.path.join(slot_dir, f"parking_slots_{video_name}.npy")

# Open video capture
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("❌ Failed to read video")
    cap.release()
    exit()

clone = frame.copy()

# Globals used in mouse callback
drawing = False
current_box = []
boxes = []
dragging_point = None
selected_box_idx = -1

# Load saved slots if available
if os.path.exists(slot_file):
    boxes = list(np.load(slot_file, allow_pickle=True))
    print(f"📂 Loaded existing slots from {slot_file}")
else:
    print("🆕 No existing slots found. Drawing new.")

def distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def mouse_callback(event, x, y, flags, param):
    global drawing, current_box, boxes, dragging_point, selected_box_idx

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if user clicked near any existing point to drag
        for b_idx, box in enumerate(boxes):
            for p_idx, point in enumerate(box):
                if distance(point, (x, y)) < 10:
                    dragging_point = p_idx
                    selected_box_idx = b_idx
                    return
        # Start drawing new box
        current_box = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if len(current_box) == 1:
                # Create rectangle corners based on current mouse pos
                current_box.append((x, current_box[0][1]))
                current_box.append((x, y))
                current_box.append((current_box[0][0], y))
            else:
                # Update rectangle corners dynamically while drawing
                current_box[1] = (x, current_box[0][1])
                current_box[2] = (x, y)
                current_box[3] = (current_box[0][0], y)
        elif dragging_point is not None and selected_box_idx != -1:
            # Drag selected point of existing box
            boxes[selected_box_idx][dragging_point] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing and len(current_box) == 4:
            boxes.append(current_box)
        drawing = False
        current_box = []
        dragging_point = None
        selected_box_idx = -1

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click deletes last box
        if boxes:
            boxes.pop()
            print("🗑️ Deleted last slot")

# Setup OpenCV window and set mouse callback
window_name = "Flexible Slot Drawer - Keys: [s] Save, [q] Quit, [c] Copy last"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    temp = clone.copy()

    # Draw the box being currently drawn
    if current_box:
        cv2.polylines(temp, [np.array(current_box)], isClosed=True, color=(0, 255, 255), thickness=2)
        for pt in current_box:
            cv2.circle(temp, pt, 5, (255, 255, 255), -1)

    # Draw all existing boxes and their points
    for i, box in enumerate(boxes):
        pts = np.array(box)
        cv2.polylines(temp, [pts], isClosed=True, color=(255, 0, 255), thickness=2)
        for pt in box:
            cv2.circle(temp, pt, 5, (0, 255, 0), -1)
        cv2.putText(temp, f"Slot {i+1}", box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow(window_name, temp)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save slots
        np.save(slot_file, boxes)
        print(f"✅ Saved to {slot_file}")
        break
    elif key == ord('q'):
        print("👋 Exiting without saving changes.")
        break
    elif key == ord('c'):
        if boxes:
            # Copy last slot with offset
            new_box = [(x + 40, y + 20) for (x, y) in boxes[-1]]
            boxes.append(new_box)
            print("📋 Copied last slot and added as new slot.")

cap.release()
cv2.destroyAllWindows()
