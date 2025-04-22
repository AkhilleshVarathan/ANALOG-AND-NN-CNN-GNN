import torch
import os
import cv2
import numpy as np
import heapq
import time

# --- A* Pathfinding ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = [(0 + heuristic(start, goal), 0, start, [])]
    visited = set()

    while open_set:
        f, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        path = path + [current]
        if current == goal:
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0 and (nx, ny) not in visited:
                heapq.heappush(open_set, (cost + 1 + heuristic((nx, ny), goal), cost + 1, (nx, ny), path))
    return []

# --- Convert image to binary grid ---
def image_to_grid(img, grid_size=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    resized = cv2.resize(binary, (binary.shape[1] // grid_size, binary.shape[0] // grid_size))
    grid = (resized > 128).astype(int)  # 1 = obstacle, 0 = free
    return grid

# --- Find free point ---
def find_free_spot(grid, from_corner='top-left'):
    rows, cols = grid.shape
    scan = ((r, c) for r in range(rows) for c in range(cols)) if from_corner == 'top-left' else (
           ((r, c) for r in reversed(range(rows)) for c in reversed(range(cols))))
    for r, c in scan:
        if grid[r, c] == 0:
            return (r, c)
    return None

# --- Visualize grid + path ---
def visualize_grid(grid, path=None, start=None, goal=None):
    color_grid = np.stack([grid * 255] * 3, axis=-1).astype(np.uint8)
    color_grid[grid == 1] = [0, 0, 0]
    color_grid[grid == 0] = [255, 255, 255]

    if path:
        for r, c in path:
            color_grid[r, c] = [0, 0, 255]
    if start:
        color_grid[start[0], start[1]] = [0, 255, 0]
    if goal:
        color_grid[goal[0], goal[1]] = [255, 0, 0]

    zoom = 10
    display_img = cv2.resize(color_grid, (color_grid.shape[1]*zoom, color_grid.shape[0]*zoom), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Grid View", display_img)

# --- Mouse callback function for selecting start/goal ---
start_point = None
goal_point = None

def click_event(event, x, y, flags, param):
    global start_point, goal_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if start_point is None:
            start_point = (y, x)  # (row, column) as (y, x)
            print(f"[INFO] Start selected: {start_point}")
        elif goal_point is None:
            goal_point = (y, x)
            print(f"[INFO] Goal selected: {goal_point}")

# --- Main function ---
def main():
    # --- Load YOLOv5 ---
    local_yolov5_path = r"C:\Users\Deepak Skandh\yolov5"
    model_path = os.path.join(local_yolov5_path, 'runs', 'train', 'obstacle_detector_v1', 'weights', 'best.pt')
    model = torch.hub.load(local_yolov5_path, 'custom', path=model_path, source='local')
    model.eval()

    # --- Connect to ESP32-CAM stream ---
    stream_url = "http://192.168.99.204:81/stream"
    cap = None
    for attempt in range(10):
        cap = cv2.VideoCapture(stream_url)
        time.sleep(3)  # Increased sleep time for connection retries
        if cap.isOpened():
            print("[INFO] Connected to ESP32-CAM stream.")
            break
        else:
            print(f"[WARN] Attempt {attempt + 1}/10: Retrying connection...")

    if not cap or not cap.isOpened():
        print("[ERROR] Could not open ESP32-CAM stream.")
        return

    # --- Set mouse callback to select start/goal ---
    cv2.namedWindow("YOLOv5 + A* Path")
    cv2.setMouseCallback("YOLOv5 + A* Path", click_event)

    print("[INFO] Detection + Pathfinding started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        img = frame.copy()

        # --- YOLOv5 Detection ---
        results = model(img)
        preds = results.xyxy[0]
        for *box, conf, cls in preds:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --- Grid + A* Path ---
        grid = image_to_grid(img, grid_size=10)

        if start_point and goal_point:
            path = astar(grid, start_point, goal_point)
            print("[INFO] Path length:", len(path) if path else "No path found")
        else:
            path = []

        # --- Visual Path Overlay ---
        scale_x = img.shape[1] // grid.shape[1]
        scale_y = img.shape[0] // grid.shape[0]
        if path:
            for (x, y) in path:
                cx, cy = y * scale_x + scale_x // 2, x * scale_y + scale_y // 2
                cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)

        cv2.imshow("YOLOv5 + A* Path", img)
        visualize_grid(grid, path=path, start=start_point, goal=goal_point)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
