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

# --- Visualize grid + path ---
def visualize_grid(grid, path=None, start=None, goal=None):
    color_grid = np.stack([grid * 255] * 3, axis=-1).astype(np.uint8)
    color_grid[grid == 1] = [0, 0, 0]  # Obstacles
    color_grid[grid == 0] = [255, 255, 255]  # Free space

    if path:
        for r, c in path:
            color_grid[r, c] = [0, 0, 255]  # Red path
    if start:
        color_grid[start[0], start[1]] = [0, 255, 0]  # Green start
    if goal:
 color_grid[goal[0], goal[1]] = [255, 0, 0]  # Blue goal

    zoom = 10
    display_img = cv2.resize(color_grid, (color_grid.shape[1] * zoom, color_grid.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Grid View", display_img)

# --- Main Function ---
def main():
    # Load YOLOv5 model locally (fine-tuned weights)
    yolov5_path = '/home/pi/yolov5'  # Cloned yolov5 repo
    weights_path = '/home/pi/yolov5_updated/best.pt'  # Fine-tuned model
    model = torch.hub.load(yolov5_path, 'custom', path=weights_path, source='local')
    model.eval()

    # ESP32-CAM Stream URL
    esp32_url = 'http://192.168.251.142:81/stream'  # <-- Updated your ESP32-CAM IP

    # Connect to ESP32-CAM stream
    cap = cv2.VideoCapture(esp32_url)
    time.sleep(2)  # Allow stream to stabilize

    if not cap.isOpened():
        print("[ERROR] Could not open ESP32-CAM stream.")
        return
    print("[INFO] ESP32-CAM stream connected successfully.")

    cv2.namedWindow("YOLOv5 + A* Path")
print("[INFO] Detection + Pathfinding started.")

    # Predefine start and goal points (example)
    start_point = (5, 5)  # Example start point (row, col)
    goal_point = (15, 15)  # Example goal point (row, col)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from ESP32-CAM.")
            break

        img = frame.copy()

        # Run YOLO detection
        results = model(img)
        preds = results.xyxy[0]
        for *box, conf, cls in preds:
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # A* Pathfinding
        grid = image_to_grid(img, grid_size=10)
        path = astar(grid, start_point, goal_point)
        print("[INFO] Path length:", len(path) if path else "No path found")

# Draw Path + Points on frame
        scale_x = img.shape[1] // grid.shape[1]
        scale_y = img.shape[0] // grid.shape[0]

        if path:
            for (r, c) in path:
                cx = c * scale_x + scale_x // 2
                cy = r * scale_y + scale_y // 2
                cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)  # Red

        if start_point:
            sx = start_point[1] * scale_x + scale_x // 2
            sy = start_point[0] * scale_y + scale_y // 2
            cv2.circle(img, (sx, sy), 6, (0, 255, 0), -1)  # Green

        if goal_point:
            gx = goal_point[1] * scale_x + scale_x // 2
            gy = goal_point[0] * scale_y + scale_y // 2
            cv2.circle(img, (gx, gy), 6, (255, 0, 0), -1)  # Blue

        # Display results
        cv2.imshow("YOLOv5 + A* Path", img)
        visualize_grid(grid, path=path, start=start_point, goal=goal_point)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
cap.release()
    cv2.destroyAllWindows()

# --- Entry Point ---
if _name_ == "_main_":
    main()
