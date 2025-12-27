import cv2
import numpy as np
from sort.sort import Sort
import time

tracker = Sort(max_age=70, min_hits=1, iou_threshold=0.1135)

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture("E:/pythontest/demca/new/1/thumucghihinh/lan29.mp4")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
counted_ids = set()
count = 0

# Tăng dist2Threshold lên một chút để giảm noise
object_detector = cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=700, detectShadows=False)

prev_time = 0
track_right_edge_history = {}

line_x = 250

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    # Blur nhẹ hơn chút để giữ chi tiết cạnh
    blurred_frame = cv2.GaussianBlur(frame, (7, 7), 0)
    mask = object_detector.apply(blurred_frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # --- XỬ LÝ TÁCH DÍNH ---
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 1. Xóa nhiễu
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)

    # 2. Cắt gọt (Erode) để tách 2 con cá ra
    # Dùng kernel nhỏ (3,3) và erode 1-2 lần
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_separated = cv2.erode(mask_clean, kernel_erode, iterations=1)

    # Hiển thị mask này để debug xem đã tách chưa
    cv2.imshow("mask_separated", mask_separated)
    # -----------------------

    coutours, _ = cv2.findContours(mask_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, coutours, -1, (0, 255, 255), 2)
    detections = []
    for cnt in coutours:
        area = cv2.contourArea(cnt)

        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)

            # 2. Tính toán Padding (Phóng to box)
            padding = 5
            x_new = max(0, x - padding)
            y_new = max(0, y - padding)
            w_new = w + padding * 2
            h_new = h + padding * 2
            #cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 255, 0), 2)

            detections.append([x_new, y_new, (x_new + w_new), (y_new + h_new), 1.0])
        detections_np = np.array(detections)
        if len(detections) == 0:
            detections_np = np.empty((0, 5))

    tracks = tracker.update(detections_np)

    # --- LOGIC ĐẾM (Giữ nguyên của bạn) ---
    active_ids = {int(t[4]) for t in tracks}
    ids_to_remove = []
    for track_id_in_history in track_right_edge_history:
        if track_id_in_history not in active_ids:
            ids_to_remove.append(track_id_in_history)

    for old_id in ids_to_remove:
        del track_right_edge_history[old_id]

    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if track_id in track_right_edge_history:
            prev_x2 = track_right_edge_history[track_id]
            if (prev_x2 < line_x) and (x2 >= line_x) and (track_id not in counted_ids):
                count += 1
                counted_ids.add(track_id)
                print(f"Phat hien ID: {track_id} , Da dem: {count}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        track_right_edge_history[track_id] = x2

    cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()