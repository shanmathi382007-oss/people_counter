import cv2
import numpy as np
import math
import csv
import datetime
import os

class PeopleCounter:
    def __init__(self, source="video.mp4", line_file="lines.txt"):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print("ERROR: Video file cannot be opened")
            return

        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # Default entry and exit lines
        self.entry_line = [(100, 400), (540, 400)]
        self.exit_line = [(100, 300), (540, 300)]

        self.line_file = line_file
        self.load_lines()

        self.drag_idx = None
        self.radius = 10

        self.tracks = []
        self.next_id = 0
        self.max_age = 10
        self.dist_thresh = 50

        self.entry = 0
        self.exit = 0

        self.log_file = "entry_exit_log.csv"
        self.init_log()

    # Logging
    def init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                csv.writer(f).writerow(["Time", "Event", "Entry", "Exit"])

    def log(self, event):
        with open(self.log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                event,
                self.entry,
                self.exit
            ])

    # Line save/load
    def save_lines(self):
        with open(self.line_file, "w") as f:
            f.write(f"{self.entry_line[0][0]},{self.entry_line[0][1]},{self.entry_line[1][0]},{self.entry_line[1][1]}\n")
            f.write(f"{self.exit_line[0][0]},{self.exit_line[0][1]},{self.exit_line[1][0]},{self.exit_line[1][1]}\n")

    def load_lines(self):
        if os.path.exists(self.line_file):
            with open(self.line_file, "r") as f:
                lines = f.readlines()
            if len(lines) >= 2:
                x1, y1, x2, y2 = map(int, lines[0].strip().split(","))
                x3, y3, x4, y4 = map(int, lines[1].strip().split(","))
                self.entry_line = [(x1, y1), (x2, y2)]
                self.exit_line = [(x3, y3), (x4, y4)]

    # Mouse callback
    def mouse(self, event, x, y, flags, param):
        points = self.entry_line + self.exit_line
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(points):
                if math.dist((x, y), p) < self.radius:
                    self.drag_idx = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx is not None:
            if self.drag_idx == 0: self.entry_line[0] = (x, y)
            elif self.drag_idx == 1: self.entry_line[1] = (x, y)
            elif self.drag_idx == 2: self.exit_line[0] = (x, y)
            elif self.drag_idx == 3: self.exit_line[1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx = None
            self.save_lines()

    # Line cross check
    def crossed(self, p1, p2, line_y):
        return (p1[1] < line_y and p2[1] >= line_y) or (p1[1] > line_y and p2[1] <= line_y)

    # Main loop
    def run(self):
        if not self.cap.isOpened():
            return

        cv2.namedWindow("People Counter")
        cv2.setMouseCallback("People Counter", self.mouse)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            fg = self.bg.apply(frame)
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []

            for c in contours:
                if cv2.contourArea(c) > 700:
                    x, y, w, h = cv2.boundingRect(c)
                    cx, cy = x + w//2, y + h//2
                    detections.append((cx, cy))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            matched = set()
            new_tracks = []

            for t in self.tracks:
                t["age"] += 1
                best = None
                best_d = self.dist_thresh
                for i, d in enumerate(detections):
                    if i in matched: continue
                    dist = math.dist(t["centroid"], d)
                    if dist < best_d:
                        best_d = dist
                        best = i
                if best is not None:
                    matched.add(best)
                    prev = t["centroid"]
                    curr = detections[best]
                    t["centroid"] = curr
                    t["age"] = 0
                    if not t["entry"] and self.crossed(prev, curr, self.entry_line[0][1]):
                        self.entry += 1
                        self.log("Entry")
                        t["entry"] = True
                    if not t["exit"] and self.crossed(prev, curr, self.exit_line[0][1]):
                        self.exit += 1
                        self.log("Exit")
                        t["exit"] = True
                    new_tracks.append(t)
                elif t["age"] < self.max_age:
                    new_tracks.append(t)

            for i, d in enumerate(detections):
                if i not in matched:
                    new_tracks.append({
                        "id": self.next_id,
                        "centroid": d,
                        "age": 0,
                        "entry": False,
                        "exit": False
                    })
                    self.next_id += 1

            self.tracks = new_tracks

            # Draw lines
            cv2.line(frame, *self.entry_line, (0, 255, 0), 2)
            cv2.line(frame, *self.exit_line, (0, 0, 255), 2)
            for p in self.entry_line + self.exit_line:
                cv2.circle(frame, p, 6, (255, 255, 0), -1)

            # Display counts
            cv2.putText(frame, f"Entry: {self.entry}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
            cv2.putText(frame, f"Exit: {self.exit}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

            cv2.imshow("People Counter", frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        print("Final Entry:", self.entry)
        print("Final Exit:", self.exit)
        self.cap.release()
        cv2.destroyAllWindows()

# Run
if __name__ == "__main__":
    PeopleCounter(source=r"C:\Users\shanm\OneDrive\Desktop\peoplecounter\video.mp4").run()