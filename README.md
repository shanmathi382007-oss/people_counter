# Real-Time People Counter

People Counter detects and tracks pedestrians in a video, webcam, or network
stream, counts crossings of a configurable virtual line, and records totals
and crossing events. It includes a live OpenCV counter and a Streamlit
dashboard for reviewing recorded events.

The supported application lives in [`people_counter/`](./people_counter/).
The root-level [`people_counter.py`](./people_counter.py) is an older,
separate OpenCV background-subtraction prototype; use the supported
`people_counter/main.py` entry point below.

## Technologies

- Python 3.10 or newer
- OpenCV and NumPy for video processing and display
- Ultralytics YOLOv8 with the bundled ONNX model and ByteTrack tracking
- SQLite for persistent crossing events; Python's standard library provides
  SQLite and CSV support
- Streamlit for dashboard controls and event display, with SVG charts
- Pandas for the logger's optional tabular-data helper
- PyYAML for application configuration
- Pytest for the existing unit tests

No external database server, API key, or environment variable is required.
The included video and ONNX model are used by the default configuration.

## Project contents

- `people_counter/main.py` — CLI/video counter entry point
- `people_counter/dashboard.py` — Streamlit analytics dashboard
- `people_counter/config.yaml` — video, model, counting-line, database, and
  output settings
- `people_counter/requirements.txt` — Python dependencies
- `people_counter/counter/` — detection, tracking, line counting, logging, and
  video/configuration utilities (`detector.py`, `line_counter.py`, `logger.py`,
  `utils.py`, and `__init__.py`)
- `people_counter/tests/` — line-crossing unit tests
  (`test_line_counter.py`)
- `people_counter/data/sample.mp4` — included sample video
- `people_counter/yolov8n.onnx` — included YOLOv8 model
- `video.mp4/` — default-config video `video.mp4.mp4` and an empty legacy
  notes file
- `models/` — legacy MobileNetSSD Caffe model and prototxt
- `SVM/` — 35 archived data files
- root `yolov8n.onnx` — additional copy of the YOLOv8 model
- `.venv/` and `venv/` — existing local Python environments, not source files

## Installation (Windows / VS Code)

Run these commands in the VS Code PowerShell terminal from the project root
(the folder containing this README):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\people_counter\requirements.txt
```

If PowerShell does not allow environment activation, use the environment's
Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\people_counter\requirements.txt
```

The existing workspace contains Python 3.11 virtual environments. The
commands above create/use `.venv` at the project root. The first installation
may take a while because computer-vision dependencies include compiled
packages.

## Run the people counter

From the project root, run:

```powershell
.\.venv\Scripts\python.exe .\people_counter\main.py --config .\people_counter\config.yaml
```

The counter opens an OpenCV video window and processes the video configured
in `people_counter/config.yaml`. Press **Q** or **Esc** to stop; press **L** to
redefine the counting line by clicking its two endpoints; press **R** to reset
the session's in-memory counts. To use a webcam or another video, edit the
`source` value in the config or pass `--source 0` (webcam) or
`--source .\people_counter\data\sample.mp4` (sample video).

To run without a video window, for example in a non-desktop environment:

```powershell
.\.venv\Scripts\python.exe .\people_counter\main.py --config .\people_counter\config.yaml --headless
```

For a quick bounded smoke run:

```powershell
.\.venv\Scripts\python.exe .\people_counter\main.py --config .\people_counter\config.yaml --headless --max-frames 2
```

### Expected output

The terminal reports the video source, model, line, and database settings,
then prints crossing events as they are detected. At the end it prints a
session summary with `Total IN`, `Total OUT`, and `Final Occupancy`. Counts
depend on the selected video and line placement. The application creates a
SQLite database (`people_counter.db`) and exports crossing events to
`outputs/crossings_log.csv` when it exits. Their locations are controlled by
the `logging` section in `config.yaml`.

## Run the dashboard

In a second VS Code terminal from the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\people_counter\dashboard.py
```

Streamlit prints a local URL (usually `http://localhost:8501`); open it in a
browser. The dashboard reads the SQLite database configured in
`people_counter/config.yaml`. Run the counter first to populate it with
crossing events; before any events are recorded, the dashboard shows zero
counts and an empty event table.

## Tests

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest .\people_counter\tests -q
```

## Configuration

Edit `people_counter/config.yaml` to change the input source, bundled model,
confidence/IoU thresholds, counting-line endpoints and direction, video
processing options, or SQLite/CSV output paths. For a local file, `source`
and relative output paths are interpreted from the project root when running
the CLI command above. The dashboard uses the same root-relative database and
CSV paths, so it displays events written by the CLI. No `.env` file or
external database setup is needed.

## GitHub usage

This project is hosted at
[github.com/shanmathi382007-oss/people_counter](https://github.com/shanmathi382007-oss/people_counter).
Clone it with:

```powershell
git clone https://github.com/shanmathi382007-oss/people_counter.git
cd people_counter
```

After making changes in the clone, commit and push them with `git add .`,
`git commit -m "Describe your changes"`, and `git push`.

The root `.gitignore` excludes virtual environments, Python/test caches, and
generated database/output files. The included model and video assets are
needed for the out-of-the-box demo and are not ignored.
