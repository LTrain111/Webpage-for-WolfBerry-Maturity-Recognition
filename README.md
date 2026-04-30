# YOLO Web Detection

This is a simple Flask web app for wolfberry maturity detection.

## Features

- Select a YOLO `.pt` model
- Upload an image
- Start detection from the main page
- Draw bounding boxes and confidence scores on the image
- Count mature fruits and total fruits
- Compute mature fruit ratio
- Output a harvest recommendation

## Decision Rules

- If mature fruit count is below `5`: do not harvest
- If mature fruit ratio is above `0.65`: harvest recommended
- If mature fruit ratio is between `0.50` and `0.65`: do not harvest
- If mature fruit ratio is below `0.50`: do not harvest

## Files

- `app.py`: Flask backend and YOLO inference
- `templates/index.html`: page structure
- `static/style.css`: page styling
- `requirements.txt`: dependencies

## Run

```bash
cd D:\BIYESHEJI\web_app
pip install -r requirements.txt
python app.py
```

Open this in the browser:

```text
http://127.0.0.1:5000
```

## Class Names

The app treats these labels as mature:

- `mature`
- `ripe`

The app treats these labels as immature:

- `immature`

If your model uses different class names, update the `DetectionItem.is_mature` rule in `app.py`.
