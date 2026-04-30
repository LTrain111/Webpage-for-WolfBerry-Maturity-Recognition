from __future__ import annotations

import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from flask import Flask, redirect, render_template, request, send_from_directory, url_for
from PIL import Image

# 直接使用pip安装的ultralytics版本
from ultralytics import YOLO  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MODEL_SUFFIXES = {".pt"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_MATURE_COUNT = 5


@dataclass
class DetectionItem:
    cls_id: int
    class_name: str
    confidence: float
    xyxy: tuple[int, int, int, int]

    @property
    def is_mature(self) -> bool:
        normalized = self.class_name.strip().lower()
        if "immature" in normalized or "unrip" in normalized:
            return False
        return normalized in {"mature", "ripe", "rip_berries"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

loaded_model: YOLO | None = None
loaded_model_name: str | None = None
last_image_name: str | None = None


def allowed_file(filename: str, suffixes: Iterable[str]) -> bool:
    return Path(filename).suffix.lower() in set(suffixes)


def save_upload(file_storage, target_dir: Path) -> Path:
    suffix = Path(file_storage.filename).suffix.lower()
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    target = target_dir / saved_name
    file_storage.save(target)
    return target


def load_selected_model(model_path: Path, original_name: str) -> None:
    global loaded_model, loaded_model_name
    loaded_model = YOLO(str(model_path))
    loaded_model_name = original_name


def parse_detections(result) -> list[DetectionItem]:
    boxes = result.boxes
    if boxes is None:
        return []

    names = result.names or {}
    items: list[DetectionItem] = []
    xyxy_array = boxes.xyxy.cpu().numpy().astype(int)
    cls_array = boxes.cls.cpu().numpy().astype(int)
    conf_array = boxes.conf.cpu().numpy()

    for xyxy, cls_id, conf in zip(xyxy_array, cls_array, conf_array):
        x1, y1, x2, y2 = xyxy.tolist()
        items.append(
            DetectionItem(
                cls_id=int(cls_id),
                class_name=str(names.get(int(cls_id), f"class_{cls_id}")),
                confidence=float(conf),
                xyxy=(x1, y1, x2, y2),
            )
        )
    return items


def render_visualization(image_path: Path, detections: list[DetectionItem]) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Failed to read the uploaded image.")

    for item in detections:
        color = (36, 68, 219) if item.is_mature else (67, 143, 91)
        x1, y1, x2, y2 = item.xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{item.class_name} {item.confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_top = max(0, y1 - text_height - 10)
        cv2.rectangle(image, (x1, text_top), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output_name = f"result_{uuid.uuid4().hex}.jpg"
    output_path = RESULT_DIR / output_name
    cv2.imwrite(str(output_path), image)
    return output_name


def compute_decision(mature_count: int, total_count: int) -> tuple[float, str]:
    if total_count == 0:
        return 0.0, "不采摘：未检测到果实。"
    ratio = mature_count / total_count
    if mature_count < MIN_MATURE_COUNT:
        return ratio, "不采摘：成熟果实数量低于5个。"
    if ratio > 0.65:
        return ratio, "建议采摘：成熟果实比例高于0.65。"
    if 0.5 <= ratio <= 0.65:
        return ratio, "不采摘：成熟果实比例在0.50到0.65之间。"
    return ratio, "不采摘：成熟果实比例低于0.50。"


def average_confidence(items: list[DetectionItem]) -> float:
    if not items:
        return 0.0
    return sum(item.confidence for item in items) / len(items)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_name=loaded_model_name or "None selected",
        image_name=last_image_name or "None selected",
        result_image=None,
        summary=None,
        detections=[],
        error=None,
    )


@app.route("/detect", methods=["POST"])
def detect():
    global last_image_name

    error = None
    result_image = None
    summary = None
    detection_rows: list[DetectionItem] = []

    model_file = request.files.get("model_file")
    image_file = request.files.get("image_file")

    try:
        if model_file and model_file.filename:
            if not allowed_file(model_file.filename, ALLOWED_MODEL_SUFFIXES):
                raise ValueError("Model file must be a .pt file.")
            saved_model = save_upload(model_file, UPLOAD_DIR)
            load_selected_model(saved_model, model_file.filename)

        if loaded_model is None:
            raise ValueError("Please select a YOLO .pt model first.")

        if image_file is None or not image_file.filename:
            raise ValueError("Please upload an image before detection.")
        if not allowed_file(image_file.filename, ALLOWED_IMAGE_SUFFIXES):
            raise ValueError("Unsupported image format. Use jpg, png, bmp, or webp.")

        saved_image = save_upload(image_file, UPLOAD_DIR)
        last_image_name = image_file.filename

        with Image.open(saved_image) as img:
            img.verify()

        prediction = loaded_model.predict(source=str(saved_image), conf=0.25, verbose=False)[0]
        detection_rows = parse_detections(prediction)
        mature_items = [item for item in detection_rows if item.is_mature]
        ratio, decision = compute_decision(len(mature_items), len(detection_rows))
        mature_avg_conf = average_confidence(mature_items)
        result_image = render_visualization(saved_image, detection_rows)

        summary = {
            "mature_count": len(mature_items),
            "total_count": len(detection_rows),
            "mature_ratio": f"{ratio:.2f}",
            "mature_avg_conf": f"{mature_avg_conf:.2f}",
            "decision": decision,
        }
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return render_template(
        "index.html",
        model_name=loaded_model_name or "None selected",
        image_name=last_image_name or "None selected",
        result_image=result_image,
        summary=summary,
        detections=detection_rows,
        error=error,
    )


@app.route("/results/<path:filename>")
def result_file(filename: str):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/reset", methods=["POST"])
def reset():
    global loaded_model, loaded_model_name, last_image_name
    loaded_model = None
    loaded_model_name = None
    last_image_name = None
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(RESULT_DIR, ignore_errors=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
