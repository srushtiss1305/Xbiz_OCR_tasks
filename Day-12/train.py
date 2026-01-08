# Imports
from inference_sdk import InferenceHTTPClient
import cv2
import os
import uuid

# CONFIG (CHANGE ONLY THIS)
IMAGE_PATH = r"C:\Users\Sanjay\Xbiz_OCR_tasks\Day-12\content\image.png"

API_KEY = "lZu8bze58evEgzCZQOTi"
WORKSPACE_NAME = "srushti-8vuus"
WORKFLOW_ID = "custom-workflow-2"

# OUTPUT FOLDER
ROOT_DIR = "output_images"
RUN_ID = uuid.uuid4().hex[:8]
BOXED_OUTPUT_DIR = os.path.join(ROOT_DIR, RUN_ID)
os.makedirs(BOXED_OUTPUT_DIR, exist_ok=True)

# LOAD IMAGE (ONCE)
img = cv2.imread(IMAGE_PATH)
assert img is not None, "❌ Image not found"
h, w = img.shape[:2]

# ROBOTFLOW WORKFLOW CALL
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

result = client.run_workflow(
    workspace_name=WORKSPACE_NAME,
    workflow_id=WORKFLOW_ID,
    images={"image": IMAGE_PATH},
    use_cache=True
)

# EXTRACT TABLE BOUNDING BOXES
tables = []

preds = result[0]["predictions"]["predictions"]

for pred in preds:
    if pred["class"] == "table":
        x, y, bw, bh = pred["x"], pred["y"], pred["width"], pred["height"]

        x1 = max(0, int(x - bw / 2))
        y1 = max(0, int(y - bh / 2))
        x2 = min(w, int(x + bw / 2))
        y2 = min(h, int(y + bh / 2))

        tables.append((x1, y1, x2, y2))

print(f"Tables found: {len(tables)}")

# DRAW BOXES
boxed_img = img.copy()

for idx, (x1, y1, x2, y2) in enumerate(tables, start=1):
    cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        boxed_img,
        f"TABLE {idx}",
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# SAVE RESULT
output_path = os.path.join(
    BOXED_OUTPUT_DIR,
    os.path.basename(IMAGE_PATH)
)

cv2.imwrite(output_path, boxed_img)
print(f"✅ Boxed image saved at: {output_path}")