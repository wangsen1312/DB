import cv2
import numpy as np
import json
import matplotlib.pyplot as plt

# Load image (replace with your actual image path)
image_path = "your_image.jpg"
image = cv2.imread(image_path)

# Load JSON
with open("20250219004158_20250219004242_b6667981-3038-4ac1-b161-fdbee24b8b1d_Front-Left.mp4_63.json", "r") as f:
    data = json.load(f)

# Extract the annotation dict (only one image key in this case)
image_key = list(data.keys())[0]
annotations = data[image_key]["vehicles"]

# Draw text box polygons
for vehicle in annotations:
    for part in vehicle.get("parts", []):
        for subpart in part.get("subParts", []):
            if subpart["category"] == "text box":
                seg = subpart["segmentation"][0]  # one polygon
                pts = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                # Optional: add label
                label = subpart.get("value", "")
                x, y = pts[0][0]
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Show image using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
