import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os

root_dir = "G:/DATASET_Infosearch/FINAL/SET_1_2/"
filename = "20250514135847_20250514142404_ca8c3210-8340-407c-a882-2435098118af_Trailer_In_1_D_1_1071"

# Load image (replace with your actual image path)
image_path = os.path.join(root_dir, "images", filename+".jpg")
image = cv2.imread(image_path)

# Load JSON
with open(os.path.join(root_dir, "Labels_filtered", filename+".json"), "r") as f:
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
                cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=4)
                # Optional: add label
                label = subpart.get("value", "")
                x, y = pts[0][0]
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Show image using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis("off")
plt.savefig("annotated_image.png", bbox_inches='tight', pad_inches=0.1)
plt.show()
