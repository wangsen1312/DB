import os
import json

# ---- Config ----
json_path = "20250219004158_20250219004242_b6667981-3038-4ac1-b161-fdbee24b8b1d_Front-Left.mp4_63.json"
output_dir = "gt_labels/"
os.makedirs(output_dir, exist_ok=True)

# ---- Define class mapping ----
class_mapping = {
    "Other": 0,
    "Truck Number": 1,
    "USDOTNumber": 2,
    # Add more if needed
}

# ---- Load JSON and extract labels ----
with open(json_path, "r") as f:
    data = json.load(f)

for image_name, content in data.items():
    vehicles = content["vehicles"]
    gt_lines = []

    for vehicle in vehicles:
        for part in vehicle.get("parts", []):
            for subpart in part.get("subParts", []):
                if subpart["category"] == "text box":
                    seg = subpart["segmentation"][0]
                    if len(seg) != 8:
                        print(f"Skipping malformed polygon in {image_name}")
                        continue
                    coords = ",".join(str(int(x)) for x in seg)
                    text_class = subpart.get("type", "unknown")
                    class_id = class_mapping.get(text_class, -1)
                    if class_id == -1:
                        print(f"Warning: Unknown class '{text_class}' in {image_name}")
                    line = f"{coords},{class_id}"
                    gt_lines.append(line)

    # Save to file
    base = os.path.splitext(os.path.basename(image_name))[0]
    output_file = os.path.join(output_dir, f"gt_{base}.txt")
    with open(output_file, "w") as f:
        f.write("\n".join(gt_lines))

    print(f"Saved {len(gt_lines)} annotations to {output_file}")
