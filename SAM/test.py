import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import SamModel, SamProcessor

# Step 1: Set device and load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Step 2: Load the image from a local path
img_path = r"C:\Users\Khushi.Jirge\sem-seg\Custom-dataset-trained\2024-09-20 15.24.png"
raw_image = Image.open(img_path).convert("RGB")  # Load image directly from the file path

# Step 3: Define input points for segmentation
input_points = [[[555, 1350], [1300, 2390], [2270, 2285], [3170, 2320]]]  # 2D location of a window in the image (you can modify this for your use case)

# Step 4: Preprocess inputs
inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)

# Step 5: Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Step 6: Post-process the mask
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

# Step 7: Extract the first mask and ensure it's a 2D array
mask = masks[0].squeeze()  # Remove any extra dimensions (this should give us a 2D mask)

# Check if the mask has 3 channels (we need to convert it to a single channel)
if mask.shape[0] == 3:
    mask = mask[0]  # Extract the first channel

# Step 8: Display the original image and mask side-by-side
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Display original image
ax[0].imshow(raw_image)
ax[0].set_title("Original Image")
ax[0].axis("off")

# Display the mask (converted to binary if needed)
ax[1].imshow(mask, cmap='Blues', alpha=0.5)  # Use `alpha` to blend the mask with the image
ax[1].set_title("Predicted Mask")
ax[1].axis("off")

plt.show()

# Step 9: Optionally, overlay the mask on the image
# Convert the raw image to a NumPy array for overlaying
raw_image_np = np.array(raw_image)

# Apply mask (you can choose the color you want for the overlay, here it's green)
masked_image = raw_image_np.copy()
masked_image[mask == 1] = [0, 255, 0]  # Apply green color where the mask is 1

# Display the result with the mask overlay
plt.figure(figsize=(8, 8))
plt.imshow(masked_image)
plt.title("Image with Mask Overlay")
plt.axis("off")
plt.show()
