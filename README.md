# LUNG-TUMOR-SEGMENTATION-USING-U-NET
🧠 Lung Cancer Region Segmentation with U-Net
This project implements a lightweight U-Net architecture for binary segmentation of lung cancer regions using the IQOTHNCCD Lung Cancer Dataset, and visualizes detected regions using overlaid red dots.

🚀 Features
Simple U-Net-based architecture for segmentation.

Mixed precision training with PyTorch AMP for faster performance.

Red-dot visualization for detected cancerous regions.

Lightweight for quick experimentation in notebooks or on Google Colab.

🗃️ Dataset
Downloaded from Kaggle via kagglehub:

makefile
Copy
Edit
Dataset: IQOTHNCCD Lung Cancer Dataset
Link: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset
Download Using KaggleHub
python
Copy
Edit
import kagglehub
path = kagglehub.dataset_download("adityamahimkar/iqothnccd-lung-cancer-dataset")
🏗️ Model Architecture
text
Copy
Edit
Input (3x128x128)
│
├── Conv2D + ReLU
├── MaxPool
│
├── Conv2D + ReLU
├── MaxPool
│
├── ConvTranspose2D + ReLU
├── ConvTranspose2D + Sigmoid
│
Output (1x128x128)
Loss Function
Binary Cross-Entropy (BCE) Loss

Optimizer
Adam with lr=1e-4

Mixed Precision
Uses torch.cuda.amp for efficient training.

🧪 Training
python
Copy
Edit
for epoch in range(epochs):
    for images, _ in train_loader:
        masks = torch.rand_like(images[:, :1, :, :])  # Random masks (placeholder)
        ...
Note: Replace placeholder masks with actual ground truth masks for real applications.

📊 Visualization
Overlays predicted mask on input image.

Highlights predicted cancerous regions with red dots.

Displays original and predicted views side-by-side.

🖼️ Example Output
Original Image

Binary Prediction

Red-Dot Overlay

(replace with actual image if hosted on GitHub)

🧰 Requirements
bash
Copy
Edit
pip install torch torchvision matplotlib kagglehub opencv-python
📌 Notes
The current example uses random masks for demonstration purposes.

Replace this with real segmentation labels for meaningful training and inference.
