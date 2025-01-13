Real-Time Unauthorized Construction Detection and Area Assessment Using Drones
Overview
This project leverages cutting-edge Mask R-CNN for segmenting unauthorized construction areas from drone-captured images and calculating their areas using Geospatial Data Integration (GDI) techniques. Additionally, Time-of-Flight (ToF) sensors are utilized in a second phase to assess depth and validate volumetric details.

📹 Demo Video: Watch the Project in Action

How It Works
1. Detection and Segmentation Using Mask R-CNN
Mask R-CNN Architecture:
The Mask R-CNN model is trained on a custom dataset of drone images to accurately segment areas of unauthorized construction.

Input: Drone images captured in real-time.
Output: Binary masks highlighting the segmented construction areas.
Steps for Area Detection:

The segmented regions are extracted as binary masks.
These masks are overlaid on georeferenced maps to associate the segmented regions with spatial coordinates.
Using Geospatial Data Integration (GDI) tools, the pixel area of the mask is converted into real-world units (e.g., square meters).
<br>
2. Validation with Time-of-Flight (ToF) Sensors
ToF Sensor Integration:
ToF sensors are deployed to measure the depth of structures and estimate volumetric details.
Depth Calculation: Provides an additional dimension of analysis by measuring the height of unauthorized structures.
This validation step ensures accuracy in detecting discrepancies between approved and unauthorized construction volumes.
<br>
Features
Real-Time Detection: Utilizes drone feeds for instant segmentation.
High Accuracy: Employs the robust Mask R-CNN model for precise area segmentation.
Spatial Analysis: Converts segmented masks into real-world measurements with geospatial tools.
Volumetric Assessment: Enhances detection accuracy with ToF sensor-based depth analysis.
<br>
Requirements
Hardware:

Drone equipped with a high-resolution camera.
Time-of-Flight (ToF) sensors for depth validation.
Software:

Python
TensorFlow
OpenCV
Geospatial Data Integration Tools (e.g., QGIS, GDAL)
<br>
Installation
bash
Copy code
# Clone the repository  
git clone https://github.com/pranjalpandeyl221/Real-Time-Unauthorized-Construction-Detection-and-Area-Assessment-Using-Drones  

# Navigate to the directory  
cd Real-Time-Unauthorized-Construction-Detection-and-Area-Assessment-Using-Drones  

# Install dependencies  
pip install -r requirements.txt  
<br>
Usage
Prepare Drone Images:
Capture drone images of the area to be analyzed.

Run the Segmentation Model:

bash
Copy code
python segment_area.py --input drone_image.jpg --output mask_output.jpg  
Calculate Area:
Use the geospatial integration tools provided in the repository to convert the segmented masks into real-world units.

ToF Depth Analysis (Optional):
Deploy the ToF module to validate the depth and volumetric data.

<br>
Applications
Monitoring unauthorized construction in real estate or urban planning.
Ensuring compliance with building regulations.
Efficient land management and resource allocation.
<br>
Demo
📹 Watch the Project Demo

<br>
