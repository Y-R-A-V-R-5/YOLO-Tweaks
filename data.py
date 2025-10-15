from roboflow import Roboflow

# 1. Add your Roboflow API key here
rf = Roboflow(api_key="ZpL1sRtawnJnuup8XTql")  # Get your key from Roboflow account

# 2. Correct workspace and project slugs from your link
workspace = rf.workspace("plant-disease-detection-89qrx")
project = workspace.project("plant-disease-detection-znzrh")

# 3. Use version 5 as per your link
dataset = project.version(5).download("yolov8")

print("Available versions:", project.versions())