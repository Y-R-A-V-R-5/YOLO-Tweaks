from roboflow import Roboflow

# 1. Replace with your actual API key
rf = Roboflow(api_key="ZpL1sRtawnJnuup8XTql")

# 2. Set workspace and project from the URL
workspace = rf.workspace("joseph-nelson")
project = workspace.project("plantdoc")

# 3. Use version 4 (as seen in the URL)
dataset = project.version(4).download("yolov8")

print("Dataset downloaded to:", dataset.location)