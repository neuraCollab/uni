#  %%
import os
import zipfile

# extract data from zip
data_sync = "2011_09_26_drive_0001_sync.zip"
data_calib = "2011_09_26_calib.zip"
data_track = "2011_09_26_drive_0001_tracklets.zip"

if not os.listdir("data"):
    with zipfile.ZipFile("./2011_09_26_drive_0001_sync.zip", "r") as sync:
        sync.extractall("data")
        
# ---or---
# import zipfile
# import os

# zip_path_folder = './'

# for filename in os.listdir(zip_path_folder):
#     if filename.endswith(".zip"):
#         with zipfile.ZipFile(os.path.join(zip_path_folder, filename), 'r') as zip_ref:
#             zip_ref.extractall(zip_path_folder)
#         print(f"Extracted {filename} to {zip_path_folder}")