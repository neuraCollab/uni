# %%
import zipfile
import os
# %%
if not os.path.exists("data"):
    with zipfile.ZipFile("./russian-car-plates-prices-prediction.zip", 'r') as zip_ref:
        zip_ref.extractall("data")
# %%
