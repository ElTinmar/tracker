from urllib.request import urlretrieve
import zipfile
import os

FOLDER_URL = 'https://owncloud.gwdg.de/index.php/s/3I1wGOdmh8W2XaZ/download'
DESTINATION = 'toy_data.zip'

print('Downloading toy data...')
urlretrieve(FOLDER_URL, DESTINATION)
print(f'Extracting {DESTINATION}...')
with zipfile.ZipFile(DESTINATION, 'r') as zip_ref:
    zip_ref.extractall()
print("Extraction complete.")
print(f'Removing {DESTINATION}...')
os.remove(DESTINATION)
