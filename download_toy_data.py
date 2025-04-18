from urllib.request import urlretrieve

FOLDER_URL = 'https://owncloud.gwdg.de/index.php/s/3I1wGOdmh8W2XaZ/download'
DESTINATION = 'toy_data'

print('Downloading toy data...')
urlretrieve(FOLDER_URL, DESTINATION)