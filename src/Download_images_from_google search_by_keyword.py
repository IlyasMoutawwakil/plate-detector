from google_images_download import google_images_download

response = google_images_download.googleimagesdownload() # Instantiation

# On doit avoir le fichier chromedriver.exe dans le même dossier que ce programme
arguments = {"keywords":"MOROCCAN PLATE, Plaque marocaine, Matricule maroc",
             "limit":1000, # On limite le nombre d'images puisqu'elles ne sont pas assez nombreuses
             "chromedriver":"E:\\ALPR\\TIPE\\data\\chromedriver.exe",
             "print_urls":True}

paths = response.download(arguments) # On commence le téléchargement et on filtre les images après
