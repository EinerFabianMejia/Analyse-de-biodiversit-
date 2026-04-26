import ee
import datetime
import time
import os
import zipfile

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

ee.Initialize(project='project-id-ee-einerfabianmejia')


# -------------------------
# PARAMÈTRES
# -------------------------

ZONE = ee.FeatureCollection(
    'projects/project-id-ee-einerfabianmejia/assets/Limite_INRS_Laval').geometry()


TRAINING = ee.FeatureCollection(
    'projects/project-id-ee-einerfabianmejia/assets/endmember'
)

EXPORT_FOLDER = "INRS_exports"

LOCAL_FOLDER = "/home/projet-genie/Téléchargements/Projet-2-linus/gee_pipeline/ClassificationINRS"

# créer dossier local
os.makedirs(LOCAL_FOLDER, exist_ok=True)

# -------------------------
# DATES
# -------------------------

today = datetime.date.today()
year = today.year

dateDebut = ee.Date.fromYMD(year, 6, 1)
dateFin = ee.Date.fromYMD(year, 8, 31)

# -------------------------
# COLLECTION
# -------------------------

SENTINEL = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')

# -------------------------
# FILTRE NUAGES
# -------------------------

def filtreNuages(p):
    return SENTINEL \
        .filterDate(dateDebut, dateFin) \
        .filterBounds(ZONE) \
        .filter(ee.Filter.lt(
            'CLOUDY_PIXEL_PERCENTAGE', p
        ))

collection5 = filtreNuages(5)

nbImages5 = collection5.size()

collectionFinale = ee.ImageCollection(
    ee.Algorithms.If(
        nbImages5.gt(0),
        collection5,
        filtreNuages(10)
    )
)

image_filtre = collectionFinale.median()

# -------------------------
# BANDS
# -------------------------

bands10m = ['B2','B3','B4','B8']
bands20m = ['B5','B6','B11','B12']

imageRGBNI = image_filtre \
.select(bands10m) \
.addBands(
    image_filtre
    .select(bands20m)
    .reproject(
        crs=image_filtre
        .select(bands10m)
        .projection(),
        scale=10
    )
).divide(10000)

# -------------------------
# RANDOM FOREST
# -------------------------

rf = ee.Classifier.smileRandomForest(500).train(
    features=TRAINING,
    classProperty='class',
    inputProperties=[
        'B2','B3','B4','B5',
        'B6','B8','B11','B12'
    ]
)

image_RF = imageRGBNI.classify(rf)


# -------------------------
# VECTORISATION SIMPLE
# -------------------------

vecteurs = image_RF.reduceToVectors(
    geometry=ZONE,
    scale=10,
    geometryType='polygon',
    eightConnected=True,
    labelProperty='class',
    reducer=ee.Reducer.countEvery()
)

# -------------------------
# EXPORT GEE
# -------------------------

task = ee.batch.Export.table.toDrive(
    collection=vecteurs,
    description='Habitats_INRS_Laval_' + str(year),
    folder=EXPORT_FOLDER,
    fileFormat='SHP'
)

task.start()

print("Export lancé...")

# Attendre que la tâche soit terminée
while task.active():
    print("Export en cours...")
    time.sleep(30)

print("Export terminé.")

# -------------------------
# AUTH GOOGLE DRIVE
# -------------------------

SCOPES = ['https://www.googleapis.com/auth/drive']

creds = None

if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)

# Rafraîchir automatiquement
if creds and creds.expired and creds.refresh_token:
    creds.refresh(Request())

elif not creds or not creds.valid:
    flow = InstalledAppFlow.from_client_secrets_file(
        'credentials_Ulaval.json',
        SCOPES
    )
    creds = flow.run_local_server(port=0)

    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('drive', 'v3', credentials=creds)

# -------------------------
# ATTENDRE EXPORT
# -------------------------

print("Attente du fichier Drive...")

found = False

while not found:

    query = (
        "name contains 'Habitats_INRS_Laval_" 
        + str(year) +
        "' and trashed=false"
    )

    results = service.files().list(
        q=query,
        fields="files(id, name)"
    ).execute()

    files = results.get('files', [])

    if files:

        print("Fichiers trouvés :")

        for file in files:

            file_id = file['id']
            file_name = file['name']

            print("-", file_name)

            filepath = os.path.join(
                LOCAL_FOLDER,
                file_name
            )

            request = service.files().get_media(
                fileId=file_id
            )

            with open(filepath, 'wb') as f:
                f.write(request.execute())

            print("Téléchargé :", file_name)

            service.files().delete(fileId=file_id).execute()
            print("Supprimé du Drive :", file_name)

            # Vérifier si ZIP
            if zipfile.is_zipfile(filepath):

                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(LOCAL_FOLDER)

                print("Décompression terminée.")

        found = True

    else:

        print("Toujours en attente...")
        time.sleep(60)

