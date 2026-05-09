import ee
import datetime
import time
import os
import zipfile

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

from geo.Geoserver import Geoserver

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


# -----------------------------------
# CHARGER DANS POSTGRESQL
# -----------------------------------

shp_path = LOCAL_FOLDER + "/" + 'Habitats_INRS_Laval_' + str(year) + ".shp"
table_name = "temp"
db_host = "localhost"
db_name = "cartographie_biodiversite_INRS_LAVAL"
db_user = "postgres"

cmd = f'shp2pgsql -I -s 4326 "{shp_path}" {table_name} | psql -h {db_host} -d {db_name} -U {db_user}'
try:
    subprocess.run(cmd, shell=True, check=True)
    print("Shapefile imported successfully.")
except subprocess.CalledProcessError as e:
    logging.error(f"Error importing shapefile: {e}")
    print("Failed to import shapefile.")

query = f"""
CREATE TABLE IF NOT EXISTS classification_INRS_LAVAL_{year} (
    id SERIAL,
    nom_classe VARCHAR(255),
    annee INT,
    num_classe INT,
    superficie DEC(32,3),
    geom GEOMETRY(MULTIPOLYGON, 4326),
    PRIMARY KEY(id)
);

INSERT INTO classification_INRS_LAVAL_{year} (nom_classe, annee, num_classe, superficie, geom)
SELECT nom_classe, {year}, class, surface_m2, geom
FROM {table_name};

DROP TABLE {table_name};
"""

cmd = f'psql -h {db_host} -d {db_name} -U {db_user} -c "{query}"'

try:
    subprocess.run(cmd, shell=True, check=True)
    print("Table rearranged successfully.")
except subprocess.CalledProcessError as e:
    logging.error(f"Error rearranging table: {e}")
    print("Failed to rearrange table.")

# -----------------------------------
# Publier dans GeoServer
# -----------------------------------

geo = Geoserver('http://localhost:8080/geoserver', username='admin', password='geoserver')

geo.publish_featurestore(
    workspace='ne',
    store_name='Connexion_PostGIS',
    pg_table= "classification_INRS_Laval" + str(year)
)

workspace = 'ne'
sld_path = "/home/projet-genie/Téléchargements/Projet-2-linus/classification_ulaval.sld"
layer_name = "classification_ulaval_" + str(year)
style_name = 'classification_style'

geo.upload_style(path=sld_path, workspace=workspace, style_name=style_name)
geo.publish_style(layer_name=layer_name, style_name=style_name, workspace=workspace)


