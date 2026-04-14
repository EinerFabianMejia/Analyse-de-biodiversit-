Execution Linux des plateformes

1. Aller dans le dossier du depot :
   cd ~/Projet-2

2. Donner les droits d'execution aux scripts :
   chmod +x linux/lancer_ulaval_linux.sh
   chmod +x linux/lancer_inrs_linux.sh

3. Lancer ULaval :
   ./linux/lancer_ulaval_linux.sh

4. Lancer INRS dans un autre terminal :
   ./linux/lancer_inrs_linux.sh

5. Ouvrir dans le navigateur :
   ULaval : http://127.0.0.1:8000
   INRS   : http://127.0.0.1:8001

6. Si acces depuis une autre machine :
   remplacer 127.0.0.1 par l'adresse IP du serveur.

Notes :
- ULaval utilise le port 8000.
- INRS utilise le port 8001.
- Les deux serveurs sont configures pour ecouter sur 0.0.0.0.
- GeoServer doit etre accessible sur http://localhost:8080/geoserver
