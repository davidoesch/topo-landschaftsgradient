# topo-landschaftsgradient

Outils pour le monitoring de la sécheresse

## Dépendances

Cet outil s'appuie sur le dépôt <https://github.com/ChristianSteger/HORAYZON>
Steger, C. R., Steger, B. and Schär, C. (2022):
HORAYZON v1.2: an efficient and flexible ray-tracing algorithm to compute
horizon and sky view factor, Geosci. Model Dev., 15, 6817–6840,
<https://doi.org/10.5194/gmd-15-6817-2022>


## Installation

Les modules supplémentaires suivants sont nécessaires :

- rasterio
- xarray
- matplotlib
- shapely
- tqdm
- requests
- geographiclib
- scipy
- fiona
- scikit-image
- skyfield
- netcdf4
- pvlib
- horayzon

### Windows

Les paquets précompilés :
- `Miniconda3-latest-Windows-x86_64.exe`
- `winhorayzon-1.2.0-cp313-cp313-win_amd64.whl`

#### Mise en place de l'environnement Python
```bat
conda create --name topo-winhorayzon python=3.13 --yes
pip install winhorayzon-1.2.0-cp313-cp313-win_amd64.whl
pip install rasterio xarray matplotlib shapely tqdm requests geographiclib scipy fiona scikit-image skyfield netcdf4
pip install --index https://gisidx.github.io/gwi gdal
```

> **Remarque :** L'index GDAL `gisidx.github.io/gwi` est nécessaire car aucun paquet GDAL officiel n'existe pour Python 3.13.

Vérifier l'installation :
```bat
python.exe -c "import horayzon; print('horayzon OK')"
```

## Utilisation

### Entrées

- `DOM.tif` ou `DSM.tif` – Modèle numérique de surface ou de terrain
- `de421.bsp` – Éphéméride planétaire (Skyfield)

### Sorties

- `Incidence_{perimeter}_{DOM|DSM}_{DOY}_{date}_{time}.tif` – Angle d'incidence
- `Illuminated_{perimeter}_{DOM|DSM}_{DOY}_{date}_{time}.tif` – Luminosité (binaire)

### Configuration (`Iluina.json`)

Les chemins d'accès aux fichiers, les chemins de sortie et `search_dist` (par défaut : 20 000 m) sont définis dans `Iluina.json`.

### Exécution (`Iluina_parallel.py`)

Paramètres importants :

| Paramètre | Description | Exemple |
|-----------|-------------|---------|
| `--date` | Date | `25.12.2023` ou  `17.06.2025`|
| `--time` | Heure | `10:34:41` ou `10:26:21`|
| `--east` | Coordonnée de départ Est (LV95) | `2480000` (CH) |
| `--north` | Coordonnée de départ Nord (LV95) | `1060000` (CH) |
| `--grid_size` | Taille de la grille [m] | `20000` |
| `--grid_step` | Résolution [m] | `10` |
| `--perimeter` | Périmètre de calcul | `CH`, `8`, `22`, `65`, `108` |

Pour toute la Suisse : `n_e = 18`, `n_n = 12`.  
Pour un tile individuel (p. ex. Niesen) : `n_e = 1`, `n_n = 1`.

Multiprocessing : `n_pro = 8` workers par défaut.

**Définir DOM ou DSM :** Dans `Iluina_parallel.py`, modifier le chemin aux lignes 143, 196, 246 et 480 (`cfg["dom_path"]` ou `cfg["dsm_path"]`).

## Fichiers

| Fichier | Description |
|---------|-------------|
| `Iluina_module.py` | Classes `HelperFunctions`, `DOM_sw`/`SonneWinkel` (luminosité), `DOM_iw`/`InzidenWinkel` (angle d'incidence) |
| `Iluina_parallel.py` | Script principal avec parseur d'arguments et multiprocessing |
| `Iluina.json` | Configuration : chemins d'accès et paramètres |