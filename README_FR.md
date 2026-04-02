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

## Mise en place de l'environnement Python

### 1. Environnement Python de compilation
```bat
conda create --prefix ...dossier\make-topo-winhorayzon python=3.13
conda install -c conda-forge cmake --prefix ...dossier\make-topo-winhorayzon --yes
```

Vérifier cmake :
```bat
...dossier\make-topo-winhorayzon\Library\bin\cmake.exe --version
```

Installer les dépendances Python de compilation :
```bat
set PATH=...dossier\make-topo-winhorayzon;%PATH%
cd ...dossier\make-topo-winhorayzon\scripts
pip install cython numpy setuptools wheel
```

---

### 2. MinGW64

1. Télécharger depuis <https://github.com/niXman/mingw-builds-binaries/releases>
   - Fichier : `x86_64-15.2.0-release-win32-seh-msvcrt-rt_v12-rev0.7z` (ou version la plus récente)
   - Important : choisir la variante `win32`, `seh`, `msvcrt`
2. Extraire dans `...dossier\mingw64`

Vérifier l'installation :
```bat
...dossier\mingw64\bin\gcc.exe --version
...dossier\mingw64\bin\g++.exe --version
```

---

### 3. oneTBB
```bat
CMD à commencer dans ...\git
git clone https://github.com/oneapi-src/oneTBB.git
cd oneTBB
set PATH=...dossier\mingw64\bin;%PATH%
```

#### Patch de `dynamic_link.cpp`

Ouvrir `...\git\oneTBB\src\tbb\dynamic_link.cpp` et remplacer à la ligne ~562 :
```cpp
// AVANT
return LOAD_LIBRARY_SAFE_CURRENT_DIRS;

// APRÈS
return LOAD_LIBRARY_SEARCH_USER_DIRS;
```

#### Compilation
```bat
mkdir build
cd build
...dossier\make-topo-winhorayzon\Library\bin\cmake.exe..-G "MinGW Makefiles"-DCMAKE_MAKE_PROGRAM=...dossier/mingw32-make.exe -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=...dossier/mingw64/bin/gcc.exe -DCMAKE_CXX_COMPILER=...dossier/mingw64/bin/g++.exe -DTBB_TEST=OFF -DCMAKE_INSTALL_PREFIX=.../git/tbb-mingw -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0602" -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0602"
mingw32-make.exe -j4
mingw32-make.exe install
copy ...\git\tbb-mingw\lib\libtbb12.dll.a ...\git\tbb-mingw\lib\libtbb12.a
```

---

### 4. Bibliothèque d'import Embree
```bat
cd ...\git\topo-winhorayzon\embree\bin
...dossier\mingw64\bin\gendef.exe embree4.dll
...dossier\mingw64\bin\dlltool.exe -d embree4.def -l libembree4.a -D embree4.dll
```

---

### 5. Compilation du module et création du wheel
```bat
set PATH=...dossier\make-topo-winhorayzon;%PATH%
set PATH=...dossier\mingw64\bin;%PATH%
python setup_windows.py build_ext --inplace
python build_wheel.py
```

---

### 6. Environnement Python d'utilisation
Dans un nouveau CMD
```bat
_conda create --prefix ...dossier\topo-winhorayzon python=3.13 --yes
cd ...dossier\topo-winhorayzon\scripts
pip install "...\git\topo-winhorayzon\whl\winhorayzon-1.2.0-cp313-cp313-win_amd64.whl"
pip install rasterio xarray matplotlib shapely tqdm requests geographiclib scipy fiona scikit-image skyfield netcdf4
pip install --index https://gisidx.github.io/gwi gdal
```

> **Remarque :** L'index GDAL `gisidx.github.io/gwi` est nécessaire car aucun paquet GDAL officiel n'existe pour Python 3.13.

Vérifier l'installation :
```bat
cd ...dossier\topo-winhorayzon
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