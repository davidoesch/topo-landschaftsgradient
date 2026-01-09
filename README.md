# topo-landschaftsgradient

Tools fürs Trockenheitsmonitoring

## Abhängigkeiten

Das Tool baut auf den Repository <https://github.com/ChristianSteger/HORAYZON> auf
Steger, C. R., Steger, B. and Schär, C. (2022):
HORAYZON v1.2: an efficient and flexible ray-tracing algorithm to compute
horizon and sky view factor, Geosci. Model Dev., 15, 6817–6840,
<https://doi.org/10.5194/gmd-15-6817-2022>

## Paketabhängigkeiten

Siehe auch Package dependencies HORAYZON

## Installation

Der Code basiert auf einem Python-Clone von ArcGIS Pro 3.6.
Zusätzlich sind folgende Module nötig:

- rasterio
- geographiclib
- skyfield
- horayzon-1.2.0-cp313-cp313-win_amd64.whl

### Windows

Dieser Code ist nur auf Windows lauffähig. Das horayzon*.whl wurde für Windows gebildet.

### Linux / Mac OS X

Für diese Plattformen kann direkt das Modul von 
<https://github.com/ChristianSteger/HORAYZON> genutzt werden.
