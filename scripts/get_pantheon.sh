#!/usr/bin/env bash
# Descarga Pantheon+SH0ES (distancias y covarianza STAT+SYS) en data/pantheonplus
# (no versionado) y comprueba las sumas SHA-256 de los ficheros usados en el Goal 4c.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/pantheonplus
cd data/pantheonplus
B="https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR"
curl -sSfL -o PantheonPlusSH0ES.dat "$B/Pantheon%2BSH0ES.dat"
curl -sSfL -o PantheonPlusSH0ES_STAT+SYS.cov "$B/Pantheon%2BSH0ES_STAT%2BSYS.cov"
sha256sum -c - <<'SUMS'
1cb0fc379ef066afdc2ffd1857681cc478024570d8a3eba284fb645775198cf8  PantheonPlusSH0ES.dat
abf806d966485e64afdb359c87bffc0ecc00d05eff0a31ced66f247385df0fdc  PantheonPlusSH0ES_STAT+SYS.cov
SUMS
