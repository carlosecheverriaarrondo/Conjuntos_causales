#!/usr/bin/env bash
# Descarga Pantheon+SH0ES (distancias y covarianza STAT+SYS) en data/pantheonplus
# (no versionado), y el BAO de DESI DR2 en data/desi_dr2 (Goal 4d); comprueba las
# sumas SHA-256 de todos los ficheros.
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

# BAO de DESI DR2 (DESI Collaboration, arXiv:2503.14738), versión gaussiana del
# likelihood de Cobaya, en data/desi_dr2 (Goal 4d).
cd ../..
mkdir -p data/desi_dr2
cd data/desi_dr2
D="https://raw.githubusercontent.com/CobayaSampler/bao_data/master/desi_bao_dr2"
curl -sSfL -O "$D/desi_gaussian_bao_ALL_GCcomb_mean.txt"
curl -sSfL -O "$D/desi_gaussian_bao_ALL_GCcomb_cov.txt"
sha256sum -c - <<'SUMS'
9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585  desi_gaussian_bao_ALL_GCcomb_mean.txt
252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509  desi_gaussian_bao_ALL_GCcomb_cov.txt
SUMS
