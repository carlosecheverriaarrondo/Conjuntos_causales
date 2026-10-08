#!/usr/bin/env bash
# Reproduce los resultados del proyecto.
#
#   ./reproduce.sh fast   figuras y tablas a partir de los JSON guardados en
#                         figures/ (~1 min) y suite de tests (~8 min)
#   ./reproduce.sh full   recalcula todo desde cero con las semillas fijas de
#                         cada script (muchas horas; ver tiempos abajo) y
#                         después hace lo mismo que "fast"
#
# Entorno: python3 -m venv venv && source venv/bin/activate &&
#          pip install -r requirements.txt
set -euo pipefail
cd "$(dirname "$0")"
source venv/bin/activate
export MPLBACKEND=Agg

analysis() {
    python scripts/goal3_fluctuations.py plot
    python scripts/smeared_fluctuations.py plot
    python scripts/goal3c_asymptotic.py fit
    python scripts/goal3d_log_origin.py fit
    python scripts/step2_4d_scaling.py fit
    python scripts/cone_fluctuations.py fit
    python scripts/flrw_fluctuations.py fit
    python scripts/cone4d_scaling.py fit
    python scripts/flrw4d_scaling.py fit
    python scripts/goal4_compatibility.py
}

full() {
    python scripts/goal1_convergence.py           # Goal 1
    python scripts/goal2_bd_action.py             # Goal 2
    python scripts/goal3_fluctuations.py run      # Goal 3            (~20 min)
    python scripts/smeared_fluctuations.py run2d  # ampliación, 2D    (~1 h)
    python scripts/smeared_fluctuations.py run4d  # ampliación, 4D    (~1 h)
    python scripts/goal3c_asymptotic.py sim       # Goal 3c           (~1.5 h)
    python scripts/goal3c_asymptotic.py pred      #                   (~1 h)
    python scripts/goal3d_log_origin.py terms     # Goal 3d           (~1 h)
    python scripts/goal3d_log_origin.py rapidity  #                   (~10 min)
    python scripts/step2_4d_scaling.py sim        # Paso 2            (~11 min)
    python scripts/step2_4d_scaling.py extra      #                   (~1 h)
    python scripts/cone_fluctuations.py sim       # Punto 1a          (~1.5 h)
    python scripts/cone_fluctuations.py pred      #                   (~30 min)
    python scripts/flrw_fluctuations.py sim       # Punto 1b          (~1 h)
    python scripts/flrw_fluctuations.py pred      #                   (~1 h)
    python scripts/cone4d_scaling.py sim          # Punto 1c          (~1.5 h)
    python scripts/flrw4d_scaling.py means 1      # Punto 1d          (~15 min)
    python scripts/flrw4d_scaling.py means 2      #                   (~15 min)
    python scripts/flrw4d_scaling.py sim 1        #                   (~1 h)
    python scripts/flrw4d_scaling.py sim 2        #                   (~1 h)
}

case "${1:-fast}" in
    fast) analysis ;;
    full) full; analysis ;;
    *) echo "uso: $0 [fast|full]"; exit 1 ;;
esac
python -m pytest -q
