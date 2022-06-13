#!/bin/bash

#DSC_CMD="python /home/saikat/Documents/work/dsc/main.py"
DSC_CMD="dsc"
TARGET=${2:-all}

trial_outdir() {
    TARGET="${1}"
    case ${TARGET} in
        "all")
            OUTDIR="trial"
            ;;
        "cpt")
            OUTDIR="trial_changepoint"
            ;;
        *)
            OUTDIR="trial"
            ;;
    esac
    echo ${OUTDIR}
}

case $1 in
    "append")
        ${DSC_CMD} linreg.dsc --target ${TARGET} -c 16 -s existing
        ;;
    "trial-append")
        OUTDIR=$(trial_outdir ${TARGET})
        ${DSC_CMD} linreg.dsc --target ${TARGET} --replicate 1 -c 16 -s existing -o ${OUTDIR}
        ;;
    "trial-rerun")
        OUTDIR=$(trial_outdir ${TARGET})
        rm -rf ${OUTDIR} ${OUTDIR}.html
        ${DSC_CMD} linreg.dsc --target ${TARGET} --replicate 1 -c 16 -s none -o trial -o ${OUTDIR}
        ;;
    *)
        ${DSC_CMD} "$@"
        ;;
esac
