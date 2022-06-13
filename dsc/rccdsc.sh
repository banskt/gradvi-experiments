#!/bin/bash

TARGET=$1

if [ -z ${TARGET} ]; then
    echo "No target specified.";
    echo "Use this script as: ${BASH_SOURCE[0]} TARGET";
    echo "Options for TARGET: linreg, trendfilter";
    exit 1;
fi

if [[ ! "${TARGET}" == "linreg" ]] &&  [[ ! "${TARGET}" == "trendfilter" ]]; then
    echo "Fatal! Wrong target.";
    echo "Use this script as: ${BASH_SOURCE[0]} TARGET";
    echo "Options for TARGET: linreg, trendfilter";
    exit 1;
fi

function bash_confirm() {
    read -p "Do you want to run the program? " -r
    echo ""    # (optional) move to a new line
    if [[ ! ${REPLY} =~ ^[Yy]$ ]];
    then
        [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
    fi
}

LINREG_OUTDIR="/scratch/midway2/saikatbanerjee/work/sparse-regression/eb-linreg-dsc/dsc_result/linreg"
TRENDF_OUTDIR="/scratch/midway2/saikatbanerjee/work/sparse-regression/eb-linreg-dsc/dsc_result/trendfilter"

if [[ "${TARGET}" == "linreg" ]]; then
    echo "dsc linreg.dsc --target linreg -o ${LINREG_OUTDIR} --host midway2.yml"
    bash_confirm
    dsc linreg.dsc --target linreg      -o ${LINREG_OUTDIR} --host midway2.yml
elif [[ "${TARGET}" == "trendfilter" ]]; then
    echo "dsc linreg.dsc --target trendfilter -o ${TRENDF_OUTDIR} --host midway2.yml"
    bash_confirm
    dsc linreg.dsc --target trendfilter -o ${TRENDF_OUTDIR} --host midway2.yml
fi
