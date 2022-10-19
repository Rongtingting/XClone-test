# Script for getting the clustering results using vireoSNP_binomialMixVB
: ex: set ft=markdown ;:<<'```shell' #

Author: Rongting Huang
Date: 2022-01-06

## Notes for BAF_VireoSNP_binomialMixVB_select_chr_region.py
conda activate TFProb
(TFProb) [rthuang@hpc02 bin]$ python CNV_ratio.py --version
RDR Analysis of CNVs (NB+HMM IN XClone)
CNV_ratio.py 0.0.1

(TFProb) [rthuang@hpc02 bin]$ python CNV_ratio.py -h
RDR Analysis of CNVs (NB+HMM IN XClone)
Usage: CNV_ratio.py [options] arg1 arg2...

Options:
  --version             show program's version number and exit
  -h, --help            show this help message and exit
  -d RDR_FILE, --Xdata=RDR_FILE
                        The RDR anndata for CNV ratio analysis, .h5ad file
  -c CNV_FILE, --CNVratio=CNV_FILE
                        The CNV ratio numpy array for analysis,.npy file
  -o OUT_DIR, --outDir=OUT_DIR
                        Output dir
(TFProb) [rthuang@hpc02 bin]$

"""

## data preparation

```shell
export src_path='/home/rthuang/Github_repos/XClone/xclone/bin'
export rdr_dat_dir='/storage/yhhuang/users/rthuang/xclone/develop/'
export RDR_file=${rdr_dat_dir}'GX109_anndata_ref_obs_lib_dispersion_selected.h5ad'
export CNV_file='/storage/yhhuang/users/rthuang/xclone/develop/CNV_ratio/test1/CNV_ratio.npy'
export out_dir='/storage/yhhuang/users/rthuang/xclone/develop/CNV_ratio/test1'
:<<'```shell' # Ignore this line
```

## Run XClone EM

```shell
cd ${src_path}
# demo1
python CNV_ratio.py -d ${RDR_file} -c ${CNV_file} -s -o ${out_dir}
# demo2
# python CNV_ratio.py -d ${RDR_file} -c ${CNV_file} -o ${out_dir}
exit $?
```
