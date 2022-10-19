# XClone
# base script for RDR analysis using XClone NB+HMM pipeline
# Rongting Huang
# Date:2022-01-06
# execute env: TFProb, with XClone RDR module

print("RDR Analysis of CNVs (NB+HMM IN XClone)")
import os
import sys
from optparse import OptionParser, OptionGroup

import xclone
xclone.__version__

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import mmread


# from vireoSNP import BinomMixtureVB
from xclone.model.analysis_utils import sub_chr, select_chr_region, filter_data, filter_2data

# from xclone.model._RDR import 

import anndata as an

def main():
    # parse command line options
    parser = OptionParser(usage = "usage: %prog [options] arg1 arg2...", version="%prog 0.0.1")
    
    parser.add_option("--Xdata", "-d", dest="RDR_file", default=None,
        help="The RDR anndata for CNV ratio analysis, .h5ad file")
    parser.add_option("--CNVratio", "-c", dest="CNV_file", default=None,
        help="The CNV ratio numpy array for analysis,.npy file")
    parser.add_option("--CNVspecific", "-s", dest="CNV_specific", default=False,
        action='store_true', help="bool for CNV ratio is gene groups specific or not.")
 
    # parser.add_option("--random_seed", "-s", dest="random_seed", default=None,
    #     type=int, help="model fitting params")
    parser.add_option("--outDir", "-o", dest="out_dir", default=None,
        help="Output dir")

    # parser.add_option_group(group0)
    (options, args) = parser.parse_args()

    ## input data_file
    if options.RDR_file is None:
        print("Error: need RDR anndat file.")
        sys.exit(1)
    else:
        RDR_file = options.RDR_file
    
    if options.CNV_file is None:
        print("Error: need CNV ratio numpy array file.")
        sys.exit(1)
    else:
        CNV_file = options.CNV_file

    ## params setting
    CNV_specific = options.CNV_specific
    
    ## out directory for saving model and results and plots
    if options.out_dir is None:
        print("Warning: no outDir provided")
        sys.exit(1)
    else:
        out_dir=options.out_dir
        
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    ## load input data and preprocess

    ref_obs_ad1 = an.read_h5ad(RDR_file)
    
    CNV_ratio = np.load(CNV_file)

    ref_obs_ad1 = xclone.model.gene_exp_group(ref_obs_ad1, n_group = 5, verbose=True)

    if CNV_specific:
        specific_states = xclone.model.gene_specific_states(ref_obs_ad1, CNV_ratio)
    else:
        specific_states = CNV_ratio.copy()

    emm_prob_log = xclone.model.calculate_Xemm_prob(ref_obs_ad1, 
                                    states = specific_states,
                                    gene_specific = True, overdispersion = ref_obs_ad1.var["dispersion"],
                                    ref_normalization_term = 1,
                                    obs_normalization_term = ref_obs_ad1.obs["library_ratio"][1:])

    update_Xdata, res_dict, res_log_dict, res, res_log = xclone.model.XHMM_smoothing(ref_obs_ad1, emm_prob_log = emm_prob_log)

    res_prob_ad, res_cnv_ad, res_cnv_weights_ad, res_cnv_weights_ad1 = xclone.model.convert_res_visual(res_dict, update_Xdata)

    
    res_cnv_weights_ad1_re = xclone.pl.reorder_data_by_cellanno(res_cnv_weights_ad1, cell_anno_key="cell_type")
    
    plot_out_file = out_dir + "/Xheatmap.pdf"
    xclone.pl.Xheatmap(res_cnv_weights_ad1_re, cell_anno_key = 'cell_type', center=True,save_file=True, out_file = plot_out_file)


if __name__ == "__main__":
    main()