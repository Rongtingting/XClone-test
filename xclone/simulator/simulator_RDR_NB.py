"""Base functions for RDR negative binomial model simulation."""

# Simulator for generating expression count matrices for single-cell RNA-seq
# Author: Rongting Huang
# Date: 2021-12-22

import numpy as np
import pandas as pd

    

def CNV_RDR_simulator(n_genes, n_cells, n_clones, CNVs, random_seed=None):
    """
    Function:
    define a simulation dataset
    
    used in different strategy!!
    
    Parameters:
    ----------
    
    Return:
    ------
    
    Example:
    -------
    
    """
    if random_seed is not None:
        rvgen = np.random.RandomState(random_seed)
    else:
        rvgen = np.random

    
    ## generate cell cluster labels; uniform distribution
    I_RNA = rvgen.choice(n_clones, size=n_cells)

    ## generate parameters for RDR, negative binomial distribution

    ## Generate RDR matrix: amplify the normal counts and then sample



# Two compoments of Negative binomial distribution

_GEX_ref = 10 # just as a toy example, we need it for each gene

_mu = np.array([0.5, 1.0], dtype=np.float32) * _GEX_ref # example for copy loss and neutral
_var = _mu + 0.1 * _mu**2 # example over dispersion

# reparameterization
_nb_prob = 1 - _mu / _var
_nb_total = _mu * (1 - _nb_prob) / _nb_prob

NB_distribution = tfd.NegativeBinomial(total_count=_nb_total, probs=_nb_prob)

def generate_NB_data():
    """
    Function:
    
    
    Parameters:
    ----------
    
    Return:
    ------
    
    Example:
    -------
    
    """
    

def assign_copy_numbers(chrs, tl, p_ins, min_cn, max_cn, cnv_list_st):
    """
    """
	copy_num = []
	cn = {}
	for ch in chrs:
		cn[ch] = []
	num_ins = int(tl * p_ins)
	num_del = tl - num_ins
	for i in range(num_del):
		copy_num.append(0)
	for i in range(num_ins):
		copy_num.append(random.randrange(min_cn,max_cn+1))
	random.shuffle(copy_num)
	j = 0
	for ch in chrs:
		for i in range(len(cnv_list_st[ch])):
			cn[ch].append(copy_num[j])
			j += 1
	return cn

        
def assign_cnv_pos(chrs, st, ed, num_cnv_list, cnv_min_len, cnv_max_len, \
	overlap_bp, seqs, method_s, method_l, cnv_listl, ran_m, flank, \
	alphas, betas, alphal, betal):
    """
    """
    pass



def CNV_ASE_RDR_simulator(tau, T_mat, normal_DP_DNA, normal_DP_RNA,
                          total_DP_DNA, total_DP_RNA,
                          n_cell_DNA=200, n_cell_RNA=200, 
                          beta_shape_DNA=30, beta_shape_RNA=1,
                          share_theta=True, random_seed=None):
    """
    First version, not supporting CNV=0
    
    tau: (n_state, 2), array_like of ints
        copy number of m & p for each CNV state.
        In future, we could change this variable 
        to theta_prior
    T_mat: (n_block, n_clone), array_like of ints
        clone configuration of copy number states
    
    """
    if random_seed is not None:
        rvgen = np.random.RandomState(random_seed)
    else:
        rvgen = np.random
        
    n_state = tau.shape[0]
    n_block = T_mat.shape[0]
    n_clone = T_mat.shape[1]
    gamma = tau.sum(axis=1) # amplification factor

    ## generate cell cluster labels; uniform distribution
    I_RNA = rvgen.choice(n_clone, size=n_cell_RNA)
    I_DNA = rvgen.choice(n_clone, size=n_cell_DNA)
    
    ## generate Theta parameter for ASR; beta distribution
    _base = 0.01  # avoiding bafs to be 0 or 1
    _theta_prior = (tau[:, 0] + _base) / (tau + _base).sum(axis=1)
    
    Theta_DNA = np.zeros((n_block, n_state))
    Theta_RNA = np.zeros((n_block, n_state))
    for j in range(Theta_DNA.shape[1]):
        _s1 = beta_shape_DNA * _theta_prior[j]
        _s2 = beta_shape_DNA * (1 - _theta_prior[j])
        Theta_DNA[:, j] = rvgen.beta(_s1, _s2, size=n_block)
        
        if share_theta:
            Theta_RNA[:, j] = Theta_DNA[:, j]
        else:
            _s1 = beta_shape_RNA * Theta_DNA[:, j]
            _s2 = beta_shape_RNA * (1 - Theta_DNA[:, j])
            _s2[_s2 == 0] += 0.01
            Theta_RNA[:, j] = rvgen.beta(_s1, _s2)
    
    ## Generate DP matrix: amplify the normal counts and then sample
    def generate_DP_matrix(normal_DP, total_DP, clonal_labels, n_cell):
        # Compute the perfect-case amplified DP counts for each cell
        DP_seed = np.array([
            [
                normal_DP[i] * gamma[T_mat[i, clonal_labels[j]]]
                for j in range(n_cell)
            ]
            for i in range(n_block)
        ])
        # Then compute the perfect-case probabilities for each cell
        DP_probs = DP_seed / DP_seed.sum(axis=0, keepdims=1)
        # Sample DP counts under the multinomial model for each cell 
        DP = np.column_stack([
            np.ravel(rvgen.multinomial(total_DP, DP_probs[:, j], size=1))
            for j in range(n_cell)
        ])
        return DP
    
    DP_RNA = generate_DP_matrix(normal_DP_RNA, total_DP_RNA, I_RNA, n_cell_RNA)
    DP_DNA = generate_DP_matrix(normal_DP_DNA, total_DP_DNA, I_DNA, n_cell_DNA)
    
    ## Generate X and AD matrices: binomial distribution
    X_RNA = np.zeros(DP_RNA.shape)
    for i in range(n_block):
        for j in range(n_cell_RNA):
            X_RNA[i, j] = Theta_RNA[i, int(T_mat[i, I_RNA[j]])]
    AD_RNA = rvgen.binomial(DP_RNA, X_RNA)
    
    X_DNA = np.zeros(DP_DNA.shape)
    for i in range(n_block):
        for j in range(n_cell_DNA):
            X_DNA[i, j] = Theta_DNA[i, int(T_mat[i, I_DNA[j]])]
    AD_DNA = rvgen.binomial(DP_DNA, X_DNA)
            
    ## return values
    RV = {}
    RV["tau"] = tau
    RV["amplification_factors"] = gamma
    RV["T_mat"] = T_mat
    RV["I_RNA"] = I_RNA
    RV["I_DNA"] = I_DNA
    RV["X_RNA"] = X_RNA
    RV["X_DNA"] = X_DNA
    RV["DP_RNA"] = DP_RNA
    RV["DP_DNA"] = DP_DNA
    RV["AD_RNA"] = AD_RNA
    RV["AD_DNA"] = AD_DNA
    RV["Theta_RNA"] = Theta_RNA
    RV["Theta_DNA"] = Theta_DNA
    return RV


    