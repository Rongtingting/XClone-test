import itertools
import os
import sys
from typing import Iterable

import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import matplotlib.pyplot as plt
plt.switch_backend("agg")
import seaborn as sns
import plotly.express as px
from tqdm.auto import tqdm

FullNormal = tfd.MultivariateNormalFullCovariance


def logfact(n):
    """
    Ramanujan's approximation of log n!
    https://math.stackexchange.com/questions/138194/approximating-log-of-factorial
    """
    n = np.atleast_1d(n)
    n[n == 0] = 1  # for computational convenience
    return (
        n * (np.log(n) - 1)
        + np.log((n * (1. + 4. * (n * (1. + 2. * n))))) / 6.
        + np.log(np.pi) / 2.
    )


def logbincoeff(n, k):
    """
    Ramanujan's approximation of log [n! / (k! (n-k)!)]
    """
    n_logfact = logfact(n)
    k_logfact = logfact(k)
    n_sub_k_logfact = logfact(n - k)
    return n_logfact - k_logfact - n_sub_k_logfact


def categorical_kl(posterior, prior):
    return posterior.mean() * tf.math.log(posterior.mean() / prior.mean())


class XCloneASE(tf.Module):
    def __init__(self, xclone_base_model):
        self.xc = xclone_base_model
        self.logbincoeffs = logbincoeff(self.xc.total_counts, self.xc.alt_counts)
        # Variational distribution variables for allelic ratio

        self.baf_beta_params_log = []
        # for i in range(2):
        #     tf.random.set_seed(self.xc.seed)
        #     param_tensor = tf.random.uniform((self.xc.n_segments, self.xc.n_states), 1, 2,
        #                                      seed=self.xc.seed, dtype=tf.float32)
        #     self.baf_beta_params_log.append(tf.Variable(tf.math.log(param_tensor)))

        self.baf_prior = None
        self.set_prior_()

        for i in range(2):
            self.baf_beta_params_log.append(tf.constant(tf.math.log(self.baf_beta_prior_params[i] + 1e-10)))

    @property
    def kl(self):
        return tfd.kl_divergence(self.bafs, self.baf_prior)

    @property
    def kl_sum(self):
        return tf.reduce_sum(self.kl)

    @property
    def log_lik(self):
        # Variational distributions (Nb, Nc, Nk, Ns)
        AD = tf.reshape(self.xc.alt_counts, (self.xc.n_segments, self.xc.n_cells, 1))  # depth of A allele
        DP = tf.reshape(self.xc.total_counts, (self.xc.n_segments, self.xc.n_cells, 1))  # depth of A & B allele
        BD = DP - AD
        logbincoeffs = tf.reshape(self.logbincoeffs, (self.xc.n_segments, self.xc.n_cells, 1))

        # marginalise element CNV state probability
        # sum_{i=1}^{N} sum_{j=1}^{M} sum_{k=1}^{K} sum_{t=1}^{T} (...)
        clonal_label_probs = tf.reshape(self.xc.clonal_label_probs.mean(),
                                       (1, self.xc.n_cells, self.xc.n_clones, 1))
        cnv_state_probs = tf.reshape(self.xc.cnv_state_probs.mean(),
                                     (self.xc.n_segments, 1, self.xc.n_clones, self.xc.n_states))
        weights = tf.reduce_sum(clonal_label_probs * cnv_state_probs, axis=2)

        # marginalise allelic ratio mess
        alt_beta_params, ref_beta_params = self.baf_beta_params
        digamma_ad = tf.reshape(tf.math.digamma(alt_beta_params), (self.xc.n_segments, 1, self.xc.n_states))
        digamma_bd = tf.reshape(tf.math.digamma(ref_beta_params), (self.xc.n_segments, 1, self.xc.n_states))
        digamma_dp = tf.reshape(tf.math.digamma(alt_beta_params + ref_beta_params),
                                (self.xc.n_segments, 1, self.xc.n_states))

        log_lik_ase = weights * (
                logbincoeffs
                + AD * digamma_ad
                + BD * digamma_bd
                - DP * digamma_dp
        )
        return tf.reduce_sum(log_lik_ase)

    def set_prior_(self, baf_prior=None):
        if baf_prior is None:
            prior_amplifier = 50
            eps = 0.01

            allelic_copy_numbers = [self.xc.cnv_states[:, i] + eps for i in range(2)]
            total_copy_numbers = tf.reduce_sum(self.xc.cnv_states, axis=1)
            self.baf_beta_prior_params = [tf.reshape(
                tf.tile(prior_amplifier * (allelic_copy_numbers[i] / total_copy_numbers),
                        [self.xc.n_segments]),
                (self.xc.n_segments, self.xc.n_states)
            ) for i in range(2)]
            self.baf_prior = tfd.Beta(self.baf_beta_prior_params[0], self.baf_beta_prior_params[1])

    @property
    def baf_beta_params(self):
        return [tf.math.exp(self.baf_beta_params_log[i]) for i in range(2)]

    @property
    def bafs(self):
        """Variational posterior for ASE ratio"""
        beta_params = self.baf_beta_params
        return tfd.Beta(beta_params[0], beta_params[1])


class XCloneRDR(tf.Module):
    def __init__(self, xclone_base_model, gp_covariance_fn=None, n_mc_samples=0):
        self.xc = xclone_base_model
        if gp_covariance_fn is None:
            gp_covariance_fn = lambda diff: np.exp(-0.5 * np.sum(diff ** 2))
        self.gp_cov_fn = gp_covariance_fn
        self.n_mc_samples = n_mc_samples
        # Variational distribution variables for depth ratio
        self.amplification_factors_prior = None
        self.gp_mean = None
        self.gp_cov = None
        self.set_prior_()

    @property
    def kl(self):
        return tfd.kl_divergence(self.amplification_factors,
                                 self.amplification_factors_prior)

    @property
    def kl_sum(self):
        return tf.reduce_sum(self.kl)

    @property
    def log_lik(self):
        """
        Compute the log-likelihood of expression count distribution.
        :return: tf.Tensor(dtype=tf.float), Nb x Nk
        """

        # Perform triple nested summation in one tensor multiplication.
        # L_RDR(params) = sum_{i=1}^{Nb} sum_{j=1}^{Nc} sum_{k=1}^{Nk} (...) (see manuscript)
        clonal_label_probs = tf.reshape(self.xc.clonal_label_probs.mean(), (1, self.xc.n_cells, self.xc.n_clones))
        total_counts = tf.reshape(self.xc.total_counts_all_snps, (self.xc.n_segments, self.xc.n_cells, 1))
        log_lik_weights = tf.reduce_sum(clonal_label_probs * total_counts, axis=1)

        if self.n_mc_samples > 0:
            # We followed the example given in the official TFP documentation page.
            mc_estimates_clonal_rdr = (
                tfp.monte_carlo.expectation(
                    f=lambda mc_sample: self.log_clonal_rdr_(mc_sample),
                    samples=self.amplification_factors.sample(self.n_mc_samples, seed=42),
                    log_prob=self.amplification_factors.log_prob,
                    use_reparameterization=(
                        self.amplification_factors.reparameterization_type
                        == tfp.distributions.FULLY_REPARAMETERIZED
                    )
                )
            )
            log_lik_rdr = tf.multiply(log_lik_weights, mc_estimates_clonal_rdr)
        else:
            raise NotImplementedError("Closed-form expression for RDR-likelihood has not been derived yet.")
        return tf.reduce_sum(log_lik_rdr)

    def set_gp_kernel_(self):
        gp_cov = np.zeros((self.xc.n_states, self.xc.n_states), dtype=np.float32)
        for i, j in itertools.product(range(self.xc.n_states), repeat=2):
            diff = self.xc.cnv_states[i, :] - self.xc.cnv_states[j, :]
            gp_cov[i, j] = self.gp_cov_fn(diff)
        self.gp_cov = tf.constant(gp_cov)

    def set_prior_(self, gamma_prior=None):
        # Prior distributions for the depth ratio
        if gamma_prior is None:
            if self.gp_cov is None:
                self.set_gp_kernel_()
            eps = tf.constant(1e-10, dtype=tf.float32)
            self.amplification_factors_prior = FullNormal(
                loc=tf.math.log(tf.reduce_sum(self.xc.cnv_states, axis=1) / 2 + eps),
                covariance_matrix=self.gp_cov
            )
        else:
            self.amplification_factors_prior = gamma_prior
        self.gp_mean = tf.constant(self.amplification_factors_prior.mean())

    def clonal_rdr_(self, amplification_factors=None):
        """
        Helper function needed for Monte-Carlo integration routine.
        :param amplification_factors: tf.Tensor(tf.float32), n_draws x dim(amplification_factors)
        Draws of logarithms of amplification factors from posterior distribution.
        :return: tf.Tensor(tf.float32), n_segments x Nk
        """
        if amplification_factors is None:
            amplification_factors = tf.expand_dims(self.amplification_factors.mean(), 0)
        amplification_factors = tf.math.exp(amplification_factors)  # To ensure non-negativity of amplification factor
        n_mc_draws = amplification_factors.shape[0]  # tfp.monte_carlo samples all the draws in one go
        # self.cnv_state_probs.mean() is a 3-tensor of posterior CNV state probabilities.
        # As amplification_factors has a batch dimension, it needs to be transposed first to make
        # the dimensions compatible. Then, the dimensions are permuted to ensure
        # that batch dimension (arising from sampling) is zeroth.
        # print(amplification_factors)
        weights = tf.transpose(
            self.xc.cnv_state_probs.mean() @ tf.transpose(amplification_factors),
            (2, 0, 1)
        )
        tf.assert_equal(tf.shape(weights), (n_mc_draws, self.xc.n_segments, self.xc.n_clones),
                        message="Wrong shape of `weights` tensor in `clonal_rdr_`:\n"
                                f"\t-Expected: {(n_mc_draws, self.xc.n_segments, self.xc.n_clones)}\n"
                                f"\t-Found: {tf.shape(weights)}")
        # print(weights)
        # These weights are then multiplied with baseline expression levels
        unnormalized_clonal_rdrs = tf.multiply(
            tf.reshape(
                self.xc.baseline_expression,
                (1, self.xc.n_segments, 1)
            ),
            weights
        )
        tf.assert_equal(tf.shape(unnormalized_clonal_rdrs), (n_mc_draws, self.xc.n_segments, self.xc.n_clones),
                        message="Wrong shape of `weights` tensor in `clonal_rdr_`:\n"
                                f"\t-Expected: {(n_mc_draws, self.xc.n_segments, self.xc.n_clones)}\n"
                                f"\t-Found: {tf.shape(unnormalized_clonal_rdrs)}")
        # and are converted into probabilities by dividing each column with sum of its elements.
        col_normalizer = tf.reduce_sum(unnormalized_clonal_rdrs, axis=1)
        # Unfortunately, some reshaping magic is necessary here.
        clonal_rdrs = tf.multiply(
            tf.reshape(1 / col_normalizer, (n_mc_draws, 1, self.xc.n_clones)),
            unnormalized_clonal_rdrs
        )

        tf.debugging.assert_near(
            tf.reduce_sum(clonal_rdrs, axis=1), 1.,
            atol=1e-4,
            message="clonal_rdr_ matrix doesn't define a count probability distribution"
        )
        return clonal_rdrs

    @property
    def amplification_factors(self):
        """Variational posterior for distribution mean"""
        return FullNormal(self.gp_mean, self.gp_cov)

    def log_clonal_rdr_(self, gamma=None):
        # If Nb is large, most of the Fs in each column will be indistinguishable from zero.
        # To avoid numerical problems, we multiply the probabilities by an arbitrarily chosen
        # reasonably large amplifier and then subtract it's log from the total log-likelihood.
        # This little hack turns out to be really useful in practice as it prevents underflow.
        # We also add a small positive epsilon under the logarithm, just in case.
        clonal_rdr = self.clonal_rdr_(amplification_factors=gamma)
        eps = tf.constant(1e-10, dtype=tf.float32)
        log_amplifier = tf.constant(self.xc.n_segments, dtype=tf.float32)
        log_f = tf.math.log(log_amplifier * (clonal_rdr + eps))
        # However, some logits may still take the value of NaN or NEGINF.
        # If this happens, they get replaced with a large negative number to prevent overflow.
        minf = -100  # https://xkcd.com/221/
        log_f = tf.where(
            tf.math.is_nan(log_f) | (~tf.math.is_finite(log_f)),
            tf.ones_like(log_f) * minf,
            log_f
        )
        tf.debugging.assert_all_finite(log_f, message="Some of logits of clonal_rdr_ are not finite.")
        return log_f - tf.math.log(log_amplifier)


class XCloneVB(tf.Module):
    """
    A Bayesian Binomial mixture model for CNV clonal lineage reconstruction.

    :param n_segments: int > 0
        Number of blocks, similar as genes.
    :param n_cells: int > 0
        Number of cells.
    :param n_clones: int > 0
        Number of clones in the cell population
    :param cnv_states: numpy.array (Ns, 2)
        The CNV states, paternal copy numbers and maternal copy numbers
    """
    def __init__(
            self, n_segments, n_cells, n_clones, cnv_states,
            alt_counts, total_counts, total_counts_all_snps,
            baseline_expression, reference_copy_numbers=None,
            ase_module=True, ase_kwargs=None,
            rdr_module=True, rdr_kwargs=None,
            seed=42
    ):

        self.n_segments = n_segments  # number of blocks
        self.n_cells = n_cells  # number of cells
        self.n_clones = n_clones  # number of clones
        self.cnv_states = tf.constant(cnv_states)
        n_states = self.cnv_states.shape[0]
        self.n_states = n_states  # number of CNV states
        self.xc_module_dict = dict()
        self.seed = seed

        self.epoch = 0
        self.label_permutation = np.arange(0, self.n_clones)

        # Baseline expression levels (currently assumed to be fixed)
        self.baseline_expression = tf.constant(baseline_expression, dtype=tf.float32)
        tf.debugging.assert_non_negative(
            self.baseline_expression,
            "Improper baseline expression: not all the values are non-negative"
        )
        tf.debugging.assert_less_equal(
            self.baseline_expression, 1.,
            "Improper baseline expression: some values  are larger than 1"
        )

        self.alt_counts = alt_counts
        self.total_counts = total_counts
        self.total_counts_all_snps = total_counts_all_snps
        self.reference_cn = reference_copy_numbers

        # Variational distribution variables for cell assignment
        # This is log(pi) in the manuscript notation
        tf.random.set_seed(self.seed)
        self.cell_logit = tf.Variable(
            tf.random.uniform((n_cells, n_clones), -0.1, 0.1,
                              seed=self.seed, dtype=tf.float32)
        )
        self.clonal_labels_prior = None
        
        # Variational distribution variables for CNV states
        # This is log(U) in the manuscript notation.
        tf.random.set_seed(self.seed)
        self.cnv_logit = tf.Variable(
            tf.random.uniform((n_segments, n_clones, n_states), -1, 1,
                              seed=self.seed, dtype=tf.float32)
        )
        self.cnv_states_prior = None

        self.set_prior()
        if ase_module:
            self.init_ase_module()
        if rdr_module:
            self.init_rdr_module(**rdr_kwargs)

    def init_ase_module(self):
        self.xc_module_dict["ase"] = XCloneASE(xclone_base_model=self)

    def init_rdr_module(self, n_mc_samples=100, gp_covariance_fn=None):
        self.xc_module_dict["rdr"] = XCloneRDR(
            xclone_base_model=self,
            n_mc_samples=n_mc_samples,
            gp_covariance_fn=gp_covariance_fn
        )

    def set_prior(self, cnv_states_prior=None, clonal_labels_prior=None):
        """
        Set prior distributions.
        :param cnv_states_prior: tf.Tensor(Nb x Nk x Ns, tf.float6), clonal CNV profile prior (binary)
        :param clonal_labels_prior: tf.Tensor(Nb x Nk, tf.float32), clonal label assignment prior (binary)
        :return: None
        """
        # Prior distributions for CNV state weights
        if cnv_states_prior is None:
            self.cnv_states_prior = tfd.Multinomial(
                total_count=1,
                probs=tf.ones((self.n_segments, self.n_clones, self.n_states), dtype=tf.float32) / self.n_states
            )
        else:
            self.cnv_states_prior = cnv_states_prior
            
        # Prior distributions for cell assignment weights
        if clonal_labels_prior is None:
            self.clonal_labels_prior = tfd.Multinomial(
                total_count=1,
                probs=tf.ones((self.n_cells, self.n_clones), dtype=tf.float32) / self.n_clones
            )
        else:
            self.clonal_labels_prior = clonal_labels_prior

    @property
    def clonal_label_probs(self):
        """Variational posterior for cell assignment"""
        return tfd.Multinomial(total_count=1, logits=self.cell_logit)

    @property
    def inferred_clonal_labels(self):
        return tf.argmax(self.clonal_label_probs.mean(), axis=1)
    
    @property
    def cnv_state_probs(self):
        """Variational posterior for CNV state"""
        return tfd.Multinomial(total_count=1, logits=self.cnv_logit)

    @property
    def inferred_cnv(self):
        return tf.argmax(self.cnv_state_probs.mean(), axis=-1)

    @property
    def inferred_copy_numbers(self):
        return

    @property
    def cnv_states_kl(self):
        return categorical_kl(self.cnv_state_probs, self.cnv_states_prior)

    @property
    def cnv_states_kl_sum(self):
        return tf.reduce_sum(self.cnv_states_kl)

    @property
    def clonal_labels_kl(self):
        return categorical_kl(self.clonal_label_probs, self.clonal_labels_prior)

    @property
    def clonal_labels_kl_sum(self):
        return tf.reduce_sum(self.clonal_labels_kl)

    @property
    def kl_sum(self):
        """Sum of KL divergences between posteriors and priors"""
        total_kl_sum = self.cnv_states_kl_sum + self.clonal_labels_kl_sum
        for xc_module in self.xc_module_dict.values():
            total_kl_sum += xc_module.kl_sum
        return total_kl_sum

    @property
    def log_lik(self):
        total_log_likelihood = 0
        for xc_module in self.xc_module_dict.keys():
            total_log_likelihood += self.xc_module_dict[xc_module].log_lik
        return total_log_likelihood

    @property
    def baf_cnv_discrepancy(self):
        inferred_cnv = self.inferred_cnv
        inferred_bafs = self.xc_module_dict["ase"].bafs.mean().numpy()
        inferred_clonal_labels = self.inferred_clonal_labels
        obvserved_discrepancy = 0
        for clone in range(self.n_clones):
            clonal_cnv = inferred_cnv[:, clone]
            indices = clonal_cnv
            full_indices = tf.stack([tf.range(self.n_segments, dtype=indices.dtype), indices], axis=1)
            inferred_clonal_bafs = tf.gather_nd(inferred_bafs, full_indices)
            clonal_allelic_cn = tf.stack(tf.gather(self.cnv_states, clonal_cnv))
            clonal_total_cn = tf.reduce_sum(clonal_allelic_cn, axis=1)
            expected_clonal_bafs = clonal_allelic_cn[:, 0] / clonal_total_cn
            cells_in_clone = tf.reduce_sum(tf.cast(inferred_clonal_labels == clone, tf.float32))
            squared_discrepancy = (inferred_clonal_bafs - expected_clonal_bafs) ** 2
            obvserved_discrepancy += (cells_in_clone / self.n_cells) * tf.reduce_mean(squared_discrepancy)
        return obvserved_discrepancy
    
    @property
    def loss(self):
        total_loss = 100 * self.kl_sum - self.log_lik
        if "ase" in self.xc_module_dict.keys():
            total_loss += 100000 * self.baf_cnv_discrepancy
        return total_loss

    def most_likely_label_permutation(self, ground_truth_assignment_mx, ord=1):
        """
        XClone can only learn the clonal labels up to a permutation.
        This is known as "identifiability problem", it affects latent variable models in general.
        To learn the permutation, we reduce the problem to the min-weight perfect bipartite matching
        that can be solved in polynomial time, O(n_clones ** 5) at worst.

        To be precise, we take two probability matrices:
        - current clonal label assignment probabilities clonal_label_probs
        - ground truth (usually known from simulations)
        Rows of both matrices correspond to cells, columns — to clones.
        Then, each column becomes a node in the bipartite graph.
        Undirected edges connect columns of one matrix to every column of another.
        Weight of each edge corresponds to the distance between two columns (L_p or any other).

        We claim, that min-weight perfect matching in such a graph yields the most likely permutation of labels.
        The algorithm needed to find the matching is already implemented in networkx library.

        :param ground_truth_assignment_mx: np.ndarray(np.float32), n_cells x n_clones, all values in {0, 1}
            Binary matrix encoding the ground truth label assignment.
            Rows correspond to cells, columns — to clones.
            1 at position (j, k) means, that cell j comes from the clone k.
        :param ord: str or float, borrowed from `ord` parameter of `tf.norm`
            Order of the norm. Supported values are 'fro', 'euclidean', 1, 2, np.inf
            and any positive real number yielding the corresponding p-norm.
        :return: np.array(np.int32),
            A permutation that establishes the connection between learned and true labels
        """
        assert ground_truth_assignment_mx.shape == (self.n_cells, self.n_clones), \
            "Ground truth matrix has wrong shape:" \
            f"\n- {ground_truth_assignment_mx.shape} found" \
            f"\n- {(self.n_cells, self.n_clones)} expected"

        inferred_assignment_mx = self.clonal_label_probs.mean()
        column_graph = nx.Graph()
        left_part = list(range(-self.n_clones, 0))
        right_part = list(range(1, self.n_clones + 1))
        column_graph.add_nodes_from(left_part + right_part)
        column_graph.add_weighted_edges_from([
            (
                left_part[i], right_part[j],
                tf.norm(
                    inferred_assignment_mx[:, i]
                    - ground_truth_assignment_mx[:, j],
                    ord=ord
                )
            )
            for i, j in itertools.product(range(self.n_clones), repeat=2)
        ])
        matching_dict = nx.algorithms.bipartite.minimum_weight_full_matching(
            column_graph,
            top_nodes=left_part
        )
        label_permutation = np.array([
            matching_dict[left_part[i]] - 1
            for i in range(self.n_clones)
        ])
        return label_permutation

    def fit(self, learning_rates, training_iterations, visdom_monitor=None, verbose=False):
        losses = []
        # kl_losses = []
        # ase_neglogliks = []
        # rdr_neglogliks = []
        # aris = []

        for learn_rate, num_steps in zip(learning_rates, training_iterations):
            optimizer = tf.optimizers.Adam(learning_rate=learn_rate)
            for _ in tqdm(range(num_steps), file=sys.stdout if verbose else open(os.devnull, "w")):
                with tf.GradientTape() as tape:
                    loss_value = self.loss
                    grads = tape.gradient(loss_value, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))

                losses.append(self.loss.numpy())
                # kl_losses.append(self.kl_sum.numpy())
                # ase_neglogliks.append(-self._log_lik_ase().numpy())
                # rdr_neglogliks.append(-self._log_lik_rdr().numpy())

                # _ID_prob = model.clonal_label_probs.mean().numpy()
                # ARI = adjusted_rand_score(np.argmax(_ID_prob, axis=1), I_DNA)
                # aris.append(ARI)
                if self.epoch % 50 == 0:
                    if verbose:
                        print(f"Total loss: {self.loss}")
                        print(f"Total KL: {self.kl_sum}.")
                        if "ase" in self.xc_module_dict.keys():
                            print(f"Observed BAF-CNV discrepancy: {100000 * self.baf_cnv_discrepancy}")
                        print(f"\t- CNV state assignment KL: {self.cnv_states_kl_sum}")
                        print(f"\t- Clonal label assignment KL: {self.clonal_labels_kl_sum}")
                        for module_name in self.xc_module_dict.keys():
                            print(f"\t- {module_name.upper()} KL: {self.xc_module_dict[module_name].kl_sum}")
                        print(f"Total log-likelihood: {self.log_lik}")
                        for module_name in self.xc_module_dict.keys():
                            print(f"\t- {module_name.upper()} log-likelihood: {self.xc_module_dict[module_name].log_lik}")

                    if visdom_monitor is not None:
                        visdom_monitor.update_figures()
                        # visdom_monitor.inferred_total_cn_heatmap(cell_order=cell_order)
                        # visdom_monitor.expected_loh_heatmap(cell_order=cell_order)
                        # visdom_monitor.bafs_expected_from_cnv_heatmap(cell_order=cell_order)
                        # visdom_monitor.cell_projection_scatterplot(self.alt_counts, self.total_counts)
                        # convergence_monitor.ari_line_plot(aris)
                        # visdom_monitor.clone_probability_heatmap()
                        # if "ase" in self.xc_module_dict.keys():
                            # visdom_monitor.inferred_bafs_vs_cnv_boxplot()
                            # visdom_monitor.inferred_bafs_heatmap(cell_order=cell_order)
                        # convergence_monitor.inferred_allelic_cnv_heatmap()
                        # visdom_monitor.inferred_amplification_rates_scatterplot()
                        # convergence_monitor.F_heatmap()
                        # convergence_monitor.loss_history_lineplot(losses)
                        # visdom_monitor.visdom_obj.line(
                        #     cnv_state_probs=kl_losses,
                        #     X=np.arange(len(kl_losses)),
                        #     opts=dict(
                        #         title="KL divergence term",
                        #         xlabel="iterations",
                        #         ylabel="value",
                        #         webgl=True
                        #     ),
                        #     win="loss"
                        # )
                        # visdom_monitor.visdom_obj.line(
                        #     cnv_state_probs=ase_neglogliks,
                        #     X=np.arange(len(ase_neglogliks)),
                        #     opts=dict(
                        #         title="ASE negative log-likelihood",
                        #         xlabel="iterations",
                        #         ylabel="value",
                        #         webgl=True
                        #     ),
                        #     win="ase_negloglik"
                        # )
                        # convergence_monitor.visdom_obj.line(
                        #     cnv_state_probs=rdr_neglogliks,
                        #     X=np.arange(len(rdr_neglogliks)),
                        #     opts=dict(
                        #         title="RDR negative log-likelihood",
                        #         xlabel="iterations",
                        #         ylabel="value",
                        #         webgl=True
                        #     ),
                        #     win="rdr_negloglik"
                        # )
                self.epoch += 1