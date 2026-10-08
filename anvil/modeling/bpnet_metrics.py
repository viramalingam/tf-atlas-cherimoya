"""Metric functions of `bpnet-predict`, copied VERBATIM so cherimoya predictions
are scored exactly like the released BPNet models.

Source: github.com/kundajelab/bpnet-refactor @ 0b11a7c5d12f19e1e99213df166271f7dd71586f
  - bpnet/cli/predict.py lines 36-376 (mnll ... metrics_update)
  - bpnet/utils/bigwig_helper.py (write_bigwig)
bpnet-refactor is the source of the bpnet 0.4.0 package that trained and scored
the released tf-atlas models (see tfatlas/eval/notes/00_benchmark_design.md).
Only the imports below were added; the function bodies are unchanged.
"""

import math

import numpy as np
import pyBigWig
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr, multinomial
from tqdm import tqdm


def mnll(true_counts, logits=None, probs=None):
    """
        Compute the multinomial negative log-likelihood between true
        counts and predicted values of a BPNet-like profile model
        
        One of `logits` or `probs` must be given. If both are
        given `logits` takes preference.

        Args:
            true_counts (numpy.array): observed counts values
            
            logits (numpy.array): predicted logits values
            
            probs (numpy.array): predicted values as probabilities
          
        Returns:
            float: cross entropy
    
    """

    dist = None 
    
    if logits is not None:
        
        # check for length mismatch
        if len(logits) != len(true_counts):
            raise NoTracebackException(
                "Length of logits does not match length of true_counts")
        
        # convert logits to softmax probabilities
        probs = logits - logsumexp(logits)
        probs = np.exp(probs)
        
    elif probs is not None:      
        
        # check for length mistmatch
        if len(probs) != len(true_counts):
            raise NoTracebackException(
                "Length of probs does not match length of true_counts")
        
        # check if probs sums to 1
        if abs(1.0 - np.sum(probs)) > 1e-3:
            raise NoTracebackException(
                "'probs' array does not sum to 1")   
           
    else:
        
        # both 'probs' and 'logits' are None
        raise NoTracebackException(
            "At least one of probs or logits must be provided. "
            "Both are None.")
  
    # compute the nmultinomial distribution
    mnom = multinomial(np.sum(true_counts), probs)
    return -(mnom.logpmf(true_counts) / len(true_counts))
    

def profile_cross_entropy(true_counts, logits=None, probs=None):
    """
        Compute the cross entropy between true counts and predicted 
        values of a BPNet-like profile model
        
        One of `logits` or `probs` must be given. If both are
        given `logits` takes preference.

        Args:
            true_counts (numpy.array): observed counts values
            
            logits (numpy.array): predicted logits values
            
            probs (numpy.array): predicted values as probabilities
          
        Returns:
            float: cross entropy
    
    """

    if logits is not None:
        
        # check for length mismatch
        if len(logits) != len(true_counts):
            raise NoTracebackException(
                "Length of logits does not match length of true_counts")
        
        # convert logits to softmax probabilities
        probs = logits - logsumexp(logits)
        probs = np.exp(probs)
        
    elif probs is not None:      
        
        # check for length mistmatch
        if len(probs) != len(true_counts):
            raise NoTracebackException(
                "Length of probs does not match length of true_counts")
        
        # check if probs sums to 1
        if abs(1.0 - np.sum(probs)) > 1e-3:
            raise NoTracebackException(
                "'probs' array does not sum to 1")        
    else:
        
        # both 'probs' and 'logits' are None
        raise NoTracebackException(
            "At least one of probs or logits must be provided. "
            "Both are None.")
        
    # convert true_counts to probabilities
    true_counts_prob = true_counts / np.sum(true_counts)
    
    return -np.sum(np.multiply(true_counts_prob, np.log(probs + 1e-7)))


def _fix_sum_to_one(probs):
    """
      Fix probability arrays whose sum is fractinally above or 
      below 1.0
      
      Args:
          probs (numpy.ndarray): An array whose sum is almost equal
              to 1.0
              
      Returns:
          np.ndarray: array that sums to 1
    """
    
    _probs = np.copy(probs)
    
    if np.sum(_probs) > 1.0:        
        _probs[np.argmax(_probs)] -= np.sum(_probs) - 1.0    
    
    if np.sum(_probs) < 1.0:
        _probs[np.argmin(_probs)] += 1.0 - np.sum(_probs)
    
    return _probs
    
def mnll_min_max_bounds(profile):
    """
        Min Max bounds for the mnll metric
        
        Args:
            profile (numpy.ndarray): true profile 
        Returns:
            tuple: (min, max) bounds values
    """
    
    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile = profile.astype(np.float64)
    
    # profile as probabilities
    profile_prob = profile / np.sum(profile)
    
    # the scipy.stats.multinomial function is very sensitive to 
    # profile_prob summing to exactly 1.0, if not you get NaN as the
    # resuls. In majority of the cases we can fix that problem by
    # adding or substracting the difference (but unfortunately it
    # doesnt always and there are cases where we still see NaNs, and
    # those we'll set to 0)
    profile_prob = _fix_sum_to_one(profile_prob)

    # mnll of profile with itself
    min_mnll = mnll(profile, probs=profile_prob)
    
    # if we still find a NaN, even after the above fix, set it to zero
    if math.isnan(min_mnll):
        min_mnll = 0.0

    if math.isinf(min_mnll):
        min_mnll = 0.0

    # mnll of profile with uniform profile
    max_mnll = mnll(profile, probs=uniform_profile)
    
    return (min_mnll, max_mnll)


def cross_entropy_min_max_bounds(profile):
    """
        Min Max bounds for the cross entropy metric
        
        Args:
            profile (numpy.ndarray): true profile 
            
        Returns:
            tuple: (min, max) bounds values
    """

    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # mnll of profile with itself
    min_cross_entropy = profile_cross_entropy(profile, probs=profile_prob)
    
    # mnll of profile with uniform profile
    max_cross_entropy = profile_cross_entropy(profile, probs=uniform_profile)

    return (min_cross_entropy, max_cross_entropy)


def jsd_min_max_bounds(profile):
    """
        Min Max bounds for the jsd metric
        
        Args:
            profile (numpy.ndarray): true profile 
            
        Returns:
            tuple: (min, max) bounds values
    """
    
    # uniform distribution profile
    uniform_profile = np.ones(len(profile)) * (1.0 / len(profile))

    # profile as probabilities
    profile_prob = profile / np.sum(profile)

    # jsd of profile with uniform profile
    max_jsd = jensenshannon(profile_prob, uniform_profile)

    # jsd of profile with itself (upper bound)
    min_jsd = 0.0

    return (min_jsd, max_jsd)

def min_max_normalize(metric_value, best_worst_bounds, default=0.0):
    """
        Best-case worst-case normalize a metric value
        
        Args:
            metric_value (float): unnormalized metric value
            best_worst_bounds (tuple): (min, max) bounds for jsd, mnll, cross-entropy
            default (float): the default value to return in case
                of runtime exceptions
    """
    
    best_bound = best_worst_bounds[0]
    worst_bound = best_worst_bounds[1]
    
    if (best_bound - worst_bound) == 0:
        return default
    
    norm_value = (metric_value - worst_bound) / (best_bound - worst_bound) 
    
    if norm_value < 0:
        return 0
    
    if norm_value > 1:
        return 1
    
    return norm_value


def metrics_update(
    metrics_tracker, example_idx, track_idx, true_profile, true_logcounts,
    pred_profile, pred_logcounts, true_profile_smoothing=[7.0, 81]):
    """
        Update metrics with new true/predicted profile & counts
        
        Args:
            metrics_tracker (dict): dictionary to track & update 
                metrics info 
            example_idx (int): index of the example to be updated
            track_idx (int): index of the ith track to be updated
            true_profile (numpy.ndarray): ground truth profile
            true_counts (float): sum of true profile
            pred_profile (numpy.ndarray): predicted profile output
            pred_counts (float): predicted 
            true_profile_smoothing (list): list of 2 values, sigma &
                window size for gaussuan 1d smoothing
            
    
    """
    
    # Step 1 - smooth true profile
    sigma = true_profile_smoothing[0]
    width = float(true_profile_smoothing[1])
    truncate = (((width - 1)/2)-0.5)/sigma

    # Step 2 - smooth true profile and convert profiles to 
    # probabilities
    if np.sum(true_profile) != 0 and np.sum(pred_profile) != 0:
        # smoothing
        true_profile_smooth = gaussian_filter1d(
            true_profile, sigma=sigma, truncate=truncate)

        # convert to probabilities
        true_profile_smooth_prob = true_profile_smooth / np.sum(
            true_profile_smooth)
        pred_profile_prob = pred_profile / np.sum(pred_profile)
        pred_profile_prob = _fix_sum_to_one(pred_profile_prob)

        # metrics 
        # profile pearson & spearman
        # with pearson we need to check if either of the arrays
        # has zero standard deviation (i.e having all same elements,
        # a zero or any other value). Unfortunately np.std
        # returns a very small non-zero value, so we'll use a 
        # different approach to check if the array has the same value.
        # If true then pearson correlation is undefined 
        if np.unique(true_profile_smooth_prob).size == 1 or \
            np.unique(pred_profile_prob).size == 1:
            metrics_tracker['profile_pearsonrs'][example_idx, track_idx] = 0
            metrics_tracker['profile_spearmanrs'][example_idx, track_idx] = 0
        else:
            metrics_tracker['profile_pearsonrs'][example_idx, track_idx] = \
                pearsonr(true_profile_smooth_prob, pred_profile_prob)[0]

            metrics_tracker['profile_spearmanrs'][example_idx, track_idx] = \
                spearmanr(true_profile_smooth_prob, pred_profile_prob)[0]
            
        # mnll
        _mnll = np.nan_to_num(mnll(true_profile, probs=pred_profile_prob))
        _mnll = min_max_normalize(_mnll, mnll_min_max_bounds(true_profile))
        metrics_tracker['profile_mnlls'][example_idx, track_idx] = _mnll

        # cross entropy
        metrics_tracker['profile_cross_entropys'][example_idx, track_idx] = \
            min_max_normalize(
                profile_cross_entropy(true_profile, probs=pred_profile_prob), 
                cross_entropy_min_max_bounds(true_profile))

        # jsd
        metrics_tracker['profile_jsds'][example_idx, track_idx] = \
            min_max_normalize(
                jensenshannon(true_profile_smooth_prob, pred_profile_prob), 
                jsd_min_max_bounds(true_profile))

        # mse
        metrics_tracker['profile_mses'][example_idx, track_idx] = \
            np.square(np.subtract(true_profile, pred_profile)).mean()
        
    metrics_tracker['all_true_logcounts'][example_idx, track_idx] = \
        true_logcounts
    metrics_tracker['all_pred_logcounts'][example_idx, track_idx] = \
        pred_logcounts



def write_bigwig(data, regions, header, bw_out, outstats_file):
    # regions may overlap but as we go in sorted order, at a given position,
    # we will pick the value from the interval whose summit is closest to 
    # current position
    
    chr_to_idx = {}
    for i,x in enumerate(header):
        chr_to_idx[x[0]] = i

    bw = pyBigWig.open(bw_out, 'w')
    bw.addHeader(header)
    
    # regions may not be sorted, so get their sorted order
    order_of_regs = sorted(range(len(regions)), key=lambda x:(chr_to_idx[regions[x][0]], regions[x][1]))

    all_entries = []
    cur_chr = ""
    cur_end = 0

    iterator = range(len(order_of_regs))

    for itr in tqdm(iterator):

        i = order_of_regs[itr]
        i_chr, i_start, i_end, i_mid = regions[i]
    
        if i_chr != cur_chr: 
            cur_chr = i_chr
            cur_end = 0
    
        # bring current end to at least start of current region
        if cur_end < i_start:
            cur_end = i_start
    
        assert(regions[i][2]>=cur_end)
    
        # figure out where to stop for this region, get next region
        # which may partially overlap with this one
        next_end = i_end
    
        if itr+1 != len(order_of_regs):
            n = order_of_regs[itr+1]
            next_chr, next_start, _, next_mid = regions[n]
       
            if next_chr == i_chr and next_start < i_end:
                # if next region overlaps with this, end between their midpoints
                next_end = (i_mid+next_mid)//2
       
        vals = data[i][cur_end - i_start:next_end - i_start]

        bw.addEntries([i_chr]*(next_end-cur_end), 
                       list(range(cur_end,next_end)), 
                       ends = list(range(cur_end+1, next_end+1)), 
                       values=[float(x) for x in vals])
    
        all_entries.append(vals)
        
        cur_end = next_end

    bw.close()

    all_entries = np.hstack(all_entries)

    with open(outstats_file, 'w') as f:
        f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
        f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
        f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
        f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
        f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
        f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
        f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
        f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
