from itertools import count
import json
import argparse
import os
import random
import sys
import pandas as pd
import numpy as np
import pyBigWig
from plotnine import *

# Copied from viramalingam/tf-atlas-pipeline anvil/modeling/peak_wise_splits.py
# (main @ 124f049). Changes, all opt-in so the default behaviour is the old one:
#   --test-chroms / --test-frac / --seed: carve ONE common held-out test set T
#     from the given chromosomes (whole blocks, stratified by block counts) and
#     use it as the "test" split of every fold. The other blocks, on all
#     chromosomes including the rest of the test chromosomes, are chunked
#     exactly as before; fold f uses chunk f as valid and every other chunk as
#     train (the old code used chunk f+1 as test).
#   chained assignments `df[col][idx] = v` became `df.loc[idx, col] = v`
#     (same result; the chained form silently stops working in newer pandas).
#   leakage gates + manifest.json at the end.

NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]
N_STRATA = 10


def carve_test_groups(group_df, test_chroms, test_frac, n_peaks_total, n_nonpeaks_total, seed):
    """Pick whole groups on test_chroms: peak-containing groups stratified by
    group counts until T holds ~test_frac of ALL peaks, then nonpeak-only
    groups until T holds ~test_frac of ALL nonpeaks."""
    on_test = group_df[group_df['chr'].isin(test_chroms)]
    cand = on_test[on_test['n_peaks'] > 0].sort_values(by='group_counts').reset_index(drop=True)
    test_chrom_peaks = int(cand['n_peaks'].sum())
    if test_chrom_peaks == 0:
        raise SystemExit("no peaks on test chromosomes {}".format(test_chroms))
    within_frac = min(1.0, test_frac * n_peaks_total / test_chrom_peaks)

    # peak-count-weighted strata over the counts-sorted candidate groups
    cum = cand['n_peaks'].cumsum()
    cand['stratum'] = np.minimum((cum - 1) * N_STRATA // test_chrom_peaks, N_STRATA - 1)

    rng = np.random.RandomState(seed)
    test_groups = []
    for _, stratum in cand.groupby('stratum'):
        target = within_frac * stratum['n_peaks'].sum()
        got = 0
        for row in stratum.sample(frac=1.0, random_state=rng).itertuples():
            if got >= target:
                break
            test_groups.append(row.groups)
            got += row.n_peaks

    # nonpeak-only groups to match the overall peak:nonpeak ratio in T
    got_neg = int(on_test[on_test['groups'].isin(test_groups)]['n_nonpeaks'].sum())
    target_neg = test_frac * n_nonpeaks_total
    neg_only = on_test[(on_test['n_peaks'] == 0)].sample(frac=1.0, random_state=rng)
    for row in neg_only.itertuples():
        if got_neg >= target_neg:
            break
        test_groups.append(row.groups)
        got_neg += row.n_nonpeaks
    return set(test_groups), within_frac


def adjacent_gate(df, buffer, label_col, name):
    """No two adjacent (sorted) regions within buffer may carry different labels.
    Adjacency suffices: a label change inside any within-buffer pair implies one
    at some adjacent within-buffer pair."""
    d = df.sort_values(by=['chr', 'pos'], kind='mergesort')
    same_chrom = d['chr'].eq(d['chr'].shift(-1))
    close = (d['pos'].shift(-1) - d['pos']) <= buffer
    differ = d[label_col].ne(d[label_col].shift(-1))
    bad = same_chrom & close & differ
    for _, r in d[bad].head(5).iterrows():
        print("  LEAK[{}] {}@{} ({})".format(name, r['chr'], r['pos'], r[label_col]))
    return int(bad.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bigwig", type=str, required=True, help="Bigwig of tn5 insertions")
    parser.add_argument("--peaks", type=str, required=True, help="Peaks")
    parser.add_argument("--nonpeaks", type=str, required=True, help="Non-Peaks")
    parser.add_argument("--inputlen", type=int, default=2114, help="Sequence input length")
    parser.add_argument("-j", "--max_jitter", type=int, default=128, help="Maximum jitter applied on either side of region")
    parser.add_argument("--number-of-folds", type=int, default=5, help="number of folds 5, 10 etc")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to store the fold information")
    parser.add_argument("--supplemental_output_path",  type=str, required=True, help="Path to store the upplemental_outputs such as the counts histgram pngs and the split beds")
    parser.add_argument("--test-chroms", nargs='+', default=None, help="carve a common held-out test set T from these chromosomes (e.g. chrX); 'None' = old behaviour")
    parser.add_argument("--test-frac", type=float, default=0.05, help="size of T as a fraction of ALL peaks (and of ALL nonpeaks)")
    parser.add_argument("--seed", type=int, default=1234, help="seed for the T block sampling")
    args = parser.parse_args()

    if args.test_chroms == ['None']:
        args.test_chroms = None


    peak_regions_df = pd.read_csv(args.peaks, sep='\t', names=NARROWPEAK_SCHEMA)
    peak_regions_df['region']='peak'
    peak_regions_df['ind']=range(len(peak_regions_df))
    nonpeak_regions_df = pd.read_csv(args.nonpeaks, sep='\t', names=NARROWPEAK_SCHEMA)
    nonpeak_regions_df['region']='nonpeak'
    nonpeak_regions_df['ind']=range(len(nonpeak_regions_df))


    all_regions_df = pd.concat([peak_regions_df,nonpeak_regions_df])
    all_regions_df['pos']=all_regions_df['start']+all_regions_df['summit']
    all_regions_df.sort_values(by=['chr', 'pos'], inplace=True)
    all_regions_df=all_regions_df.reset_index(drop=True)

    print("Creating Splits")

    group_dict = {}


    cur_chrom = ''
    cur_group = ''
    last_pos = 0
    for index,row in all_regions_df.iterrows():
        if cur_chrom != '':
            if row['chr'] != cur_chrom:
                cur_chrom = row['chr']
                cur_group += 1
                group_dict[cur_group] = [row]
            else:
                if row['pos'] <= int(last_pos) + int(args.inputlen) + int(2 * args.max_jitter):
                    group_dict[cur_group].append(row)
                else:
                    cur_group += 1
                    group_dict[cur_group] = [row]
        else:
            cur_chrom = row['chr']
            cur_group = 0
            group_dict[cur_group] = [row]
        last_pos = row['pos']

    groups = []
    group_counts = []
    bw = pyBigWig.open(args.bigwig)

    for group in group_dict:
        groups.append(group)
        sum = 0
        for element in group_dict[group]:
            try:
                labels = bw.values(element['chr'], int(element['pos']) - (args.inputlen // 2), int(element['pos']) + (args.inputlen // 2))
            except:
                labels = np.zeros((args.inputlen,))
                print("warning: unable to fetch values for ",element['chr'], int(element['pos']) - (args.inputlen // 2), int(element['pos']) + (args.inputlen // 2))
            labels = np.array(labels)
            labels = np.nan_to_num(labels)
            labels = np.sum(labels)
            sum += labels
        group_counts.append(sum)
    group_df = pd.DataFrame({'groups': groups, 'group_counts': group_counts})
    group_df = group_df.sort_values(by='group_counts').reset_index(drop=True)

    test_group_df = None
    if args.test_chroms is not None:
        group_df['chr'] = [group_dict[g][0]['chr'] for g in group_df['groups']]
        group_df['n_peaks'] = [int(np.sum([e['region'] == 'peak' for e in group_dict[g]])) for g in group_df['groups']]
        group_df['n_nonpeaks'] = [len(group_dict[g]) - n for g, n in zip(group_df['groups'], group_df['n_peaks'])]
        test_groups, within_frac = carve_test_groups(
            group_df, args.test_chroms, args.test_frac,
            len(peak_regions_df), len(nonpeak_regions_df), args.seed)
        test_group_df = group_df[group_df['groups'].isin(test_groups)].reset_index(drop=True)
        group_df = group_df[~group_df['groups'].isin(test_groups)].reset_index(drop=True)
        print("test set T: {} groups, {} peaks, {} nonpeaks on {} ({:.1%} of the test-chrom peak groups sampled)".format(
            len(test_group_df), int(test_group_df['n_peaks'].sum()),
            int(test_group_df['n_nonpeaks'].sum()), args.test_chroms, within_frac))


        # split the data into x number of chuncks for x folds. allocate the chuncks to train, test, valid folds in unique ways
    # across folds
    chuncksets_dict={}
    for fold in range(args.number_of_folds):
        chuncksets_dict[f"chuncksets_{fold}"]=list(range(fold,len(group_df),args.number_of_folds))

    val_chuncks = list(range(0,args.number_of_folds))
    print("val_chuncks:",val_chuncks)

    test_chuncks = list(range(1,args.number_of_folds))+[0]
    print("test_chuncks:",test_chuncks if args.test_chroms is None else "common test set T")

    group_fold_df=pd.DataFrame(index=np.arange(len(group_df)))

    for fold in range(args.number_of_folds):
        group_fold_df[f"fold{fold}"]='train'
        group_fold_df.loc[chuncksets_dict[f"chuncksets_{val_chuncks[fold]}"], f"fold{fold}"] = 'valid'
        if args.test_chroms is None:
            group_fold_df.loc[chuncksets_dict[f"chuncksets_{test_chuncks[fold]}"], f"fold{fold}"] = 'test'

    print(group_fold_df)

    for fold in range(args.number_of_folds):
        print("fold:",fold)
        len(group_fold_df[group_fold_df[f"fold{fold}"]=="train"])/len(group_fold_df)

        len(group_fold_df[group_fold_df[f"fold{fold}"]=="test"])/len(group_fold_df)

        len(group_fold_df[group_fold_df[f"fold{fold}"]=="valid"])/len(group_fold_df)


    for fold in range(args.number_of_folds):
        group_df['fold' + str(fold)] = group_fold_df['fold' + str(fold)]

    group_df
    output_path ="."
    print("Saving Splits")
    split_of_region = {}
    per_fold = []
    for fold in range(args.number_of_folds):
        print("fold:",fold)
        per_fold.append({'fold': fold})
        for split in ['valid','train','test']:
            if split == 'test' and test_group_df is not None:
                split_groups = test_group_df['groups']
            else:
                split_groups = group_df['groups'][group_df[f"fold{fold}"]==split]
            temp_lst = [group_dict.get(key) for key in split_groups]
            peak_indices = [i['ind'] for b in map(lambda x:[x] if not isinstance(x, list) else x, temp_lst) for i in b if i['region']=='peak']
            nonpeak_indices = [i['ind'] for b in map(lambda x:[x] if not isinstance(x, list) else x, temp_lst) for i in b if i['region']=='nonpeak']
            print("split:",split)
            print("proportion of peaks:",len(peak_indices)/len(peak_regions_df))
            print("proportion of nonpeaks:",len(nonpeak_indices)/len(nonpeak_regions_df))

            f = open(f"{args.output_path}/loci_{split}_indices_fold{fold}.txt", "w")
            for items in peak_indices:
                f.writelines(str(items)+'\n')
            f.close()
            f = open(f"{args.output_path}/background_{split}_indices_fold{fold}.txt", "w")
            for items in nonpeak_indices:
                f.writelines(str(items)+'\n')
            f.close()
            nonpeak_regions_df.iloc[nonpeak_indices,0:10].to_csv(f"{args.supplemental_output_path}/background_peaks_{split}_fold{fold}.bed",sep="\t",header=False,index=False)
            peak_regions_df.iloc[peak_indices,0:10].to_csv(f"{args.supplemental_output_path}/peaks_{split}_fold{fold}.bed",sep="\t",header=False,index=False)
            per_fold[-1][f"{split}_peaks"] = len(peak_indices)
            per_fold[-1][f"{split}_nonpeaks"] = len(nonpeak_indices)
            for i in peak_indices:
                split_of_region[(fold, 'peak', i)] = split
            for i in nonpeak_indices:
                split_of_region[(fold, 'nonpeak', i)] = split
        print("\n")

    if test_group_df is not None:
        for fold in range(args.number_of_folds):
            test_group_df['fold' + str(fold)] = 'test'
        group_df = pd.concat([group_df, test_group_df]).sort_values(by='group_counts').reset_index(drop=True)

    group_df["log_groupcounts"]=np.log10(group_df["group_counts"]+1)
    for fold in range(args.number_of_folds):
        print("fold:",fold)
        plot = (ggplot(group_df,aes("log_groupcounts",color=f"fold{fold}"))
                    +stat_ecdf()
                    +theme_classic()
           )
        plot.save(f'{args.supplemental_output_path}/fold{fold}_counts_histogram_plot.png')

    group_df.to_csv(f"{args.supplemental_output_path}/group_df.csv",index=False)

    # ---- leakage gates (exit nonzero on any violation) ----
    buffer = int(args.inputlen) + int(2 * args.max_jitter)
    total_leak = 0
    n_peaks_ok = True
    for fold in range(args.number_of_folds):
        lab = [split_of_region.get((fold, r, i), 'unassigned')
               for r, i in zip(all_regions_df['region'], all_regions_df['ind'])]
        d = all_regions_df[['chr', 'pos']].copy()
        d['split'] = lab
        if (d['split'] == 'unassigned').any():
            print("  UNASSIGNED regions in fold", fold, int((d['split'] == 'unassigned').sum()))
            n_peaks_ok = False
        total_leak += adjacent_gate(d, buffer, 'split', "fold{}".format(fold))
    t_peaks = peak_regions_df.iloc[[i for (f, r, i), s in split_of_region.items()
                                    if f == 0 and r == 'peak' and s == 'test']]
    t_on_test_chroms = True
    if test_group_df is not None:
        t_on_test_chroms = bool(t_peaks['chr'].isin(args.test_chroms).all())
    gate = "PASS" if (total_leak == 0 and n_peaks_ok and t_on_test_chroms) else "FAIL"
    print("LEAKAGE GATE:", gate, "(cross-split pairs within {} bp: {}; all regions assigned: {}; T on test chroms: {})".format(
        buffer, total_leak, n_peaks_ok, t_on_test_chroms))

    manifest = {
        'peaks': args.peaks, 'nonpeaks': args.nonpeaks, 'bigwig': args.bigwig,
        'inputlen': args.inputlen, 'max_jitter': args.max_jitter, 'buffer': buffer,
        'number_of_folds': args.number_of_folds,
        'test_chroms': args.test_chroms, 'test_frac': args.test_frac, 'seed': args.seed,
        'n_peaks': len(peak_regions_df), 'n_nonpeaks': len(nonpeak_regions_df),
        'n_groups': len(group_dict),
        'n_test_peaks': int(len(t_peaks)),
        'test_peaks_per_chrom': {str(k): int(v) for k, v in t_peaks['chr'].value_counts().items()},
        'per_fold': per_fold,
        'leakage_gate': gate,
    }
    with open(f"{args.supplemental_output_path}/manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(pd.DataFrame(per_fold).to_string(index=False))
    if gate != "PASS":
        sys.exit(1)


if __name__=="__main__":
    main()
