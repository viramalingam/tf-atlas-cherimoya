#!/bin/bash
# Write the splits json for peak-wise fold $1 to $2, in the layout of the
# tf-atlas peak_wise_split_fold<k>.json templates (indices under /project/splits_indices).
fold=$1
d=/project/splits_indices
printf '{"0": {"loci_val_indices_file": "%s/loci_valid_indices_fold%s.txt", "loci_train_indices_file": "%s/loci_train_indices_fold%s.txt", "background_val_indices_file": "%s/background_valid_indices_fold%s.txt", "background_train_indices_file": "%s/background_train_indices_fold%s.txt", "loci_test_indices_file": "%s/loci_test_indices_fold%s.txt", "background_test_indices_file": "%s/background_test_indices_fold%s.txt"}}\n' \
    $d $fold $d $fold $d $fold $d $fold $d $fold $d $fold > $2
