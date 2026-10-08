#!/bin/bash

# Cherimoya version of tf-atlas-pipeline anvil/modeling/modelling_pipeline.sh
# (v2.1.0-rc.5). Same positional arguments, /project layout and output folders;
# bpnet-train / bpnet-predict are replaced by cherimoya_train.py /
# cherimoya_predict.py (same flags, same output files). Differences:
#   - argument 4 is a cherimoya params json (no counts_loss_weight step:
#     cherimoya learns its loss weights)
#   - argument 17 = seed, argument 18 = predict_all_chroms (true|false); the
#     two *_all_chroms prediction sets are only written when true (peak-wise
#     teachers skip them)
#   - the model is <experiment>_split000.torch (no SavedModel tar)
#   - only the current fold's indices files are copied into the model dir

function timestamp {
    # Function to get the current time with the new line character
    # removed

    # current time
    date +"%Y-%m-%d_%H-%M-%S" | tr -d '\n'
}

experiment=$1
training_input_json=$2
testing_input_json=$3
cherimoya_params_json=$4
splits_json=$5
reference_file=$6
reference_file_index=$7
chrom_sizes=$8
chroms_txt=$9
bigwigs=${10}
peaks=${11}
background_regions=${12}
learning_rate=${13}
input_seq_len=${14}
output_len=${15}
indices_files=${16}
seed=${17:-1234}
predict_all_chroms=${18:-true}

scripts_dir=$(dirname $(readlink -f $0))

mkdir /project
project_dir=/project

# create the log file
logfile=$project_dir/${1}_modeling.log
touch $logfile

# create the data directory
data_dir=$project_dir/data
echo $( timestamp ): "mkdir" $data_dir | tee -a $logfile
mkdir $data_dir

# create the reference directory
reference_dir=$project_dir/reference
echo $( timestamp ): "mkdir" $reference_dir | tee -a $logfile
mkdir $reference_dir

# create the model directory
model_dir=$project_dir/model
echo $( timestamp ): "mkdir" $model_dir | tee -a $logfile
mkdir $model_dir

# create indices directory
indices_dir=$project_dir/splits_indices
echo $( timestamp ): "mkdir" $indices_dir | tee -a $logfile
mkdir $indices_dir

# create the predictions directories (same names as modelling_pipeline.sh)
predictions_dir_all_peaks_all_chroms=$project_dir/predictions_and_metrics_all_peaks_all_chroms
predictions_dir_all_peaks_test_chroms=$project_dir/predictions_and_metrics_all_peaks_test_chroms
predictions_dir_test_peaks_test_chroms=$project_dir/predictions_and_metrics_test_peaks_test_chroms
predictions_dir_test_peaks_all_chroms=$project_dir/predictions_and_metrics_test_peaks_all_chroms
predictions_dir_all_peaks_test_chroms_wo_bias=$project_dir/predictions_and_metrics_all_peaks_test_chroms_wo_bias
predictions_dir_test_peaks_test_chroms_wo_bias=$project_dir/predictions_and_metrics_test_peaks_test_chroms_wo_bias
for d in $predictions_dir_all_peaks_all_chroms $predictions_dir_all_peaks_test_chroms \
         $predictions_dir_test_peaks_test_chroms $predictions_dir_test_peaks_all_chroms \
         $predictions_dir_all_peaks_test_chroms_wo_bias $predictions_dir_test_peaks_test_chroms_wo_bias; do
    echo $( timestamp ): "mkdir" $d | tee -a $logfile
    mkdir $d
done


# copy down data and reference (same file names as the bpnet pipeline)

cp $reference_file $reference_dir/hg38.genome.fa
cp $reference_file_index $reference_dir/hg38.genome.fa.fai
cp $chrom_sizes $reference_dir/chrom.sizes
cp $chroms_txt $reference_dir/hg38_chroms.txt
echo $( timestamp ): "cp reference files to" $reference_dir | tee -a $logfile


# Step 1: Copy the bigwig and peak files

echo $bigwigs | sed 's/,/ /g' | xargs cp -t $data_dir/

echo $( timestamp ): "cp" $bigwigs ${data_dir}/ |\
tee -a $logfile

cp $peaks ${data_dir}/${experiment}_peaks.bed.gz
gunzip ${data_dir}/${experiment}_peaks.bed.gz

cp $background_regions ${data_dir}/${experiment}_background_regions.bed.gz
gunzip ${data_dir}/${experiment}_background_regions.bed.gz

echo $( timestamp ): "cat" ${data_dir}/${experiment}_peaks.bed ${data_dir}/${experiment}_background_regions.bed ">" ${data_dir}/${experiment}_combined.bed |\
tee -a $logfile

cat ${data_dir}/${experiment}_peaks.bed ${data_dir}/${experiment}_background_regions.bed > ${data_dir}/${experiment}_combined.bed


# cp input json templates

cp $training_input_json $project_dir/training_input.json
sed -i -e "s/<experiment>/$1/g" $project_dir/training_input.json

cp $testing_input_json $project_dir/testing_input.json

cp $cherimoya_params_json $project_dir/cherimoya_params.json

cp $splits_json $project_dir/splits.json

echo $indices_files
# cp train val test indices files
if [[ -n "${indices_files}" ]];then
    echo "indices variable set"
    if [[ $indices_files != '' ]];then
        echo $indices_files | sed 's/,/ /g' | xargs cp -t $indices_dir/
        echo $( timestamp ): "cp" $indices_files ${indices_dir}/ |\
        tee -a $logfile
    fi
fi

echo "/project/data/"
ls /project/data/
echo '$project_dir/training_input.json'
cat $project_dir/training_input.json
cat $project_dir/splits.json
cat $project_dir/cherimoya_params.json

nvidia-smi | tee -a $logfile

echo $( timestamp ): "
python $scripts_dir/cherimoya_train.py \\
    --input-data $project_dir/training_input.json \\
    --output-dir $model_dir \\
    --reference-genome $reference_dir/hg38.genome.fa \\
    --chrom-sizes $reference_dir/chrom.sizes \\
    --chroms $(paste -s -d ' ' $reference_dir/hg38_chroms.txt)  \\
    --epochs 100 \\
    --splits $project_dir/splits.json \\
    --params $project_dir/cherimoya_params.json \\
    --model-output-filename $1 \\
    --input-seq-len ${input_seq_len} \\
    --output-len ${output_len} \\
    --batch-size 64 \\
    --learning-rate $learning_rate \\
    --seed $seed" | tee -a $logfile

python $scripts_dir/cherimoya_train.py \
    --input-data $project_dir/training_input.json \
    --output-dir $model_dir \
    --reference-genome $reference_dir/hg38.genome.fa \
    --chrom-sizes $reference_dir/chrom.sizes \
    --chroms $(paste -s -d ' ' $reference_dir/hg38_chroms.txt)  \
    --epochs 100 \
    --splits $project_dir/splits.json \
    --params $project_dir/cherimoya_params.json \
    --model-output-filename $1 \
    --input-seq-len ${input_seq_len} \
    --output-len ${output_len} \
    --batch-size 64 \
    --learning-rate $learning_rate \
    --seed $seed 2>&1 | tee $model_dir/trainer.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "training failed"; exit 1; fi


# modify the testing_input json for prediction
cp $project_dir/testing_input.json $project_dir/testing_input_all.json
sed -i -e "s/<experiment>/$1/g" $project_dir/testing_input_all.json
sed -i -e "s/<test_loci>/combined/g" $project_dir/testing_input_all.json

cp $project_dir/testing_input.json $project_dir/testing_input_peaks.json
sed -i -e "s/<experiment>/$1/g" $project_dir/testing_input_peaks.json
sed -i -e "s/<test_loci>/peaks/g" $project_dir/testing_input_peaks.json

# default values for the test_indices files; will be overwritten with acutall values if present
test_peaks_test_chroms_indices_file='None'
test_peaks_all_chroms_indices_file='None'
all_peaks_all_chroms_indices_file='None'
all_peaks_test_chroms_indices_file='None'

if [[ -n "${indices_files}" ]];then
    if [[ ${indices_files} != '' ]];then

     seq 0 $(wc -l ${data_dir}/${experiment}_peaks.bed | awk '{print $1-1}')> $indices_dir/test_peaks_all_chroms_indices.txt

     seq 0 $(wc -l ${data_dir}/${experiment}_combined.bed | awk '{print $1-1}')> $indices_dir/all_peaks_all_chroms_indices.txt

     test_peaks_test_chroms_indices_file=$(jq '.["0"]["loci_test_indices_file"]' $project_dir/splits.json | sed 's/"//g')

     echo "test_peaks_test_chroms_indices_file:" $test_peaks_test_chroms_indices_file

     test_peaks_all_chroms_indices_file=$indices_dir/test_peaks_all_chroms_indices.txt
     all_peaks_all_chroms_indices_file=$indices_dir/all_peaks_all_chroms_indices.txt

     background_test_indices_file=$(jq '.["0"]["background_test_indices_file"]' $project_dir/splits.json | sed 's/"//g')
     number_of_peaks=$(wc -l < ${data_dir}/${experiment}_peaks.bed)
     awk -v var="$number_of_peaks" '{print ($1 + var)}' $background_test_indices_file > $indices_dir/background_test_indices_file_global_index.txt
     cat $test_peaks_test_chroms_indices_file $indices_dir/background_test_indices_file_global_index.txt > $indices_dir/all_peaks_test_chroms_indices.txt

     all_peaks_test_chroms_indices_file=$indices_dir/all_peaks_test_chroms_indices.txt

    fi
fi

#get the test chromosome for chromosome wise training regime

if [[ -n "$(jq '.["0"]["test"] // empty' $project_dir/splits.json)" ]]; then
    test_chromosome=`jq '.["0"]["test"] | join(" ")' $project_dir/splits.json | sed 's/"//g'`
    test_all_chromosome=$(paste -s -d ' ' $reference_dir/hg38_chroms.txt)
else
    test_chromosome='None'
    test_all_chromosome='None'
fi
echo "test_chromosome=$test_chromosome"


function cherimoya_predict {
    # $1 output dir, $2 chroms, $3 indices file, $4 testing input json, $5 extra flags
    echo $( timestamp ): "
python $scripts_dir/cherimoya_predict.py \\
    --model $model_dir/${experiment}_split000.torch \\
    --chrom-sizes $reference_dir/chrom.sizes \\
    --chroms $2 \\
    --test-indices-file $3 \\
    --reference-genome $reference_dir/hg38.genome.fa \\
    --output-dir $1 \\
    --input-data $4 \\
    --sequence-generator-name BPNet \\
    --input-seq-len ${input_seq_len} \\
    --output-len ${output_len} \\
    --output-window-size ${output_len} \\
    --batch-size 256 \\
    --generate-predicted-profile-bigWigs \\
    --reverse-complement-average $5" | tee -a $logfile

    python $scripts_dir/cherimoya_predict.py \
        --model $model_dir/${experiment}_split000.torch \
        --chrom-sizes $reference_dir/chrom.sizes \
        --chroms $2 \
        --test-indices-file $3 \
        --reference-genome $reference_dir/hg38.genome.fa \
        --output-dir $1 \
        --input-data $4 \
        --sequence-generator-name BPNet \
        --input-seq-len ${input_seq_len} \
        --output-len ${output_len} \
        --output-window-size ${output_len} \
        --batch-size 256 \
        --generate-predicted-profile-bigWigs \
        --reverse-complement-average $5 || exit 1
}

function auprc_auroc {
    # $1 predictions dir
    echo $( timestamp ): "
python $scripts_dir/auprc_auroc_calculations.py \\
    --h5_file $1/${experiment}_split000_predictions.h5 \\
    --output_dir $1 \\
    --peak_file ${data_dir}/${experiment}_peaks.bed \\
    --neg_file ${data_dir}/${experiment}_background_regions.bed \\
    --output_len ${output_len} \\
    --chroms $test_chromosome" | tee -a $logfile

    python $scripts_dir/auprc_auroc_calculations.py \
        --h5_file $1/${experiment}_split000_predictions.h5 \
        --output_dir $1 \
        --peak_file ${data_dir}/${experiment}_peaks.bed \
        --neg_file ${data_dir}/${experiment}_background_regions.bed \
        --output_len ${output_len} \
        --chroms $test_chromosome || exit 1
}

cherimoya_predict $predictions_dir_all_peaks_test_chroms "$test_chromosome" $all_peaks_test_chroms_indices_file $project_dir/testing_input_all.json ""
echo $( timestamp ): "Calculating the AUPRC and AUROC metrics ..."
auprc_auroc $predictions_dir_all_peaks_test_chroms

cherimoya_predict $predictions_dir_all_peaks_test_chroms_wo_bias "$test_chromosome" $all_peaks_test_chroms_indices_file $project_dir/testing_input_all.json "--set-bias-as-zero"
echo $( timestamp ): "Calculating the AUPRC and AUROC metrics without bias..."
auprc_auroc $predictions_dir_all_peaks_test_chroms_wo_bias

cherimoya_predict $predictions_dir_test_peaks_test_chroms "$test_chromosome" $test_peaks_test_chroms_indices_file $project_dir/testing_input_peaks.json ""
cherimoya_predict $predictions_dir_test_peaks_test_chroms_wo_bias "$test_chromosome" $test_peaks_test_chroms_indices_file $project_dir/testing_input_peaks.json "--set-bias-as-zero"

if [[ "$predict_all_chroms" == "true" ]]; then
    cherimoya_predict $predictions_dir_all_peaks_all_chroms "$test_all_chromosome" $all_peaks_all_chroms_indices_file $project_dir/testing_input_all.json ""
    cherimoya_predict $predictions_dir_test_peaks_all_chroms "$test_all_chromosome" $test_peaks_all_chroms_indices_file $project_dir/testing_input_peaks.json ""
fi


# save the values used for training
cp $project_dir/training_input.json $model_dir/
cp ${data_dir}/${experiment}_peaks.bed $model_dir/
cp ${data_dir}/${experiment}_background_regions.bed $model_dir/
cp ${data_dir}/${experiment}_combined.bed $model_dir/
cp $project_dir/cherimoya_params.json $model_dir/
cp $project_dir/splits.json $model_dir/
cp $logfile $model_dir/

if [[ -n "${indices_files}" ]];then
    if [[ ${indices_files} != '' ]];then
        for f in $(jq -r '.["0"] | to_entries[] | select(.key | endswith("indices_file")) | .value' $project_dir/splits.json); do
            cp $f $model_dir/
        done
    fi
fi
