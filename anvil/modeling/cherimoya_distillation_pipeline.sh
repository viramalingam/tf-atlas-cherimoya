#!/bin/bash

# Distillation stage of the cherimoya peak-wise recipe, in the layout of
# tf-atlas-pipeline anvil/modeling/modelling_pipeline.sh.
#
#   1. soft targets from the peak-wise teachers on the pool (all regions
#      except the common test set T)       cherimoya_distill_targets.py
#   2. distilled student                    cherimoya_distill_student.py
#   3. student predictions + metrics, same folders and files as the
#      modelling pipeline (test set = T = the fold-0 test indices)
#   4. teacher ensemble on T: every teacher predicts T, predictions averaged
#      (..._ensemble folders)
#   5. released model on T, if its prediction files are given: its existing
#      chrX predictions h5 restricted to T (..._released folders)

function timestamp {
    date +"%Y-%m-%d_%H-%M-%S" | tr -d '\n'
}

experiment=$1
training_input_json=$2
testing_input_json=$3
teacher_models=$4
reference_file=$5
reference_file_index=$6
chrom_sizes=$7
chroms_txt=$8
bigwigs=$9
peaks=${10}
background_regions=${11}
input_seq_len=${12}
output_len=${13}
indices_files=${14}
number_of_folds=${15}
student_seed=${16}
released_test_peaks_files=${17:-None}
released_all_peaks_files=${18:-None}

scripts_dir=$(dirname $(readlink -f $0))

mkdir /project
project_dir=/project
logfile=$project_dir/${1}_distillation.log
touch $logfile

data_dir=$project_dir/data
reference_dir=$project_dir/reference
model_dir=$project_dir/model
indices_dir=$project_dir/splits_indices
teachers_dir=$project_dir/teachers
distill_dir=$project_dir/distill
mkdir $data_dir $reference_dir $model_dir $indices_dir $teachers_dir $distill_dir

predictions_dir_all_peaks_all_chroms=$project_dir/predictions_and_metrics_all_peaks_all_chroms
predictions_dir_all_peaks_test_chroms=$project_dir/predictions_and_metrics_all_peaks_test_chroms
predictions_dir_test_peaks_test_chroms=$project_dir/predictions_and_metrics_test_peaks_test_chroms
predictions_dir_test_peaks_all_chroms=$project_dir/predictions_and_metrics_test_peaks_all_chroms
predictions_dir_all_peaks_test_chroms_wo_bias=$project_dir/predictions_and_metrics_all_peaks_test_chroms_wo_bias
predictions_dir_test_peaks_test_chroms_wo_bias=$project_dir/predictions_and_metrics_test_peaks_test_chroms_wo_bias
for d in $predictions_dir_all_peaks_all_chroms $predictions_dir_all_peaks_test_chroms \
         $predictions_dir_test_peaks_test_chroms $predictions_dir_test_peaks_all_chroms \
         $predictions_dir_all_peaks_test_chroms_wo_bias $predictions_dir_test_peaks_test_chroms_wo_bias \
         ${predictions_dir_test_peaks_test_chroms}_ensemble ${predictions_dir_all_peaks_test_chroms}_ensemble \
         ${predictions_dir_test_peaks_test_chroms}_released ${predictions_dir_all_peaks_test_chroms}_released; do
    mkdir $d
done

cp $reference_file $reference_dir/hg38.genome.fa
cp $reference_file_index $reference_dir/hg38.genome.fa.fai
cp $chrom_sizes $reference_dir/chrom.sizes
cp $chroms_txt $reference_dir/hg38_chroms.txt

echo $bigwigs | sed 's/,/ /g' | xargs cp -t $data_dir/
cp $peaks ${data_dir}/${experiment}_peaks.bed.gz
gunzip ${data_dir}/${experiment}_peaks.bed.gz
cp $background_regions ${data_dir}/${experiment}_background_regions.bed.gz
gunzip ${data_dir}/${experiment}_background_regions.bed.gz
cat ${data_dir}/${experiment}_peaks.bed ${data_dir}/${experiment}_background_regions.bed > ${data_dir}/${experiment}_combined.bed

cp $training_input_json $project_dir/training_input.json
sed -i -e "s/<experiment>/$1/g" $project_dir/training_input.json
cp $testing_input_json $project_dir/testing_input.json

echo $indices_files | sed 's/,/ /g' | xargs cp -t $indices_dir/

# the common test set T = the fold-0 test indices (identical for every fold)
printf '{"0": {"loci_val_indices_file": "/project/splits_indices/loci_valid_indices_fold0.txt", "loci_train_indices_file": "/project/splits_indices/loci_train_indices_fold0.txt", "background_val_indices_file": "/project/splits_indices/background_valid_indices_fold0.txt", "background_train_indices_file": "/project/splits_indices/background_train_indices_fold0.txt", "loci_test_indices_file": "/project/splits_indices/loci_test_indices_fold0.txt", "background_test_indices_file": "/project/splits_indices/background_test_indices_fold0.txt"}}\n' > $project_dir/splits.json

# teacher checkpoints: the best-by-validation <experiment>_split000.torch of each fold
i=0
for f in $(echo $teacher_models | sed 's/,/ /g'); do
    case $(basename $f) in
        ${experiment}_split000.torch)
            cp $f $teachers_dir/teacher$(printf '%02d' $i).torch
            i=$((i + 1));;
    esac
done
echo $( timestamp ): "$i teacher checkpoints" | tee -a $logfile
if [ $i -lt 1 ]; then echo "no teacher checkpoints"; exit 1; fi

nvidia-smi | tee -a $logfile

echo $( timestamp ): "distillation targets" | tee -a $logfile
python $scripts_dir/cherimoya_distill_targets.py \
    --teachers $teachers_dir/teacher*.torch \
    --input-data $project_dir/training_input.json \
    --indices-dir $indices_dir \
    --number-of-folds $number_of_folds \
    --reference-genome $reference_dir/hg38.genome.fa \
    --input-seq-len ${input_seq_len} \
    --output-len ${output_len} \
    --out $distill_dir/${experiment}_targets.h5 2>&1 | tee -a $logfile
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "targets failed"; exit 1; fi

echo $( timestamp ): "student" | tee -a $logfile
python $scripts_dir/cherimoya_distill_student.py \
    --targets $distill_dir/${experiment}_targets.h5 \
    --teacher $teachers_dir/teacher00.torch \
    --output-dir $model_dir \
    --model-output-filename $experiment \
    --seed $student_seed 2>&1 | tee $model_dir/trainer.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then echo "student training failed"; exit 1; fi


cp $project_dir/testing_input.json $project_dir/testing_input_all.json
sed -i -e "s/<experiment>/$1/g" $project_dir/testing_input_all.json
sed -i -e "s/<test_loci>/combined/g" $project_dir/testing_input_all.json
cp $project_dir/testing_input.json $project_dir/testing_input_peaks.json
sed -i -e "s/<experiment>/$1/g" $project_dir/testing_input_peaks.json
sed -i -e "s/<test_loci>/peaks/g" $project_dir/testing_input_peaks.json

seq 0 $(wc -l ${data_dir}/${experiment}_peaks.bed | awk '{print $1-1}') > $indices_dir/test_peaks_all_chroms_indices.txt
seq 0 $(wc -l ${data_dir}/${experiment}_combined.bed | awk '{print $1-1}') > $indices_dir/all_peaks_all_chroms_indices.txt
test_peaks_test_chroms_indices_file=$indices_dir/loci_test_indices_fold0.txt
number_of_peaks=$(wc -l < ${data_dir}/${experiment}_peaks.bed)
awk -v var="$number_of_peaks" '{print ($1 + var)}' $indices_dir/background_test_indices_fold0.txt > $indices_dir/background_test_indices_file_global_index.txt
cat $test_peaks_test_chroms_indices_file $indices_dir/background_test_indices_file_global_index.txt > $indices_dir/all_peaks_test_chroms_indices.txt
all_peaks_test_chroms_indices_file=$indices_dir/all_peaks_test_chroms_indices.txt
test_peaks_all_chroms_indices_file=$indices_dir/test_peaks_all_chroms_indices.txt
all_peaks_all_chroms_indices_file=$indices_dir/all_peaks_all_chroms_indices.txt


function cherimoya_predict {
    # $1 model, $2 output dir, $3 indices file, $4 testing input json, $5 extra flags
    echo $( timestamp ): "cherimoya_predict.py --model $1 --output-dir $2 --test-indices-file $3 --input-data $4 $5" | tee -a $logfile
    python $scripts_dir/cherimoya_predict.py \
        --model $1 \
        --chrom-sizes $reference_dir/chrom.sizes \
        --chroms None \
        --test-indices-file $3 \
        --reference-genome $reference_dir/hg38.genome.fa \
        --output-dir $2 \
        --input-data $4 \
        --sequence-generator-name BPNet \
        --input-seq-len ${input_seq_len} \
        --output-len ${output_len} \
        --output-window-size ${output_len} \
        --batch-size 256 \
        --reverse-complement-average $5 || exit 1
}

function auprc_auroc {
    # $1 predictions dir, $2 model tag
    python $scripts_dir/auprc_auroc_calculations.py \
        --h5_file $1/$2_predictions.h5 \
        --output_dir $1 \
        --peak_file ${data_dir}/${experiment}_peaks.bed \
        --neg_file ${data_dir}/${experiment}_background_regions.bed \
        --output_len ${output_len} \
        --chroms None || exit 1
}

student=$model_dir/${experiment}_split000.torch
cherimoya_predict $student $predictions_dir_all_peaks_test_chroms $all_peaks_test_chroms_indices_file $project_dir/testing_input_all.json "--generate-predicted-profile-bigWigs"
auprc_auroc $predictions_dir_all_peaks_test_chroms ${experiment}_split000
cherimoya_predict $student $predictions_dir_all_peaks_test_chroms_wo_bias $all_peaks_test_chroms_indices_file $project_dir/testing_input_all.json "--generate-predicted-profile-bigWigs --set-bias-as-zero"
auprc_auroc $predictions_dir_all_peaks_test_chroms_wo_bias ${experiment}_split000
cherimoya_predict $student $predictions_dir_test_peaks_test_chroms $test_peaks_test_chroms_indices_file $project_dir/testing_input_peaks.json "--generate-predicted-profile-bigWigs"
cherimoya_predict $student $predictions_dir_test_peaks_test_chroms_wo_bias $test_peaks_test_chroms_indices_file $project_dir/testing_input_peaks.json "--generate-predicted-profile-bigWigs --set-bias-as-zero"
cherimoya_predict $student $predictions_dir_all_peaks_all_chroms $all_peaks_all_chroms_indices_file $project_dir/testing_input_all.json "--generate-predicted-profile-bigWigs"
cherimoya_predict $student $predictions_dir_test_peaks_all_chroms $test_peaks_all_chroms_indices_file $project_dir/testing_input_peaks.json "--generate-predicted-profile-bigWigs"


echo $( timestamp ): "teacher ensemble on T" | tee -a $logfile
for t in $teachers_dir/teacher*.torch; do
    tag=$(basename $t .torch)
    cherimoya_predict $t $project_dir/teacher_predictions/$tag/test_peaks $test_peaks_test_chroms_indices_file $project_dir/testing_input_peaks.json ""
    cherimoya_predict $t $project_dir/teacher_predictions/$tag/all_peaks $all_peaks_test_chroms_indices_file $project_dir/testing_input_all.json ""
done
python $scripts_dir/score_predictions_h5.py \
    --h5 $project_dir/teacher_predictions/teacher*/test_peaks/teacher*_predictions.h5 \
    --peaks ${data_dir}/${experiment}_peaks.bed --test-indices-file $test_peaks_test_chroms_indices_file \
    --output-len ${output_len} --output-dir ${predictions_dir_test_peaks_test_chroms}_ensemble \
    --model-tag ${experiment}_split000 || exit 1
python $scripts_dir/score_predictions_h5.py \
    --h5 $project_dir/teacher_predictions/teacher*/all_peaks/teacher*_predictions.h5 \
    --peaks ${data_dir}/${experiment}_peaks.bed --test-indices-file $test_peaks_test_chroms_indices_file \
    --background ${data_dir}/${experiment}_background_regions.bed \
    --background-indices-file $indices_dir/background_test_indices_fold0.txt \
    --output-len ${output_len} --output-dir ${predictions_dir_all_peaks_test_chroms}_ensemble \
    --model-tag ${experiment}_split000 || exit 1
auprc_auroc ${predictions_dir_all_peaks_test_chroms}_ensemble ${experiment}_split000


if [[ "$released_test_peaks_files" != "None" && "$released_all_peaks_files" != "None" ]]; then
    echo $( timestamp ): "released model on T" | tee -a $logfile
    released_test_h5=$(echo $released_test_peaks_files | tr ',' '\n' | grep '_predictions.h5$' | head -1)
    released_all_h5=$(echo $released_all_peaks_files | tr ',' '\n' | grep '_predictions.h5$' | head -1)
    python $scripts_dir/score_predictions_h5.py \
        --h5 $released_test_h5 \
        --peaks ${data_dir}/${experiment}_peaks.bed --test-indices-file $test_peaks_test_chroms_indices_file \
        --output-len ${output_len} --output-dir ${predictions_dir_test_peaks_test_chroms}_released \
        --model-tag ${experiment}_split000 || exit 1
    python $scripts_dir/score_predictions_h5.py \
        --h5 $released_all_h5 \
        --peaks ${data_dir}/${experiment}_peaks.bed --test-indices-file $test_peaks_test_chroms_indices_file \
        --background ${data_dir}/${experiment}_background_regions.bed \
        --background-indices-file $indices_dir/background_test_indices_fold0.txt \
        --output-len ${output_len} --output-dir ${predictions_dir_all_peaks_test_chroms}_released \
        --model-tag ${experiment}_split000 || exit 1
    auprc_auroc ${predictions_dir_all_peaks_test_chroms}_released ${experiment}_split000
else
    for d in ${predictions_dir_test_peaks_test_chroms}_released ${predictions_dir_all_peaks_test_chroms}_released; do
        for m in pearson spearman jsd auprc auroc; do echo "nan" > $d/$m.txt; done
    done
fi


# save the values used for training
cp $project_dir/training_input.json $model_dir/
cp $project_dir/splits.json $model_dir/
cp ${data_dir}/${experiment}_peaks.bed $model_dir/
cp ${data_dir}/${experiment}_background_regions.bed $model_dir/
cp $logfile $model_dir/
ls -la $teachers_dir > $model_dir/teachers.txt
