version 1.0

# Peak-wise cherimoya teachers. One task per fold, scattered; the task is
# tf-atlas-pipeline's run_modelling (v2.1.0-rc.5) with cherimoya in place of
# bpnet: same inputs, same /project layout, same output files and names.
# The fold's splits json is written in the task, in the layout of the old
# peak_wise_split_fold<k>.json templates; indices_files = the
# create_peak_wise_splits outputs (all folds).

task run_cherimoya_modelling {
	input {
		String experiment
		Int fold
		File training_input_json
		File testing_input_json
		File cherimoya_params_json
		File reference_file
		File reference_file_index
		File chrom_sizes
		File chroms_txt
		Array [File] bigwigs
		File peaks
		File background_regions
		Float learning_rate
		Int input_seq_len
		Int output_len
		Array [File] indices_files
		Int seed
		Boolean predict_all_chroms
		String docker
		String gpuType
		String zones
	}

	command {
		# the task working directory (/cromwell_root on PAPI, /mnt/disks/cromwell_root on
		# GCP Batch): outputs are copied here, so record it before cd-ing away
		workdir=$(pwd)

		#create data directories and download scripts
		cd /; mkdir my_scripts
		cd /my_scripts
		git clone --depth 1 --branch v0.1.1 https://github.com/viramalingam/tf-atlas-cherimoya.git
		chmod -R 777 tf-atlas-cherimoya
		cd tf-atlas-cherimoya/anvil/modeling/

		# splits json for this fold (layout of peak_wise_split_fold0.json)
		/my_scripts/tf-atlas-cherimoya/anvil/modeling/peak_wise_split_fold_json.sh ${fold} $workdir/splits_fold${fold}.json
		cat $workdir/splits_fold${fold}.json

		##modelling

		echo "run /my_scripts/tf-atlas-cherimoya/anvil/modeling/cherimoya_modelling_pipeline.sh" ${experiment} ${training_input_json} ${testing_input_json} ${cherimoya_params_json} $workdir/splits_fold${fold}.json ${reference_file} ${reference_file_index} ${chrom_sizes} ${chroms_txt} ${sep=',' bigwigs} ${peaks} ${background_regions} ${learning_rate} ${input_seq_len} ${output_len} ${sep=',' indices_files} ${seed} ${true='true' false='false' predict_all_chroms}
		/my_scripts/tf-atlas-cherimoya/anvil/modeling/cherimoya_modelling_pipeline.sh ${experiment} ${training_input_json} ${testing_input_json} ${cherimoya_params_json} $workdir/splits_fold${fold}.json ${reference_file} ${reference_file_index} ${chrom_sizes} ${chroms_txt} ${sep=',' bigwigs} ${peaks} ${background_regions} ${learning_rate} ${input_seq_len} ${output_len} ${sep=',' indices_files} ${seed} ${true='true' false='false' predict_all_chroms} || exit 1

		echo "copying all files to cromwell_root folder"

		cp /project/cherimoya_params.json $workdir/cherimoya_params.json
		cp -r /project/model $workdir/
		cp -r /project/predictions_and_metrics_test_peaks_test_chroms $workdir/
		cp -r /project/predictions_and_metrics_test_peaks_all_chroms $workdir/
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms $workdir/
		cp -r /project/predictions_and_metrics_all_peaks_all_chroms $workdir/

		cp -r /project/predictions_and_metrics_test_peaks_test_chroms/spearman.txt $workdir/spearman.txt
		cp -r /project/predictions_and_metrics_test_peaks_test_chroms/pearson.txt $workdir/pearson.txt
		cp -r /project/predictions_and_metrics_test_peaks_test_chroms/jsd.txt $workdir/jsd.txt

		cp -r /project/predictions_and_metrics_all_peaks_test_chroms/spearman.txt $workdir/spearman_all_peaks.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms/pearson.txt $workdir/pearson_all_peaks.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms/jsd.txt $workdir/jsd_all_peaks.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms/auprc.txt $workdir/auprc.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms/auroc.txt $workdir/auroc.txt

		cp -r /project/predictions_and_metrics_test_peaks_test_chroms_wo_bias/spearman.txt $workdir/spearman_wo_bias.txt
		cp -r /project/predictions_and_metrics_test_peaks_test_chroms_wo_bias/pearson.txt $workdir/pearson_wo_bias.txt
		cp -r /project/predictions_and_metrics_test_peaks_test_chroms_wo_bias/jsd.txt $workdir/jsd_wo_bias.txt

		cp -r /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/spearman.txt $workdir/spearman_all_peaks_wo_bias.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/pearson.txt $workdir/pearson_all_peaks_wo_bias.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/jsd.txt $workdir/jsd_all_peaks_wo_bias.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/auprc.txt $workdir/auprc_wo_bias.txt
		cp -r /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/auroc.txt $workdir/auroc_wo_bias.txt
	}

	output {
		File cherimoya_params_updated_json = "cherimoya_params.json"
		Array[File] model = glob("model/*")
		Array[File] predictions_and_metrics_test_peaks_test_chroms = glob("predictions_and_metrics_test_peaks_test_chroms/*")
		Array[File] predictions_and_metrics_test_peaks_all_chroms = glob("predictions_and_metrics_test_peaks_all_chroms/*")
		Array[File] predictions_and_metrics_all_peaks_all_chroms = glob("predictions_and_metrics_all_peaks_all_chroms/*")
		Array[File] predictions_and_metrics_all_peaks_test_chroms = glob("predictions_and_metrics_all_peaks_test_chroms/*")

		Float spearman = read_float("spearman.txt")
		Float pearson = read_float("pearson.txt")
		Float jsd = read_float("jsd.txt")

		Float spearman_all_peaks = read_float("spearman_all_peaks.txt")
		Float pearson_all_peaks = read_float("pearson_all_peaks.txt")
		Float jsd_all_peaks = read_float("jsd_all_peaks.txt")
		Float auprc = read_float("auprc.txt")
		Float auroc = read_float("auroc.txt")

		Float spearman_wo_bias = read_float("spearman_wo_bias.txt")
		Float pearson_wo_bias = read_float("pearson_wo_bias.txt")
		Float jsd_wo_bias = read_float("jsd_wo_bias.txt")

		Float spearman_all_peaks_wo_bias = read_float("spearman_all_peaks_wo_bias.txt")
		Float pearson_all_peaks_wo_bias = read_float("pearson_all_peaks_wo_bias.txt")
		Float jsd_all_peaks_wo_bias = read_float("jsd_all_peaks_wo_bias.txt")
		Float auprc_wo_bias = read_float("auprc_wo_bias.txt")
		Float auroc_wo_bias = read_float("auroc_wo_bias.txt")
	}

	runtime {
		docker: docker
		cpu: 8
		memory: 30 + "GB"
		bootDiskSizeGb: 50
		disks: "local-disk 50 HDD"
		gpuType: gpuType
		gpuCount: 1
		zones: zones
		nvidiaDriverVersion: "535.161.08"
		maxRetries: 1
	}
}

workflow cherimoya_modelling {
	input {
		String experiment
		File training_input_json
		File testing_input_json
		File cherimoya_params_json
		File reference_file
		File reference_file_index
		File chrom_sizes
		File chroms_txt
		Array [File] bigwigs
		File peaks
		File background_regions
		Float learning_rate = 0.001
		Int input_seq_len = 2114
		Int output_len = 1000
		Array [File] indices_files
		Int number_of_folds = 20
		Int seed_offset = 1000
		Boolean predict_all_chroms = false
		String docker = "vivekramalingam/tf-atlas:gcp-cherimoya_v0.2.0-g2"
		String gpuType = "nvidia-tesla-t4"
		String zones = "us-west4-a us-west4-b us-west4-c"
	}

	scatter (fold in range(number_of_folds)) {
		call run_cherimoya_modelling {
			input:
				experiment = experiment,
				fold = fold,
				training_input_json = training_input_json,
				testing_input_json = testing_input_json,
				cherimoya_params_json = cherimoya_params_json,
				reference_file = reference_file,
				reference_file_index = reference_file_index,
				chrom_sizes = chrom_sizes,
				chroms_txt = chroms_txt,
				bigwigs = bigwigs,
				peaks = peaks,
				background_regions = background_regions,
				learning_rate = learning_rate,
				input_seq_len = input_seq_len,
				output_len = output_len,
				indices_files = indices_files,
				seed = seed_offset + fold,
				predict_all_chroms = predict_all_chroms,
				docker = docker,
				gpuType = gpuType,
				zones = zones
		}
	}
	output {
		Array[File] cherimoya_params_updated_json = run_cherimoya_modelling.cherimoya_params_updated_json
		Array[Array[File]] model = run_cherimoya_modelling.model
		Array[Array[File]] predictions_and_metrics_all_peaks_test_chroms = run_cherimoya_modelling.predictions_and_metrics_all_peaks_test_chroms
		Array[Array[File]] predictions_and_metrics_test_peaks_test_chroms = run_cherimoya_modelling.predictions_and_metrics_test_peaks_test_chroms
		Array[Array[File]] predictions_and_metrics_all_peaks_all_chroms = run_cherimoya_modelling.predictions_and_metrics_all_peaks_all_chroms
		Array[Array[File]] predictions_and_metrics_test_peaks_all_chroms = run_cherimoya_modelling.predictions_and_metrics_test_peaks_all_chroms
		Array[Float] spearman = run_cherimoya_modelling.spearman
		Array[Float] pearson = run_cherimoya_modelling.pearson
		Array[Float] jsd = run_cherimoya_modelling.jsd
		Array[Float] spearman_all_peaks = run_cherimoya_modelling.spearman_all_peaks
		Array[Float] pearson_all_peaks = run_cherimoya_modelling.pearson_all_peaks
		Array[Float] jsd_all_peaks = run_cherimoya_modelling.jsd_all_peaks
		Array[Float] auprc = run_cherimoya_modelling.auprc
		Array[Float] auroc = run_cherimoya_modelling.auroc
		Array[Float] spearman_wo_bias = run_cherimoya_modelling.spearman_wo_bias
		Array[Float] pearson_wo_bias = run_cherimoya_modelling.pearson_wo_bias
		Array[Float] jsd_wo_bias = run_cherimoya_modelling.jsd_wo_bias
		Array[Float] spearman_all_peaks_wo_bias = run_cherimoya_modelling.spearman_all_peaks_wo_bias
		Array[Float] pearson_all_peaks_wo_bias = run_cherimoya_modelling.pearson_all_peaks_wo_bias
		Array[Float] jsd_all_peaks_wo_bias = run_cherimoya_modelling.jsd_all_peaks_wo_bias
		Array[Float] auprc_wo_bias = run_cherimoya_modelling.auprc_wo_bias
		Array[Float] auroc_wo_bias = run_cherimoya_modelling.auroc_wo_bias
	}
}
