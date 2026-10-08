version 1.0

# Distilled cherimoya student from the peak-wise teachers (run_cherimoya_modelling.wdl).
# Outputs follow run_modelling.wdl for the student; the teacher ensemble and the
# released model (optional inputs: its predictions_and_metrics_*_test_chroms
# columns) are scored on the same test set T into *_ensemble / *_released outputs.

task run_cherimoya_distillation {
	input {
		String experiment
		File training_input_json
		File testing_input_json
		Array [File] teacher_models
		File reference_file
		File reference_file_index
		File chrom_sizes
		File chroms_txt
		Array [File] bigwigs
		File peaks
		File background_regions
		Int input_seq_len
		Int output_len
		Array [File] indices_files
		Int number_of_folds
		Int student_seed
		Array [File] released_test_peaks_predictions
		Array [File] released_all_peaks_predictions
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

		released_test=${sep=',' released_test_peaks_predictions}
		released_all=${sep=',' released_all_peaks_predictions}

		/my_scripts/tf-atlas-cherimoya/anvil/modeling/cherimoya_distillation_pipeline.sh ${experiment} ${training_input_json} ${testing_input_json} ${sep=',' teacher_models} ${reference_file} ${reference_file_index} ${chrom_sizes} ${chroms_txt} ${sep=',' bigwigs} ${peaks} ${background_regions} ${input_seq_len} ${output_len} ${sep=',' indices_files} ${number_of_folds} ${student_seed} "$released_test" "$released_all" || exit 1

		echo "copying all files to cromwell_root folder"
		cp -r /project/model $workdir/
		for d in /project/predictions_and_metrics_*; do cp -r $d $workdir/; done

		for s in "" _ensemble _released; do
			cp /project/predictions_and_metrics_test_peaks_test_chroms$s/spearman.txt $workdir/spearman$s.txt
			cp /project/predictions_and_metrics_test_peaks_test_chroms$s/pearson.txt $workdir/pearson$s.txt
			cp /project/predictions_and_metrics_test_peaks_test_chroms$s/jsd.txt $workdir/jsd$s.txt
			cp /project/predictions_and_metrics_all_peaks_test_chroms$s/spearman.txt $workdir/spearman_all_peaks$s.txt
			cp /project/predictions_and_metrics_all_peaks_test_chroms$s/pearson.txt $workdir/pearson_all_peaks$s.txt
			cp /project/predictions_and_metrics_all_peaks_test_chroms$s/jsd.txt $workdir/jsd_all_peaks$s.txt
			cp /project/predictions_and_metrics_all_peaks_test_chroms$s/auprc.txt $workdir/auprc$s.txt
			cp /project/predictions_and_metrics_all_peaks_test_chroms$s/auroc.txt $workdir/auroc$s.txt
		done
		cp /project/predictions_and_metrics_test_peaks_test_chroms_wo_bias/spearman.txt $workdir/spearman_wo_bias.txt
		cp /project/predictions_and_metrics_test_peaks_test_chroms_wo_bias/pearson.txt $workdir/pearson_wo_bias.txt
		cp /project/predictions_and_metrics_test_peaks_test_chroms_wo_bias/jsd.txt $workdir/jsd_wo_bias.txt
		cp /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/spearman.txt $workdir/spearman_all_peaks_wo_bias.txt
		cp /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/pearson.txt $workdir/pearson_all_peaks_wo_bias.txt
		cp /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/jsd.txt $workdir/jsd_all_peaks_wo_bias.txt
		cp /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/auprc.txt $workdir/auprc_wo_bias.txt
		cp /project/predictions_and_metrics_all_peaks_test_chroms_wo_bias/auroc.txt $workdir/auroc_wo_bias.txt
	}

	output {
		Array[File] model = glob("model/*")
		Array[File] predictions_and_metrics_test_peaks_test_chroms = glob("predictions_and_metrics_test_peaks_test_chroms/*")
		Array[File] predictions_and_metrics_test_peaks_all_chroms = glob("predictions_and_metrics_test_peaks_all_chroms/*")
		Array[File] predictions_and_metrics_all_peaks_all_chroms = glob("predictions_and_metrics_all_peaks_all_chroms/*")
		Array[File] predictions_and_metrics_all_peaks_test_chroms = glob("predictions_and_metrics_all_peaks_test_chroms/*")
		Array[File] predictions_and_metrics_test_peaks_test_chroms_ensemble = glob("predictions_and_metrics_test_peaks_test_chroms_ensemble/*")
		Array[File] predictions_and_metrics_all_peaks_test_chroms_ensemble = glob("predictions_and_metrics_all_peaks_test_chroms_ensemble/*")
		Array[File] predictions_and_metrics_test_peaks_test_chroms_released = glob("predictions_and_metrics_test_peaks_test_chroms_released/*")
		Array[File] predictions_and_metrics_all_peaks_test_chroms_released = glob("predictions_and_metrics_all_peaks_test_chroms_released/*")

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

		Float spearman_ensemble = read_float("spearman_ensemble.txt")
		Float pearson_ensemble = read_float("pearson_ensemble.txt")
		Float jsd_ensemble = read_float("jsd_ensemble.txt")
		Float spearman_all_peaks_ensemble = read_float("spearman_all_peaks_ensemble.txt")
		Float pearson_all_peaks_ensemble = read_float("pearson_all_peaks_ensemble.txt")
		Float jsd_all_peaks_ensemble = read_float("jsd_all_peaks_ensemble.txt")
		Float auprc_ensemble = read_float("auprc_ensemble.txt")
		Float auroc_ensemble = read_float("auroc_ensemble.txt")

		Float spearman_released = read_float("spearman_released.txt")
		Float pearson_released = read_float("pearson_released.txt")
		Float jsd_released = read_float("jsd_released.txt")
		Float spearman_all_peaks_released = read_float("spearman_all_peaks_released.txt")
		Float pearson_all_peaks_released = read_float("pearson_all_peaks_released.txt")
		Float jsd_all_peaks_released = read_float("jsd_all_peaks_released.txt")
		Float auprc_released = read_float("auprc_released.txt")
		Float auroc_released = read_float("auroc_released.txt")
	}

	runtime {
		docker: docker
		cpu: 8
		memory: 30 + "GB"
		bootDiskSizeGb: 50
		disks: "local-disk 100 HDD"
		gpuType: gpuType
		gpuCount: 1
		zones: zones
		nvidiaDriverVersion: "535.161.08"
		maxRetries: 1
	}
}

workflow cherimoya_distillation {
	input {
		String experiment
		File training_input_json
		File testing_input_json
		Array [Array [File]] teacher_models
		File reference_file
		File reference_file_index
		File chrom_sizes
		File chroms_txt
		Array [File] bigwigs
		File peaks
		File background_regions
		Int input_seq_len = 2114
		Int output_len = 1000
		Array [File] indices_files
		Int number_of_folds = 20
		Int student_seed = 42
		Array [File] released_test_peaks_predictions = []
		Array [File] released_all_peaks_predictions = []
		String docker = "vivekramalingam/tf-atlas:gcp-cherimoya_v0.2.0-g2"
		String gpuType = "nvidia-tesla-t4"
		String zones = "us-west4-a us-west4-b us-west4-c"
	}

	call run_cherimoya_distillation {
		input:
			experiment = experiment,
			training_input_json = training_input_json,
			testing_input_json = testing_input_json,
			teacher_models = flatten(teacher_models),
			reference_file = reference_file,
			reference_file_index = reference_file_index,
			chrom_sizes = chrom_sizes,
			chroms_txt = chroms_txt,
			bigwigs = bigwigs,
			peaks = peaks,
			background_regions = background_regions,
			input_seq_len = input_seq_len,
			output_len = output_len,
			indices_files = indices_files,
			number_of_folds = number_of_folds,
			student_seed = student_seed,
			released_test_peaks_predictions = released_test_peaks_predictions,
			released_all_peaks_predictions = released_all_peaks_predictions,
			docker = docker,
			gpuType = gpuType,
			zones = zones
	}
	output {
		Array[File] model = run_cherimoya_distillation.model
		Array[File] predictions_and_metrics_test_peaks_test_chroms = run_cherimoya_distillation.predictions_and_metrics_test_peaks_test_chroms
		Array[File] predictions_and_metrics_test_peaks_all_chroms = run_cherimoya_distillation.predictions_and_metrics_test_peaks_all_chroms
		Array[File] predictions_and_metrics_all_peaks_all_chroms = run_cherimoya_distillation.predictions_and_metrics_all_peaks_all_chroms
		Array[File] predictions_and_metrics_all_peaks_test_chroms = run_cherimoya_distillation.predictions_and_metrics_all_peaks_test_chroms
		Array[File] predictions_and_metrics_test_peaks_test_chroms_ensemble = run_cherimoya_distillation.predictions_and_metrics_test_peaks_test_chroms_ensemble
		Array[File] predictions_and_metrics_all_peaks_test_chroms_ensemble = run_cherimoya_distillation.predictions_and_metrics_all_peaks_test_chroms_ensemble
		Array[File] predictions_and_metrics_test_peaks_test_chroms_released = run_cherimoya_distillation.predictions_and_metrics_test_peaks_test_chroms_released
		Array[File] predictions_and_metrics_all_peaks_test_chroms_released = run_cherimoya_distillation.predictions_and_metrics_all_peaks_test_chroms_released
		Float spearman = run_cherimoya_distillation.spearman
		Float pearson = run_cherimoya_distillation.pearson
		Float jsd = run_cherimoya_distillation.jsd
		Float spearman_all_peaks = run_cherimoya_distillation.spearman_all_peaks
		Float pearson_all_peaks = run_cherimoya_distillation.pearson_all_peaks
		Float jsd_all_peaks = run_cherimoya_distillation.jsd_all_peaks
		Float auprc = run_cherimoya_distillation.auprc
		Float auroc = run_cherimoya_distillation.auroc
		Float spearman_wo_bias = run_cherimoya_distillation.spearman_wo_bias
		Float pearson_wo_bias = run_cherimoya_distillation.pearson_wo_bias
		Float jsd_wo_bias = run_cherimoya_distillation.jsd_wo_bias
		Float spearman_all_peaks_wo_bias = run_cherimoya_distillation.spearman_all_peaks_wo_bias
		Float pearson_all_peaks_wo_bias = run_cherimoya_distillation.pearson_all_peaks_wo_bias
		Float jsd_all_peaks_wo_bias = run_cherimoya_distillation.jsd_all_peaks_wo_bias
		Float auprc_wo_bias = run_cherimoya_distillation.auprc_wo_bias
		Float auroc_wo_bias = run_cherimoya_distillation.auroc_wo_bias
		Float spearman_ensemble = run_cherimoya_distillation.spearman_ensemble
		Float pearson_ensemble = run_cherimoya_distillation.pearson_ensemble
		Float jsd_ensemble = run_cherimoya_distillation.jsd_ensemble
		Float spearman_all_peaks_ensemble = run_cherimoya_distillation.spearman_all_peaks_ensemble
		Float pearson_all_peaks_ensemble = run_cherimoya_distillation.pearson_all_peaks_ensemble
		Float jsd_all_peaks_ensemble = run_cherimoya_distillation.jsd_all_peaks_ensemble
		Float auprc_ensemble = run_cherimoya_distillation.auprc_ensemble
		Float auroc_ensemble = run_cherimoya_distillation.auroc_ensemble
		Float spearman_released = run_cherimoya_distillation.spearman_released
		Float pearson_released = run_cherimoya_distillation.pearson_released
		Float jsd_released = run_cherimoya_distillation.jsd_released
		Float spearman_all_peaks_released = run_cherimoya_distillation.spearman_all_peaks_released
		Float pearson_all_peaks_released = run_cherimoya_distillation.pearson_all_peaks_released
		Float jsd_all_peaks_released = run_cherimoya_distillation.jsd_all_peaks_released
		Float auprc_released = run_cherimoya_distillation.auprc_released
		Float auroc_released = run_cherimoya_distillation.auroc_released
	}
}
