version 1.0

# tf-atlas-pipeline anvil/modeling/create_peak_wise_splits.wdl with three
# optional inputs for a common held-out test set T (see peak_wise_splits.py):
# test_chroms ("None" = old behaviour), test_frac, seed.

task run_peak_wise_splits {
	input {
		String experiment
		Array[File] bigwigs
		File peaks
		File nonpeaks
		Int number_of_folds
		Int input_seq_len
		String test_chroms
		Float test_frac
		Int seed
		String docker
	}

	command {
		#create data directories and download scripts
		cd /; mkdir my_scripts
		cd /my_scripts
		git clone --depth 1 --branch v0.1.0 https://github.com/viramalingam/tf-atlas-cherimoya.git
		chmod -R 777 tf-atlas-cherimoya
		cd tf-atlas-cherimoya/anvil/modeling/

		##peak_wise_splits

		echo "run /my_scripts/tf-atlas-cherimoya/anvil/modeling/peak_wise_splits_pipeline.sh" ${experiment} ${sep=',' bigwigs} ${peaks} ${nonpeaks} ${number_of_folds} ${input_seq_len} ${test_chroms} ${test_frac} ${seed}
		/my_scripts/tf-atlas-cherimoya/anvil/modeling/peak_wise_splits_pipeline.sh ${experiment} ${sep=',' bigwigs} ${peaks} ${nonpeaks} ${number_of_folds} ${input_seq_len} ${test_chroms} ${test_frac} ${seed} || exit 1

		echo "copying all files to cromwell_root folder"

		cp -r /project/splits_indices /cromwell_root/
		cp -r /project/supplemental_outputs /cromwell_root/
	}

	output {
		Array[File] peak_wise_splits = glob("splits_indices/*")
		Array[File] supplemental_outputs = glob("supplemental_outputs/*")
		File group_df_csv = "supplemental_outputs/group_df.csv"
		File splits_manifest_json = "supplemental_outputs/manifest.json"
		File test_peaks_bed = "supplemental_outputs/peaks_test_fold0.bed"
	}

	runtime {
		docker: docker
		memory: 8 + "GB"
		bootDiskSizeGb: 50
		disks: "local-disk 50 HDD"
		maxRetries: 1
	}
}

workflow create_peak_wise_splits {
	input {
		String experiment
		Array[File] bigwigs
		File peaks
		File nonpeaks
		Int number_of_folds
		Int input_seq_len = 2114
		String test_chroms = "None"
		Float test_frac = 0.05
		Int seed = 1234
		String docker = "vivekramalingam/tf-atlas:gcp-cherimoya_v0.2.0-g2"
	}

	call run_peak_wise_splits {
		input:
			experiment = experiment,
			bigwigs = bigwigs,
			peaks = peaks,
			nonpeaks = nonpeaks,
			number_of_folds = number_of_folds,
			input_seq_len = input_seq_len,
			test_chroms = test_chroms,
			test_frac = test_frac,
			seed = seed,
			docker = docker
	}
	output {
		Array[File] peak_wise_splits = run_peak_wise_splits.peak_wise_splits
		Array[File] supplemental_outputs = run_peak_wise_splits.supplemental_outputs
		File group_df_csv = run_peak_wise_splits.group_df_csv
		File splits_manifest_json = run_peak_wise_splits.splits_manifest_json
		File test_peaks_bed = run_peak_wise_splits.test_peaks_bed
	}
}
