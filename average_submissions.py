import pandas


def average_submissions(filenames_in, filename_out, power = 1.0):
		
	dfs = []
	print('computing average of submissions...')
	for filename in filenames_in:
		print(filename)
		df = pandas.read_csv(filename)
		df = df.sort_values(by='img')

		dfs.append(df)

	df_all = pandas.concat(dfs)
	df_all = df_all.groupby('img').mean()

	df_all = df_all.applymap(lambda x: x ** power)

	df_all.to_csv(filename_out)


def main():
	filenames = ["subm2/predictions_run_gen_vgg16_30_0.1_0.1_10.0_0.1_10.0_randomSplitfolds_0-10test_samples_5.csv",
				"subm/average of generate_randomSplit + 10 random test augments + generator3.csv",
				"subm2/predictions_run_gen_vgg16_30_0.1_0.1_10.0_0.1_10.0_randomSplitfolds_0-20test_samples_1.csv"
	]

	filenames = ["subm2/predictions__vgg_16_generator_randomSplitfolds_0-4test_samples_10pow1.0.csv"] * 10
	filenames += ['subm2/predictions__vgg_16_generator_randomSplitfolds_0-4test_samples_3pow1.0.csv'] * 3 	
	filenames += ['subm2/predictions__vgg_16_generator_randomSplitfolds_0-4test_samples_4pow1.0.csv'] * 4
	filenames += ['subm/submission_loss__vgg_16_generator_randomSplit_r_224_c_224_folds_4_ep_20_2016-07-07-08-45.csv'] * 15
	filenames += ['subm/submission_loss__vgg_16_generator3_randomSplit_r_224_c_224_folds_4_ep_20_2016-07-08-09-19.csv'] * 17

	average_submissions(filenames, 'subm2/predictions__vgg_16_generator_randomSplitfolds_0-4test_samples_17_NonAugmentEqualWeight_withgenerator3.csv')

if __name__ == "__main__":
	main()
