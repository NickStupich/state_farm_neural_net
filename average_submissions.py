import pandas


def average_submissions(filenames_in, filename_out):
		
	dfs = []

	for filename in filenames_in:
		df = pandas.read_csv(filename)
		df = df.sort_values(by='img')

		dfs.append(df)

	df_all = pandas.concat(dfs)
	df_all = df_all.groupby('img').mean()

	df_all.to_csv(filename_out)


def main():
	filenames = ["subm2/predictions_run_gen_vgg16_30_0.1_0.1_10.0_0.1_10.0_randomSplitfolds_0-10test_samples_5.csv",
				"subm/average of generate_randomSplit + 10 random test augments + generator3.csv",
				"subm2/predictions_run_gen_vgg16_30_0.1_0.1_10.0_0.1_10.0_randomSplitfolds_0-20test_samples_1.csv"
	]
	average_submissions(filenames, 'averaged_predictions.csv')

if __name__ == "__main__":
	main()
