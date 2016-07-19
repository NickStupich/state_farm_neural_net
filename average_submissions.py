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
	filenames = ["subm/submission_loss__vgg_16_generator_randomSplit_testaugment10_r_224_c_224_folds_4_ep_20_2016-07-09-05-53.csv",
				 "subm/submission_loss__vgg_16_generator_randomSplit_r_224_c_224_folds_4_ep_20_2016-07-07-08-45.csv",
				 "subm/submission_loss__vgg_16_generator3_randomSplit_r_224_c_224_folds_4_ep_20_2016-07-08-09-19.csv"
		]
		average_submissions(filenames, 'averaged_predictions.csv')

if __name__ == "__main__":
	main()