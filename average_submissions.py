import pandas

filenames = ["subm/submission_loss__vgg_16_generator_randomSplit_testaugment10_r_224_c_224_folds_4_ep_20_2016-07-09-05-53.csv",
			 "subm/submission_loss__vgg_16_generator_randomSplit_r_224_c_224_folds_4_ep_20_2016-07-07-08-45.csv",
			 "subm/submission_loss__vgg_16_generator3_randomSplit_r_224_c_224_folds_4_ep_20_2016-07-08-09-19.csv"
	]
	
dfs = []

for filename in filenames:
	df = pandas.read_csv(filename)
	df = df.sort_values(by='img')

	dfs.append(df)

print(dfs[0][:5])
print(dfs[1][:5])

df_all = pandas.concat(dfs)
print(df_all.shape)
df_all = df_all.groupby('img').mean()
print(df_all.shape)

# df_all = pandas.concat(dfs, axis=1)
# print(df_all.shape)
# df_all = df_all.mean(axis=1)
# print(df_all.shape)


df_all.to_csv('averaged_predictions.csv')
# pands.write_csv(df_all)

# print(df_all[:5])
# print((pds[0] + pds[1])[:5])