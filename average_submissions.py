import pandas

filenames = ["subm/submission_loss_continuous_runs_1_r_224_c_224_model_num_0_2016-06-28-14-29.csv",
		     "subm/submission_loss_continuous_runs_1_r_224_c_224_model_num_3_2016-06-28-17-28.csv", 
		     "subm/submission_loss_continuous_runs_1_r_224_c_224_model_num_1_2016-06-28-15-38.csv",
		     "subm/submission_loss_continuous_runs_1_r_224_c_224_model_num_4_2016-06-28-18-23.csv", 
		     "subm/submission_loss_continuous_runs_1_r_224_c_224_model_num_2_2016-06-28-16-46.csv",
		     "subm/submission_loss_continuous_runs_2_r_224_c_224_model_num_0_2016-06-29-01-09.csv"
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