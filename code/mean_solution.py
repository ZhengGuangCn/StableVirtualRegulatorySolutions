"""
The following code selects the models with the top-ranked losses, calculates the average of their parameters as the mean solution, and writes it to a new file.

First, read the files that record connection weights, node biases, output results, and loss values. The values in these files correspond to each other in the order of the trained models.
Then, sort the loss values in ascending order and retrieve the corresponding indices to filter the connection weights, node biases, and output results in the same order.
Subsequently, compute the mean of the parameters from the top N models with the smallest loss values as a relatively stable virtual solution and write it to a new file

File Introduction:
df: File storing non-zero weights and biases
df_outputs: A file used to store layer outputs
df_loss: A file used to store the loss values of each model
"""

import pandas as pd
import numpy as np

# load data
df = pd.read_csv('../data/final_weights-T-RA16.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df = pd.DataFrame(df)
df = df.T
df_a = df.iloc[0::2]      # Odd-numbered columns are weights
df_a = df_a.iloc[:,:240]  # The number of non-zero weights for each model
df_a = df_a.reset_index(drop=True)
print(df_a)

df_b = df.iloc[1::2]      # even-numbered columns are biases
df_b = df_b.iloc[:,:181]  # The number of biases for each model, which corresponds to the number of nodes in the network excluding the input layer
df_b = df_b.reset_index(drop=True)
print(df_b)

df_outputs = pd.read_csv('../data/processed_data_file-T-RA16.csv',header=None)
df_outputs = df_outputs.apply(pd.to_numeric, errors='coerce')
df_outputs = pd.DataFrame(df_outputs)
# df_outputs = df_outputs.iloc[0::2]
df_outputs = df_outputs.reset_index(drop=True)
print(df_outputs)

df_loss = pd.read_csv('../data/layer_losses.csv',header=None)
df_loss = df_loss.apply(pd.to_numeric, errors='coerce')
df_loss = pd.DataFrame(df_loss)
print(df_loss)

# Filter the data
# sort the values of df_loss in ascending order, and select the smallest 100 entries
bottom_300_indices = df_loss[0].nsmallest(300).index
bottom_200_indices = df_loss[0].nsmallest(200).index
bottom_100_indices = df_loss[0].nsmallest(100).index
bottom_50_indices = df_loss[0].nsmallest(50).index
bottom_10_indices = df_loss[0].nsmallest(10).index
bottom_3_indices = df_loss[0].nsmallest(3).index
bottom_1_indices = df_loss[0].nsmallest(1).index

common_df_a_300 = df_a.loc[bottom_300_indices]
common_df_b_300 = df_b.loc[bottom_300_indices]
common_df_o_300 = df_outputs.loc[bottom_300_indices]
common_df_loss_300 = df_loss.loc[bottom_300_indices]

common_df_a_200 = df_a.loc[bottom_200_indices]
common_df_b_200 = df_b.loc[bottom_200_indices]
common_df_o_200 = df_outputs.loc[bottom_200_indices]
common_df_loss_200 = df_loss.loc[bottom_200_indices]

common_df_a_100 = df_a.loc[bottom_100_indices]
common_df_b_100 = df_b.loc[bottom_100_indices]
common_df_o_100 = df_outputs.loc[bottom_100_indices]
common_df_loss_100 = df_loss.loc[bottom_100_indices]

common_df_a_50 = df_a.loc[bottom_50_indices]
common_df_b_50 = df_b.loc[bottom_50_indices]
common_df_o_50 = df_outputs.loc[bottom_50_indices]
common_df_loss_50 = df_loss.loc[bottom_50_indices]

common_df_a_10 = df_a.loc[bottom_10_indices]
common_df_b_10 = df_b.loc[bottom_10_indices]
common_df_o_10 = df_outputs.loc[bottom_10_indices]
common_df_loss_10 = df_loss.loc[bottom_10_indices]

common_df_a_3 = df_a.loc[bottom_3_indices]
common_df_b_3 = df_b.loc[bottom_3_indices]
common_df_o_3 = df_outputs.loc[bottom_3_indices]
common_df_loss_3 = df_loss.loc[bottom_3_indices]

common_df_a_1 = df_a.loc[bottom_1_indices]
common_df_b_1 = df_b.loc[bottom_1_indices]
common_df_o_1 = df_outputs.loc[bottom_1_indices]
common_df_loss_1 = df_loss.loc[bottom_1_indices]

# Calculate the mean of each column and print the results
# weight
mean_a_300 = common_df_a_300.mean(axis=0)
mean_a_200 = common_df_a_200.mean(axis=0)
mean_a_100 = common_df_a_100.mean(axis=0)
mean_a_50 = common_df_a_50.mean(axis=0)
mean_a_10 = common_df_a_10.mean(axis=0)
mean_a_3 = common_df_a_3.mean(axis=0)
mean_a_1 = common_df_a_1.mean(axis=0)
mean_a_mean = pd.concat([mean_a_10, mean_a_3, mean_a_1], axis=1)
mean_a_mean.to_csv('mean_a_closest_100_1_virtual-T-RA16.csv', index=False)

# bias
mean_b_300 = common_df_b_300.mean(axis=0)
mean_b_200 = common_df_b_200.mean(axis=0)
mean_b_100 = common_df_b_100.mean(axis=0)
mean_b_50 = common_df_b_50.mean(axis=0)
mean_b_10 = common_df_b_10.mean(axis=0)
mean_b_3 = common_df_b_3.mean(axis=0)
mean_b_1 = common_df_b_1.mean(axis=0)
mean_b_mean = pd.concat([mean_b_10, mean_b_3, mean_b_1], axis=1)
mean_b_mean.to_csv('mean_b_closest_100_1_virtual-T-RA16.csv', index=False)

# outputs
mean_o_300 = common_df_o_300.mean(axis=0)
mean_o_200 = common_df_o_200.mean(axis=0)
mean_o_100 = common_df_o_100.mean(axis=0)
mean_o_50 = common_df_o_50.mean(axis=0)
mean_o_10 = common_df_o_10.mean(axis=0)
mean_o_3 = common_df_o_3.mean(axis=0)
mean_o_1 = common_df_o_1.mean(axis=0)
mean_o_mean = pd.concat([mean_o_10, mean_o_3, mean_o_1], axis=1)
mean_o_mean = mean_o_mean.T
mean_o_mean.to_csv('mean_c_closest_100_1_virtual-T-RA16.csv', index=False)
