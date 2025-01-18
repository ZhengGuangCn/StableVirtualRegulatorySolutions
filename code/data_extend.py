'''
Since the number of real samples is too small, the data needs to be expanded by adding a Gaussian noise of 0.1 to the real samples and expanding to 10,000 samples

File Induction:
data: The real sample input values and output values used for data augmentation.
'''
import pandas as pd
import numpy as np

data = pd.read_excel('../data/T-RA16.xlsx')
# Input layer data and output layer data for real samples.
data = pd.DataFrame(data)
data = data.T
print(data)
rad = data.iloc[0,:]
print(rad)

def augment_data(x,num_augments=9999):  # Here num_augments are multiples of the expansion
    X_augmented = [x]
    for _ in range(num_augments):
        augmented_sample = x.copy()   # Put the original data in the first line of the file that expands the data
        for i in range(len(x)):
            noise = np.random.normal(0, 0.1)
            augmented_sample[i] += noise   # Adding 0.1 Gaussian noise
        X_augmented.append(augmented_sample)
    return np.vstack(X_augmented)

data_rad = augment_data(rad)
data_rad = pd.DataFrame(data_rad)
data_rad.to_csv('../data/T-RA16-train.csv', index=False)


