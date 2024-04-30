import numpy as np
import scipy
import re
import pandas as pd

with open("fmri_rs.npy", "rb") as f:
  fmri_rs = np.load(f)

#Each sample is a row
fmri_rs = fmri_rs.T

#Get Split to divide into train + test
mat_file = scipy.io.loadmat("MMP_HCP_60_splits.mat")
seed_1 = mat_file['folds']['seed_1']
subject_lists = seed_1[0, 0]['sub_fold'][0, 0]['subject_list']
test_subjects = [int(item[0]) for item in subject_lists[0,0].flatten()]

#Get HCP test subjects
HCP_753_Subjects = []
with open('MMP_HCP_753_subs.txt', 'r') as file:
    HCP_753_Subjects = [int(re.sub('\n', '', line)) for line in file.readlines()]

#Put the HCP test subjects into a dataframe
df = pd.read_csv("MMP_HCP_componentscores.csv")
df['Subject'] = pd.to_numeric(df['Subject'], errors='coerce')
df = df[df['Subject'].isin(HCP_753_Subjects)].reset_index(drop = True)

#Split all our data into a Train and Test Set
df_train, df_test = df[~df['Subject'].isin(test_subjects)], df[df['Subject'].isin(test_subjects)]

#Create train and test arrays
train_subjects = df_train.index.to_list()
test_subjects = df_test.index.to_list()

X_train, Y_train = fmri_rs[train_subjects], df_train["varimax_cog"].to_numpy()
X_test, Y_test = fmri_rs[test_subjects], df_test["varimax_cog"].to_numpy()
