import os
import glob
import pandas as pd
from datasets import Dataset, DatasetDict


def assemble_df_data_type(file_folder_drive, data_type, sep=","):
    # setting the path for joining multiple files
    files = os.path.join(f'{file_folder_drive}/', f'translated_{data_type}******.csv')
    print(files)
    files = glob.glob(files)
    print(files)
    files.sort()
    df_list = [pd.read_csv(f, sep=sep) for f in files]  
    df = pd.concat(df_list, ignore_index=True)    
    df_merged = df.copy()
    df_merged = df_merged.iloc[: , :]
    return Dataset.from_pandas(df_merged)


train = assemble_df_data_type("cnn_dailymail/cnn_dailymail_nor","train")

#train_dataset = assemble_df_data_type("cnn_dailymail/cnn_dailymail_nor", "cnn_dailymail","train")
test = pd.read_csv("cnn_dailymail/cnn_dailymail_nor/df_cnn_dailymail_test1.csv", sep=",")
test = Dataset.from_pandas(test)

val = pd.read_csv("cnn_dailymail/cnn_dailymail_nor/df_cnn_dailymail_validation1.csv", sep=",")
val = Dataset.from_pandas(val)

#val_dataset =  assemble_df_data_type("cnn_dailymail/cnn_dailymail_nor", "cnn_dailymail","val", sep=",")


print(train)
print(test)
master_dataset_dict = DatasetDict({"test":test, "train":train,"validation":val})


master_dataset_dict.push_to_hub("cnn_dailymail_nor_v2")