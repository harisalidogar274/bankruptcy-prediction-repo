from data_prep_new import prepped
train, test = prepped["csv_result-3year.csv"]
print(f"{train} is train")
print(f"{test} is test")
train.to_csv('training_dataframe.csv',index=False)
test.to_csv('testing_dataframe.csv',index=False)