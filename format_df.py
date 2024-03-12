from utility import get_dataset_path, get_dataframe_split
import numpy as np

def format_dataset(df):
    formatted_lines = []
    for index, row in df.iterrows():
        values = [str(value) for value in row.values]
        formatted_line = "{" + ", ".join([f"{value}f" for value in values]) + "}"
        formatted_lines.append(formatted_line)
    return formatted_lines

pathDSTest = get_dataset_path("TestBinarySHAPSNNImpactfulFeatures")
x_test , y_test = get_dataframe_split(pathDSTest)

'''y_test = y_test.head(100)
y_test = np.asarray(y_test)
y_test = [str(value) for value in y_test]
y_test = "{" + ", ".join(y_test) + "}"
print(y_test)'''

'''formatted_lines = format_dataset(x_test.head(100))

with open('formatted_data.txt', 'w') as file:
    for line in formatted_lines:
        file.write(line + '\n')'''
