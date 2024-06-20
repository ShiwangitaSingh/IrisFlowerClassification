import pandas as pd
from pandas import DataFrame


def my_predict(df1: DataFrame) -> DataFrame:
    output = []
    for ind in df1.index:
        if df1['PetalLengthCm'][ind] <= 2:
            output.append('Iris-setosa')
        elif df1['PetalWidthCm'][ind] > 0.75 and df1['PetalWidthCm'][ind] < 1.7:
            output.append('Iris-versicolor')
        else:
            output.append('Iris-virginica')
    return pd.DataFrame(output, columns=['Species'])


