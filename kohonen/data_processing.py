import pandas as pd

df = pd.read_csv("./Iris.csv")

print(df.shape)

def preprocessing(filename: str):
    df = pd.read_csv(filename)
    labels = df[""]