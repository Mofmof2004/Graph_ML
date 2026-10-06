import pandas as pd
import json

df = pd.read_csv("data/raw/full_dataset.csv", nrows=1000,
                 usecols=["title", "ingredients", "directions", "link", "source", "NER"])

print(df.shape)
print(df.dtypes)
print(df.iloc[0])

for col in ["ingredients", "directions", "NER"]:
    df[col] = df[col].apply(json.loads)

for _, row in df.head(5).iterrows():
    print("\n" + row["title"], f"({row['source']})")
    print("  raw:       ", row["ingredients"])
    print("  NER:       ", row["NER"])
    print("  directions:", row["directions"][:2], "...")