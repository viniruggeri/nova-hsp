import pandas as pd

print("=" * 70)
print("LABELS HORIZON 15 - TRAIN")
print("=" * 70)
df15 = pd.read_csv("data/processed/sir_graph/labels_horizon_15_train.csv")
print(df15.head(10))
print(f"\nDistribuição de labels:\n{df15['label_horizon_15'].value_counts()}")

print("\n" + "=" * 70)
print("LABELS HORIZON 20 - TRAIN")
print("=" * 70)
df20 = pd.read_csv("data/processed/sir_graph/labels_horizon_20_train.csv")
print(df20.head(10))
print(f"\nDistribuição de labels:\n{df20['label_horizon_20'].value_counts()}")

print("\n" + "=" * 70)
print("TRAIN.CSV - Verificar T_event")
print("=" * 70)
train = pd.read_csv("data/processed/sir_graph/train.csv")
print(f"T_event stats:\n{train['T_event'].describe()}")
