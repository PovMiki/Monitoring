import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X_train, y_train = make_classification(n_samples=500, n_features=5, random_state=42)
df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(5)])
df_train["target"] = y_train

X_prod, y_prod = make_classification(n_samples=300, n_features=5, random_state=999)
df_prod = pd.DataFrame(X_prod, columns=[f"feature_{i}" for i in range(5)])
df_prod["target"] = y_prod

model = RandomForestClassifier(random_state=42)
model.fit(df_train.drop("target", axis=1), df_train["target"])


df_prod["prediction"] = model.predict(df_prod.drop("target", axis=1))

print("ZBIOR TRENINGOWY")
print(f"wiersze :  {df_train.shape}")
print(f"\nTypy kolumn i brakujące wartości: {df_train.info()}")

print("ZBIOR PRODUKCYJNY")
print(f"wiersze : : {df_prod.shape}")
print(f"\nTypy kolumn i brakujące wartości: {df_prod.info()}")