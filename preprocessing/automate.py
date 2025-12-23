import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("data.csv")
print("Data awal:", df.shape)

# ===============================
# 2. AMBIL KOLOM NUMERIK
# ===============================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

# ===============================
# 3. SCALING
# ===============================
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

print("Scaling selesai")

# ===============================
# 4. BINNING PRICE (SCALED)
# ===============================
bins = [
    df_scaled["price"].quantile(0.0),
    df_scaled["price"].quantile(0.33),
    df_scaled["price"].quantile(0.66),
    df_scaled["price"].quantile(1.0)
]

labels = ["Low", "Medium", "High"]

df_scaled["price_category"] = pd.cut(
    df_scaled["price"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

print("\nDistribusi kategori harga:")
print(df_scaled["price_category"].value_counts())

# ===============================
# 5. SPLIT FEATURE & TARGET
# ===============================
X = df_scaled.drop(columns=["price", "price_category"])
y = df_scaled["price_category"]

# ===============================
# 6. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData siap digunakan")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ===============================
# 7. SIMPAN DATA
# ===============================
df_scaled.to_csv("housedata-preprocessing.csv", index=False)
print("\nhousedata-preprocessing.csv berhasil dibuat")
