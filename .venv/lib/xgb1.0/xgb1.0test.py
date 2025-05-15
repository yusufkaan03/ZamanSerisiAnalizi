import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

train_csv = pd.read_csv("/home/yusuf/Downloads/academy2025/train.csv")
test_csv = pd.read_csv("/home/yusuf/Downloads/academy2025/testFeatures.csv")

train_csv["tarih"] = pd.to_datetime(train_csv["tarih"])
test_csv["tarih"] = pd.to_datetime(test_csv["tarih"])

train_csv["yıl"] = train_csv["tarih"].dt.year - 2019
train_csv["ay"] = train_csv["tarih"].dt.month
train_csv["ay_sin"] = np.sin(2 * np.pi * train_csv["ay"]/12)
train_csv["ay_cos"] = np.cos(2 * np.pi * train_csv["ay"]/12)

test_csv["yıl"] = test_csv["tarih"].dt.year - 2019
test_csv["ay"] = test_csv["tarih"].dt.month
test_csv["ay_sin"] = np.sin(2 * np.pi * test_csv["ay"]/12)
test_csv["ay_cos"] = np.cos(2 * np.pi * test_csv["ay"]/12)

grup_sutun = ["ürün", "ürün kategorisi", "ürün üretim yeri", "market", "şehir"]

train_yeni = train_csv.sort_values(by = ["şehir", "market", "ürün", "ürün üretim yeri", "tarih"])

train_yeni["1_fiyat"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].shift(1)
train_yeni["2_fiyat"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].shift(2)
train_yeni["3_fiyat"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].shift(3)
train_yeni["4_fiyat"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].shift(4)
train_yeni["5_fiyat"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].shift(5)
train_yeni["6_fiyat"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].shift(6)

train_yeni["3ort"] = (train_yeni["1_fiyat"] + train_yeni["2_fiyat"] + train_yeni["3_fiyat"]) / 3
train_yeni["6ort"] = (train_yeni["3ort"] + ((train_yeni["4_fiyat"] + train_yeni["5_fiyat"] + train_yeni["6_fiyat"]) / 3)) /2

train_yeni["1d"] = train_yeni["1_fiyat"] - train_yeni["2_fiyat"]
train_yeni["2d"] = train_yeni["2_fiyat"] - train_yeni["3_fiyat"]
train_yeni["3d"] = train_yeni["3_fiyat"] - train_yeni["4_fiyat"]
train_yeni["4d"] = train_yeni["4_fiyat"] - train_yeni["5_fiyat"]
train_yeni["5d"] = train_yeni["5_fiyat"] - train_yeni["6_fiyat"]

train_yeni["2o"] = (train_yeni["1d"] + train_yeni["2d"]) / 2
train_yeni["3o"] = (train_yeni["2o"] + train_yeni["3d"]) / 2
train_yeni["4o"] = (train_yeni["3o"] + train_yeni["4d"]) / 2
train_yeni["5o"] = (train_yeni["4o"] + train_yeni["5d"]) / 2

train_yeni["1c"] = train_yeni["1_fiyat"] / train_yeni["2_fiyat"]
train_yeni["2c"] = train_yeni["2_fiyat"] / train_yeni["3_fiyat"]
train_yeni["3c"] = train_yeni["3_fiyat"] / train_yeni["4_fiyat"]
train_yeni["4c"] = train_yeni["4_fiyat"] / train_yeni["5_fiyat"]
train_yeni["5c"] = train_yeni["5_fiyat"] / train_yeni["6_fiyat"]

train_yeni["3ay_trend"] = train_yeni["1_fiyat"] - train_yeni["3_fiyat"]
train_yeni["4ay_trend"] = train_yeni["1_fiyat"] - train_yeni["4_fiyat"]
train_yeni["5ay_trend"] = train_yeni["1_fiyat"] - train_yeni["5_fiyat"]
train_yeni["6ay_trend"] = train_yeni["1_fiyat"] - train_yeni["6_fiyat"]

train_yeni["toplam"] = train_yeni.groupby(grup_sutun)["ürün fiyatı"].cumsum()

train_yeni["roll_std_3"] = (
    train_yeni.groupby(grup_sutun)["ürün fiyatı"]
    .transform(lambda x: x.rolling(3).std())
)

ana_veri = train_yeni.dropna().copy()

test_veri = ana_veri[ana_veri["tarih"] == "2023-12-01"].copy()

egitim_veri = ana_veri[ana_veri["tarih"] != "2023-12-01"].copy()

test_veri.drop("ürün fiyatı", axis = 1, inplace = True)

test_yeni = test_csv[["id", "ürün", "ürün üretim yeri", "market", "şehir"]]

test_veri = pd.merge(test_veri, test_yeni, on = ["ürün", "ürün üretim yeri", "market", "şehir"])

ek_sutun = grup_sutun + ["yıl", "ay_sin", "ay_cos", "1_fiyat", "2_fiyat", "3_fiyat", "4_fiyat", "5_fiyat", "6_fiyat",
                         "3ort", "6ort", "1d", "2d", "3d", "4d", "5d", "1c", "2c", "3c", "4c", "5c", "3ay_trend",
                         "4ay_trend", "5ay_trend", "6ay_trend", "2o", "3o", "4o", "5o", "toplam", "roll_std_3"]

encoders = {}
for col in grup_sutun:
    le = LabelEncoder()
    egitim_veri[col] = le.fit_transform(egitim_veri[col])
    test_veri[col] = test_veri[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    encoders[col] = le

X_train = egitim_veri[ek_sutun]
y_train = egitim_veri["ürün fiyatı"]
X_test = test_veri[ek_sutun]

modelxgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    n_jobs=-1,
    random_state=42
)

modelxgb.fit(X_train, y_train)

test_veri["ürün fiyatı"] = modelxgb.predict(X_test)

karsilastirma = ana_veri[ana_veri["tarih"] == "2023-12-01"]

for col in grup_sutun:
    le = encoders[col]
    mask = test_veri[col] != -1
    test_veri.loc[mask, col + "_"] = le.inverse_transform(test_veri.loc[mask, col])
    test_veri.loc[~mask, col + "_"] = "Bilinmiyor"

karsilastirma = karsilastirma[["şehir", "market", "ürün", "ürün üretim yeri", "ürün fiyatı", "1_fiyat", "2_fiyat", "3_fiyat"]]


test_veri_son = test_veri[["id", "şehir_", "market_", "ürün_", "ürün üretim yeri_", "ürün fiyatı"]].copy()

test_veri_son.rename(columns = {"ürün fiyatı": "tahmin fiyat",
                                "şehir_": "şehir",
                                "market_": "market",
                                "ürün_": "ürün",
                                "ürün üretim yeri_": "ürün üretim yeri"},
                     inplace = True)


karsilastirma = pd.merge(karsilastirma, test_veri_son, on = ["şehir", "market", "ürün", "ürün üretim yeri"])

y_true = karsilastirma["ürün fiyatı"]
y_pred = karsilastirma["tahmin fiyat"]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print("RMSE:", rmse)

karsilastirma = karsilastirma.sort_values(by = "id")

karsilastirma[["id","ürün", "tahmin fiyat", "ürün fiyatı", "1_fiyat", "2_fiyat", "3_fiyat"]].to_csv("xgb1.0test.csv", index = False)

xgb.plot_importance(modelxgb, max_num_features=20)
plt.savefig("feature_importance1.0.png", dpi=300, bbox_inches="tight")

#RMSE: 1.3562646230573574