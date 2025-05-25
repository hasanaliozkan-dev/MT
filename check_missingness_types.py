# Veri seti ChatGPT ile oluşturulmuştur.
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression


## Missing Completely at Random (MCaR) Data
data = pd.read_csv("MCaR_data.csv")

print(data)

print(data.isnull().sum())

data["Meslek_Kayıp"] = data["Meslek"].isnull().astype(int)
data["İl_Kayıp"] = data["İl"].isnull().astype(int)


chi2, p, _, _ = chi2_contingency(pd.crosstab(data['Meslek_Kayıp'], data['İl_Kayıp']))

if p < 0.05:
    print("MCaR hipotezi reddedilir")
else:
    print("MCaR hipotezi reddedilmez - değişkenler bağımsızdır")


## Missing at Random (MAR) Data
data = pd.read_csv("MAR_data.csv")

print(data)

print(data.isnull().sum())
data["İl_Kayıp"] = data["İl"].isnull().astype(int)

model = LogisticRegression()

model.fit(data[["Yaş"]], data["İl_Kayıp"])

coefficients = model.coef_

if coefficients[0][0] != 0:
    print("MAR hipotezi reddedilir - İl değişkeni diğer değişkenlerle ilişkili")
else:
    print("MAR hipotezi reddedilmez - İl değişkeni diğer değişkenlerle ilişkili değil")