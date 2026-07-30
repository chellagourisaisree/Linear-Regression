import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("HousingData.csv")

df= df.fillna(df.median())

y=df["MEDV"]
x=df[["RM","AGE"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("mse value:",mse)
print("r2 value:",r2)

new_house = pd.DataFrame({
    "RM": [6.5],
    "AGE": [40]
})

predicted_price = model.predict(new_house)

print("predicted price",predicted_price)