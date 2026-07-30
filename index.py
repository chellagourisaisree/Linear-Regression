import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("dataset.csv")

print(df.columns)
y=df["Salary"]
x=df[["YearsExperience"]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

print(x_test.shape)
print(x_train.shape)

model=LinearRegression()

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("actual values:",y_test)
print("\n predicted values:",y_pred)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

new_salary = model.predict(pd.DataFrame([[7]], columns=['YearsExperience']))

print("predicted new-salary:",new_salary)