
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine

# connection credentials
db_server = "vgbi707.database.windows.net"
db_name = "vgbi2025nem"
db_username = "vgbiH17"
db_password = "JfSJapDatzxCCi0s"

# ODBC Driver
odbc_driver = "{ODBC Driver 18 for SQL SERVER}"

# connection strings
connection_string_sqlalchemy = (
"mssql+pyodbc://{username}:{password}@{db_server}/{db_name}"
    "?driver=ODBC+Driver+18+for+SQL+Server"
).format(
    username=db_username,
    password=db_password,
    db_server=db_server,
    db_name=db_name
)

# create a connection to the database
engine = create_engine(connection_string_sqlalchemy)

# construct a squery and read from the database
query = "select idCalendar, unitsSold from h17.factSales"
df = pd.read_sql_query(query, engine)
# print(df.head(10))

# check for null values in dataset
df.info()

# drop null values
df = df.dropna()
df.info()

# convert date from object datatype to dateTime data type
df['idCalendar'] = pd.to_datetime(df['idCalendar'])
df.info()

# converting date to month period, and then sum the number of items in each month
df['idCalendar'] = df['idCalendar'].dt.to_period("M")
monthly_sales = df.groupby('idCalendar').sum().reset_index()

# convert the resulting date to timestamp daatatype
monthly_sales['idCalendar'] = monthly_sales['idCalendar'].dt.to_timestamp()
print("MONTHLY: ", monthly_sales)

# visualization
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['idCalendar'], monthly_sales['unitsSold'], color='blue')
plt.xlabel("Date")
plt.xlabel("Sales")
plt.title("Montly Customer Sales")
plt.show()

# calculate the difference in sales
monthly_sales['unitsSold_diff'] = monthly_sales['unitsSold'].diff()
monthly_sales = monthly_sales.dropna()
# print(monthly_sales)

# plot sales difference
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['idCalendar'], monthly_sales['unitsSold_diff'], color='red')
plt.xlabel("Date")
plt.xlabel("Sales")
plt.title("Montly Customer Sales Difference")
plt.show()

# prepare the supervised data
supervised_data = monthly_sales.drop(['idCalendar', 'unitsSold'], axis=1)

for i in range(1, 13):
    column_name = 'month_' + str(i)
    supervised_data[column_name] = supervised_data['unitsSold_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
print(supervised_data)

# split the data into train and test to train the model
train_data = supervised_data.iloc[:-7]
test_data = supervised_data.iloc[-7:]
print("Train Data Shape: ", train_data.shape)
print("Test Data Shape: ", test_data.shape)

# normalize the data
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

# split the data into input and output
x_train, y_train = train_data[:,1:], train_data[:,0:1]
x_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
print("X_train Shape: ", x_train.shape)
print("Y_train Shape: ", y_train.shape)
print("X_test Shape: ", x_test.shape)
print("Y_test Shape: ", y_test.shape)

# make prediction data frame to merge the predictate sale prices of all trained algs
sales_dates = monthly_sales['idCalendar'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
print(predict_df)

# get the actual sales
actual_sales = monthly_sales['unitsSold'][-13:].to_list()

# LINEAR REGRESSION MODEL
linreg_model = LinearRegression()
linreg_model.fit(x_train, y_train)
linreg_pred = linreg_model.predict(x_test)

# set matrix - contains the input features of the test data and also the predicted output
linreg_pred = linreg_pred.reshape(-1,1)
linreg_pred_test_set = np.concatenate([linreg_pred,x_test], axis=1)
linreg_pred_test_set = scaler.inverse_transform(linreg_pred_test_set)

# calculate the predicted sales
result_list = []
for index in range(0, len(linreg_pred_test_set)):
    result_list.append(linreg_pred_test_set[index][0] + actual_sales[index])
linreg_pred_series = pd.Series(result_list,name='linreg_pred')
predict_df = predict_df.merge(linreg_pred_series, left_index=True, right_index=True)
print("Result of Linear Regression: ", result_list)


linreg_rmse = np.sqrt(mean_squared_error(predict_df['linreg_pred'], monthly_sales['unitsSold'][-7:]))
linreg_mae = mean_absolute_error(predict_df['linreg_pred'], monthly_sales['unitsSold'][-7:])
linreg_r2 = r2_score(predict_df['linreg_pred'], monthly_sales['unitsSold'][-7:])
print('Linear Regression RMSE: ', linreg_rmse)
print('Linear Regression MAE: ', linreg_mae)
print('Linear Regression R2 Score: ', linreg_r2)

# plot the results
# plt.figure(figsize=(15,5))
# plt.plot(monthly_sales['idCalendar'], monthly_sales['unitsSold'])
# plt.plot(predict_df['idCalendar'], predict_df['linreg_pred'])
# plt.title("Customer Sales Forecast using Linear Regression")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()

# RANDOM FOREST REGRESSOR MODEL
rf_model = RandomForestRegressor(n_estimators=100, max_depth=20)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

# revert original scale
rf_pred = rf_pred.reshape(-1,1)
rf_pred_test_set = np.concatenate([rf_pred,x_test], axis=1)
rf_pred_test_set = scaler.inverse_transform(rf_pred_test_set)

# merge the predicted sales
result_list = []
for index in range(0, len(rf_pred_test_set)):
    result_list.append(rf_pred_test_set[index][0] + actual_sales[index])
rf_pred_series = pd.Series(result_list, name='rf_pred')
predict_df = predict_df.merge(rf_pred_series, left_index=True, right_index=True)
print(predict_df)

# calculate the metrics
rf_rmse = np.sqrt(mean_squared_error(predict_df['rf_pred'], monthly_sales['unitsSold'][-7:]))
rf_mae = mean_absolute_error(predict_df['rf_pred'], monthly_sales['unitsSold'][-7:])
rf_r2 = r2_score(predict_df['rf_pred'], monthly_sales['unitsSold'][-7:])
print('Random Forest RMSE: ', rf_rmse)
print('Random Forest MAE: ', rf_mae)
print('Random Forest R2 Score: ', rf_r2)

# plot the results
# plt.figure(figsize=(15,5))
# plt.plot(monthly_sales['idCalendar'], monthly_sales['unitsSold'])
# plt.plot(predict_df['idCalendar'], predict_df['rf_pred'])
# plt.title("Customer Sales Forecast using Random Forest")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()


# XGBoost REGRESSOR MODEL
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror')
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)

# convert back to original scale
xgb_pred = xgb_pred.reshape(-1,1)
xgb_pred_test_set = np.concatenate([xgb_pred,x_test], axis=1)
xgb_pred_test_set = scaler.inverse_transform(xgb_pred_test_set)

result_list = []
for index in range(0, len(xgb_pred_test_set)):
    result_list.append(xgb_pred_test_set[index][0] + actual_sales[index])
xgb_pred_series = pd.Series(result_list, name='xgb_pred')
predict_df = predict_df.merge(xgb_pred_series, left_index=True, right_index=True)

xgb_rmse = np.sqrt(mean_squared_error(predict_df['xgb_pred'], monthly_sales['unitsSold'][-7:]))
xgb_mae = mean_absolute_error(predict_df['xgb_pred'], monthly_sales['unitsSold'][-7:])
xgb_r2 = r2_score(predict_df['xgb_pred'], monthly_sales['unitsSold'][-7:])
print('XG Boost RMSE: ', xgb_rmse)
print('XG Boost MAE: ', xgb_mae)
print('XG Boost R2 Score: ', xgb_r2)

# plot the results
# plt.figure(figsize=(15,5))
# plt.plot(monthly_sales['idCalendar'], monthly_sales['unitsSold'])
# plt.plot(predict_df['idCalendar'], predict_df['xgb_pred'])
# plt.title("Customer Sales Forecast using XGBoost")
# plt.xlabel("Date")
# plt.ylabel("Sales")
# plt.legend(["Original Sales", "Predicted Sales"])
# plt.show()
