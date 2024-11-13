import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 匯入資料
file_path = '/Users/linjianxun/Desktop/github/vs code/boston/Boston House Prices.csv'
boston_data = pd.read_csv(file_path)

# 資料清理與欄位名稱處理
boston_data.columns = boston_data.iloc[0]
boston_data = boston_data.drop(0).reset_index(drop=True)
boston_data.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston_data = boston_data.apply(pd.to_numeric, errors='coerce')

# 繪製房價分布直方圖 (以 10 為區間，柱狀間有間隔)
plt.figure(figsize=(8, 5))
bins = [0, 10, 20, 30, 40, 50]  # 定義區間
plt.hist(boston_data['MEDV'], bins=bins, color='skyblue', edgecolor='black', rwidth=0.8)  # rwidth 調整間隔
plt.title('Distribution of House Price')
plt.xlabel('House Price Range (thousand dollars)')
plt.ylabel('Count')
plt.xticks(ticks=[5, 15, 25, 35, 45], labels=['0-10', '10-20', '20-30', '30-40', '40-50'])
plt.show()

# 將RM值四捨五入並分析不同RM值的平均房價
boston_data['Rounded_RM'] = boston_data['RM'].round()
average_price_by_rm = boston_data.groupby('Rounded_RM')['MEDV'].mean()

# 繪製不同RM值的房價分布圖
average_price_by_rm.plot(kind='bar', figsize=(10, 6), color='skyblue', edgecolor='black')
plt.title('Distribution of Boston Housing Prices Group by RM')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.xticks(rotation=0)
plt.show()

# 線性回歸模型預測房價
X = boston_data.drop(['MEDV', 'Rounded_RM'], axis=1)  # 移除目標變數和新增欄位
y = boston_data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 評估模型表現
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("均方誤差 (MSE):", mse)
print("R2 分數:", r2)

# 列出最高房價、最低房價、平均房價和中位數房價
max_price = boston_data['MEDV'].max()
min_price = boston_data['MEDV'].min()
mean_price = boston_data['MEDV'].mean()
median_price = boston_data['MEDV'].median()

print("最高房價:", max_price)
print("最低房價:", min_price)
print("平均房價:", mean_price)
print("中位數房價:", median_price)