import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


# لود کردن اطلاعات و دیتاهای داده شده با استفاده از Pandas
column_names = []
with open(r'DataSet/adult.names', "r") as f:
    lines = f.readlines()
    for line in lines:
        if ":" in line:
            column_name = line.split(":")[0].strip()
            column_names.append(column_name)

data = pd.read_csv(r'DataSet/adult.data', names=column_names, sep=',\s*', engine='python')

print(data.head())

# مرتب سازی کشور ها بر اساس ساعات کاری در هفته
top_10_countries = data.groupby('native-country')['hours-per-week'].mean().sort_values(ascending=False).head(10)

data = pd.get_dummies(data, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])

# نمودار داده های ساعات کاری هفتگی کشورها
plt.figure(figsize=(10, 6))
top_10_countries.plot(kind='bar')
plt.title('Top 10 Countries with Highest Average Hours per Week')
plt.xlabel('Country')
plt.ylabel('Average Hours per Week')
plt.show()

k50_greater = data[data['income'] == '>50K'].sample(7000, random_state=42)
k50_less_equal = data[data['income'] == '<=50K'].sample(7000, random_state=42)
balanced_data = pd.concat([k50_greater, k50_less_equal])

X = balanced_data.drop('income', axis=1)
y = balanced_data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# مقدار همسایگی k=1  را که در تمرین نوضیح داده شده بود گذاشته شده میتواند تغییر کند
k = 1
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# محاسبه میزان دقت شبکه
start_time = time.time()
accuracy = accuracy_score(y_test, y_pred)
end_time = time.time()

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Execution Time: {end_time - start_time:.2f} seconds')
