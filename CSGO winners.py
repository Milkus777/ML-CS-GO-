import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier 
from sklearn import tree

data = pd.read_csv('csgo_round_snapshots.csv')

print(data.columns)

print(data.head())
print(data)
print(data.info())

#нул.знач
print(data.isnull().sum())
print('Нулевых значений:', data.isnull().sum().sum())


#тип данных

for column_name, dtype in data.dtypes.items():
    print((column_name),(dtype))




#столбцы только с одним значением
col = data.columns
t=[]
for i in col:
    t.append(data[i].nunique())
temp =[]
for i in range(len(t)):
    if t[i]==1:
        temp.append(i)
        print(i)

data.drop([col[22], col[30], col[37], col[52], col[58], col[60]], axis = 1, inplace = True)
print(col[22])
print(col[30])
print(col[37])
print(col[52])
print(col[58])
print(col[60])
print(data.dtypes)

types = data.dtypes.tolist()
print(data.info())

desc = data.describe()
print(desc)
desc.to_csv('describe.csv') 





#ГРАФИКИ==========================================================================================
plt.figure(figsize=(8,6))
ax = sns.countplot(x = "map", hue = "round_winner", data = data)
ax.set(title = 'Победители в раунде на каждой карте')
plt.show()


plt.figure(figsize=(8,6))
ax = sns.countplot(x = "map", hue = "bomb_planted", data = data)
ax.set(title='Была заложена бомба')
plt.show()

#Самые часто играемые карты
counts = data['map'].value_counts()
total = counts.sum()
percentages = counts / total * 100

plt.bar(counts.index, counts.values)

plt.xticks(rotation=45, ha='right')
plt.xlabel('Map')

plt.ylabel('Count')


for map_name, count, percent in zip(counts.index, counts.values, percentages.values):
    print(f'{map_name}: {percent:.2f}%','/',count)

#соотношение побед каждой из сторон
plt.figure(figsize=(8,6))
ax = sns.barplot(x = data ['round_winner'].unique(), y = data['round_winner'].value_counts())
ax.set(title='Соотношение побед каждой из сторон', xlabel='Side', ylabel='Wins')
plt.show()




    
    
# Соотношение побед и поражений de_inferno

inferno_matches = data[data['map'] == 'de_inferno']

total_rounds = len(inferno_matches)

round_winner_counts = inferno_matches.groupby('round_winner').size()

ct_percentage = round_winner_counts['CT'] / total_rounds * 100
t_percentage = round_winner_counts['T'] / total_rounds * 100

print(f"Процент раундов в которых победили КТ на de_inferno: {ct_percentage:.2f}%")
print(f"Процент раундов в которых победили Т на de_inferno: {t_percentage:.2f}%")

plt.bar(['CT', 'T'], [ct_percentage, t_percentage])
plt.title('Поражения/Победы на de_inferno в процентах')
plt.xlabel('Сторона')
plt.ylabel('Процент выигрыша')
plt.show()

#inferno деньги
t_wins = data[(data['round_winner'] == 'T') & (data['map'] == 'de_inferno')]


money_data = data.groupby('round_winner')[['ct_money', 't_money']].mean()

money_data.plot(kind='bar', stacked=False)
plt.title('Средние кол-во денег за раунд для CT и T')
plt.xlabel('Победители раунда')
plt.ylabel('Среднее кол-во денег')
plt.show()

#Соотношение побед и поражений de_dust2
dust_matches = data[data['map'] == 'de_dust2']

total_rounds = len(dust_matches)

round_winner_counts = dust_matches.groupby('round_winner').size()

ct_percentage = round_winner_counts['CT'] / total_rounds * 100
t_percentage = round_winner_counts['T'] / total_rounds * 100

print(f"Процент раундов в которых победили КТ на de_dust2: {ct_percentage:.2f}%")
print(f"Процент раундов в которых победили Т на de_dust2: {t_percentage:.2f}%")

plt.bar(['CT', 'T'], [ct_percentage, t_percentage])
plt.title('Поражения/Победы на de_dust2 в процентах')
plt.xlabel('Сторона')
plt.ylabel('Процент выигрыша')
plt.show()

#dust деньги
t_wins = data[(data['round_winner'] == 'T') & (data['map'] == 'de_dust2')]


money_data = data.groupby('round_winner')[['ct_money', 't_money']].mean()

money_data.plot(kind='bar', stacked=False)
plt.title('Средние кол-во денег за раунд для CT и T')
plt.xlabel('Победители раунда')
plt.ylabel('Среднее кол-во денег')
plt.show()

#КТ предпочтительные карты 
map_win_pct = data.groupby(['map', 'round_winner'])['round_winner'].count().unstack() / data.groupby('map')['map'].count().values.reshape(-1, 1)
ct_win_pct = map_win_pct['CT']

ct_win_pct = ct_win_pct.sort_values(ascending=False)

plt.figure(figsize=(10,6))
plt.bar(ct_win_pct.index, ct_win_pct)
plt.xticks(rotation=45)
plt.xlabel('Карта')
plt.ylabel('Процент раундов выгранных КТ')
plt.title('Карты, которые способствуют КТ для победы в раунде')
plt.show()


#Т предпочтительные карты
map_win_pct = data.groupby(['map', 'round_winner'])['round_winner'].count().unstack() / data.groupby('map')['map'].count().values.reshape(-1, 1)
t_win_pct = map_win_pct['T']

t_win_pct = t_win_pct.sort_values(ascending=False)

plt.figure(figsize=(10,6))
plt.bar(t_win_pct.index, t_win_pct)
plt.xticks(rotation=45)
plt.xlabel('Карта')
plt.ylabel('Процент раундов выгранных Т')
plt.title('Карты, которые способствуют T для победы в раунде')
plt.show()





encoder = LabelEncoder()
scaler = StandardScaler()

for column in data.columns:
    if len(data[column].unique()) == 1:
        data = data.drop([column], axis = 1)
        
data['round_winner'] = data['round_winner'].replace({'T' : 0, 'CT' : 1})
data['bomb_planted'] = data['bomb_planted'].astype(np.int16)

y = data['round_winner']
data = data.drop('round_winner', axis = 1)

data['map'] = encoder.fit_transform(data['map'])

X = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
#y = data["round_winner"]
#X = data.drop("round_winner", axis=1)
corr = data.corr()
sns.heatmap( data=corr )
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, train_size = 0.8, random_state = 24)



model_1 = LogisticRegression()
model_1.fit(X_train,y_train)
pred_1 = model_1.predict(X_test)
cr1 = classification_report(y_test,pred_1)
print('=======LogisticRegressio=======')
print(cr1)


model_2 = DecisionTreeClassifier()
model_2.fit(X_train,y_train)
pred_2 = model_2.predict(X_test)
cr2 = classification_report(y_test,pred_2)
print('=======DecisionTreeClassifie=======')
print(cr2)


model_3 = RandomForestClassifier()
model_3.fit(X_train,y_train)
pred_3 = model_3.predict(X_test)
cr3 = classification_report(y_test,pred_3)
print('======RandomForestClassifier=====')
print(cr3)


model_4 = MLPClassifier()
model_4.fit(X_train,y_train)
pred_4 = model_4.predict(X_test)
cr4 = classification_report(y_test,pred_4)
print('======MLPClassifier======')
print(cr4)

print('Оптимальная точность для метода случайного леса:', str(accuracy_score(y_test,pred_3)*100) + '%')

