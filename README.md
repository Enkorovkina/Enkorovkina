#Загрузим необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Выгрузим данные и преобразуем в рабочий датасет
df = pd.read_csv('telecom_users.csv')

#Посмотрим характеристики полученного датасета
#Информация про датасет
df.info()

#Описательные характеристики (статистика) по столбцам данного датасета
df.describe()

#Размеры датасета
df.shape
#проверим наличие пустых значений в строках
df.isnull()
#проссумируем кол-во пустых значений в строках
df.isnull().sum()

#преобразуем данные object - int
df_work[["InternetService", 'OnlineSecurity', 
         'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 
         'StreamingMovies', 'Contract', 
         'PaymentMethod']] = df_work[['InternetService',
                                      'OnlineSecurity', 'OnlineBackup',
                                      'DeviceProtection', 'TechSupport', 
                                      'StreamingTV', 'StreamingMovies', 'Contract', 
                                      'PaymentMethod']].astype('int')
#взглянем на наши данные
df_test = df_work['Churn']
df_train = df_work.drop(columns = 'Churn')

df_train1 = df_work.iloc[:, :4]
df_train2 = df_work.iloc[:, 4:9]
df_train3 = df_work.iloc[:, 9:13]
df_train4 = df_work.iloc[:, 13:19]

df_trains_list = [df_train1, df_train2, df_train3, df_train4]

for train in df_trains_list:
    scatter_mtrx = pd.plotting.scatter_matrix(train, c = df_test, 
                                              figsize = (15, 15), marker = 'o', s = 40)
#проведем корреляционный анализ и попробуем найти взаимосвязи между признаками
#Построим матрицу попарных корреляций для отобранных данных (проверим линейные зависимости м/у данными)
corr_m = df_work.corr()

#Визуализируем полученную матрицу
plt.figure(figsize = (15, 15))
sns.heatmap(corr_m, annot = True)
plt.show()

#Проверим несколько гипотез: 1 Самое популярное среди пользователей услуга - подключение к интернету
counts_internetservice_dsl = df_work[df_work['InternetService'] == 0]['gender'].count()
counts_internetservice_fiber = df_work[df_work['InternetService'] == 1]['gender'].count()
counts_internetservice = counts_internetservice_dsl + counts_internetservice_fiber

name_columns = ['PhoneService', 'OnlineSecurity', 
                  'OnlineBackup', 'DeviceProtection', 'TechSupport',
                 'StreamingTV', 'StreamingMovies']
counts = list()
for name_column in name_columns:
    count = df_work[df_work[name_column] == 0]['gender'].count()
    print('Количество человек, использовавших услугу {} равно {}'.format(name_column, count))
    counts.append(count)
print('Количество человек, использовавших услугу InternetService DSL равно {}'.format(counts_internetservice_dsl))
print('Количество человек, использовавших услугу InternetService Fiber optic равно {}'.format(counts_internetservice_fiber))
print('Общее количество человек, использовавших услугу InternetService равно {}'.format(counts_internetservice))  


#Следующая гипотеза: 2 Среди большого количества пользователей молодые мужчины
df_genders = df_work[['gender', 'SeniorCitizen']].groupby(['gender', 'SeniorCitizen'])['gender'].count()

#Еще одна гипотеза: 3 Большой отток клиентов у тех, кто пользуется услугами компании малое кол-во времени
df_churn = df_work[['tenure', 'Churn']].groupby('Churn')['tenure'].median()

#Последняя гипотеза, которую попытаемся проверить: 4 Среди тех пользователей, которые хотят отказаться от услуг компании, много тех, кто не доволен ценами на услуги связи (слишком большие затраты)
df_expenses = df_work[['gender', 'MonthlyCharges', 'Churn']].groupby(['gender', 'Churn'])['MonthlyCharges'].median()

#Построение модели для прогнозирования оттока
def stacking(models, meta_alg, data_train, targets_train, data_test, targets_test=None, random_state=None, test_size=None, cv=5):
    if test_size is None:
        meta_mtrx = np.empty((data_train.shape[0], len(models)))
        for n, model in enumerate(models):
            meta_mtrx[:, n] = cross_val_predict(model, data_train, targets_train, cv=cv, method = 'predict')
            fit_model = model.fit(data_train, targets_train)
        
        meta_model = meta_alg.fit(meta_mtrx, targets_train)
        
        meta_mtrx_test = np.empty((x_test.shape[0], len(models)))
        for n, model in enumerate(models):
            meta_mtrx_test[:, n] = fit_model.predict(data_test)
            
        meta_predict = meta.predict(meta_mtrx_test)
        
        print(meta_predict)
        
        if targets_test is not None:
            print(f'Stacking AUC: {roc_auc_score(targets_test, meta_predict)}')
            print(f'Accuracy_score: {accuracy_score(targets_test, meta_predict)}')
            print(f'Precision_score: {precision_score(targets_test, meta_predict)}')
        
    elif test_size > 0 and test_size < 1:
        train, valid, train_true, valid_true = train_test_split(data_train, 
                                                        targets_train,
                                                        train_size=test_size,
                                                        random_state=random_state)
        
        meta_mtrx = np.empty((valid.shape[0], len(models)))
        for n, model in enumerate(models):
            fit_model = model.fit(train, train_true)
            meta_mtrx[:, n] = model.predict(valid)
#             predicted = model.predict(x_test)

        meta_model = meta_alg.fit(meta_mtrx, valid_true)
        meta_mtrx_test = np.empty((x_test.shape[0], len(models)))
        for n, model in enumerate(models):
            meta_mtrx_test[:, n] = model.predict(data_test)
        
        meta_predict = meta_alg.predict(meta_mtrx_test)
        print(meta_predict)
        
        if targets_test is not None:
            print(f'Stacking AUC: {roc_auc_score(targets_test, meta_predict)}')
            print(f'Accuracy_score: {accuracy_score(targets_test, meta_predict)}')
            print(f'Precision_score: {precision_score(targets_test, meta_predict)}')
        
    else:
        raise ValueError("test_size must be between 0 and 1")


#для начала подготовим данные для дальнейшего обучения моделей
df_train = df_work.drop(columns = ['Churn', 'TotalCharges'])
df_test = df_work['Churn']

#импортируем необходимые библиотеки
from sklearn.model_selection import train_test_split

#разделим данные на тренировочные и тестовые
x_train, x_test, y_train, y_test = train_test_split(df_train, df_test, test_size = 0.3, stratify = df_test)

#определимся с моделями машинного обучения, дополнительно воспользуемся stacking
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

#параметры для моделей
gbc_params = {'learning_rate': np.arange(0.1, 0.6, 0.1)} # GradientBoostingClassifier

rfc_params = {'n_estimators': range(10, 100, 20), # RandomForestClassifier
              'min_samples_leaf': range(1, 5)}

svc_params = {'kernel': ['linear', 'rbf'], # SVC
              'C': np.arange(0.1, 1, 0.2)}

lr_params = {'C': np.arange(0.5, 1, 0.1)}


#обучим модели для начала по отдельности и проверим их на точность и качество упорядоченности алгоритмом 

meta = XGBClassifier(n_estimators=40)

lr = LogisticRegression(C=10, random_state=17)
lr_gs = GridSearchCV(lr, lr_params, cv = 5, scoring = 'roc_auc')
lr_model = lr_gs.fit(x_train, y_train)

rf1 = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=17)
rf1_gs = GridSearchCV(rf1, rfc_params, cv = 5, scoring = 'roc_auc')
rf1_model = rf1_gs.fit(x_train, y_train)


rf2 = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=17)
rf2_gs = GridSearchCV(rf2, rfc_params, cv = 5, scoring = 'roc_auc')
rf2_model = rf2_gs.fit(x_train, y_train)

rf3 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=17)
rf3_gs = GridSearchCV(rf3, rfc_params, cv = 5, scoring = 'roc_auc')
rf3_model = rf3_gs.fit(x_train, y_train)

gb1 = GradientBoostingClassifier(learning_rate=0.1, random_state=17)
gb1_gs = GridSearchCV(gb1, gbc_params, cv = 5, scoring = 'roc_auc')
gb1_model = gb1_gs.fit(x_train, y_train)

gb2 = GradientBoostingClassifier(learning_rate=0.4, random_state=17)
gb2_gs = GridSearchCV(gb2, gbc_params, cv = 5, scoring = 'roc_auc')
gb2_model = gb2_gs.fit(x_train, y_train)

svc = SVC(degree=3, random_state=17)
svc_gs = GridSearchCV(svc, svc_params, cv = 5, scoring = 'roc_auc')
svc_model = svc_gs.fit(x_train, y_train)

models = [lr_gs, rf1_gs, rf2_gs, rf3_gs, gb1_gs, gb2_gs, svc_gs]

#Теперь сравним качества моделей по параметрам: ROC_AUC score, accuracy_score, presicion_score
from sklearn.metrics import precision_score
#дополнительно обучим стэкингу
stacking(models, meta, x_train, y_train, x_test, targets_test=y_test, random_state=None, test_size=0.3, cv=5)

# models_fit = [knn1_model, knn2_model, lr_model, rf1_model, rf2_model, rf3_model, gb1_model, gb2_model, svc_model]

models_fit = [lr_model, rf1_model, rf2_model, rf3_model, gb1_model, gb2_model, svc_model]
for n, model_fit in enumerate(models_fit):
    model_predicted = model_fit.predict(x_test)
    print('ROC_AUC score for {}: {}'.format(models[n], roc_auc_score(y_test, model_predicted)))
    print('Precision_score for {}: {}'.format(models[n], precision_score(y_test, model_predicted)))
    print('\n')
 #Самый лучший результат получился у модели SVC (метода опорных векторов), поэтому она лучше всего подходит для прогнозирования оттока пользователей.
