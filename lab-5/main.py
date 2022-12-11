# Лабораторная работа №5
# Исследование инструментов классификации библиотеки Scikit-learn
# Выполнил студент группы ИСТбд-41
# Калашников Михаил Андреевич
# Классификаторы: 1) KNN KNeighborsClassifier; 2) Древо решений DecisionTreeClassifier; 3) Логистическая регрессия LogisticRegression
# Набор данных: Red Wine Quality https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
#imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
# Метод KNN
def knn_METHOD(KNN_X_Train, KNN_X_Test, KNN_Y_Train, KNN_Y_Test):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(KNN_X_Train, KNN_Y_Train)
    Y_nachal_KNN = knn_model.predict(KNN_X_Test)
# Статистика метрики классификации
# Матрица неточностей
# Оценка модели
    print('Отчет, показывающий основные метрики классификации\n', classification_report(KNN_Y_Test,
                                                                                        Y_nachal_KNN, zero_division=0),
          '\nМатрица неточностей для оценки точности классификации', confusion_matrix(KNN_Y_Test, Y_nachal_KNN),
          '\nОценка модели KNN: оценка точности ', accuracy_score(KNN_Y_Test, Y_nachal_KNN))
# Рисует основные признаки классификации
    v1, orig_v1 = plt.subplots(3, 1, figsize=(12, 12))
    orig_v1[0].scatter(KNN_X_Test[:, 5], KNN_X_Test[:, 6], c=Y_nachal_KNN)
    orig_v1[0].set_title('Зависимость свободного диоксида серы от общего диоксида серы')
    orig_v1[1].scatter(KNN_X_Test[:, 5], KNN_X_Test[:, 10], c=Y_nachal_KNN)
    orig_v1[1].set_title('Зависимость свободного диоксида серы от алкоголя')
    orig_v1[2].scatter(KNN_X_Test[:, 6], KNN_X_Test[:, 10], c=Y_nachal_KNN)
    orig_v1[2].set_title('Зависимость общего содержания диоксида серы от содержания алкоголя')
    plt.show()
# Рисует тестовые значения
    Graph_V1 = pd.crosstab(index=KNN_Y_Test, columns='% observations')
    plt.pie(Graph_V1['% observations'], labels=Graph_V1['% observations'].index, autopct='%.0f%%')
    plt.title('Массив тестовых значений')
    plt.show()
# Рисует предказанные значения
    Graph_V2 = pd.crosstab(index=Y_nachal_KNN, columns='% observations')
    plt.pie(Graph_V2['% observations'], labels=Graph_V2['% observations'].index, autopct='%.0f%%')
    plt.title('Массив прогнозируемых значений')
    plt.show()
# Совпадение тестовых данных и предсказания
    plt.plot(KNN_Y_Test, label='Тестовое')
    plt.plot(Y_nachal_KNN, label='Предсказанное')
    plt.legend()
    plt.show()
    Interval_Ka4estva = max(KNN_Y_Test) - min(KNN_Y_Test)
    plt.title('Качество от{} до {}'.format(min(KNN_Y_Test), max(KNN_Y_Test)))
    plt.hist(KNN_Y_Test, bins=Interval_Ka4estva)
    plt.show()
# Метод Древа решений
def Decision_Tree_METHOD(X_train_t, X_test_t, y_train_t, y_test_t):
    Derevo = DecisionTreeClassifier(max_depth=12)
    Derevo.fit(X_train_t, y_train_t)
    y_nachalnaya_T = Derevo.predict(X_test_t)
# Статистика классификации
# Метрика классификации
    print('Отчет, показывающий основные метрики классификации', classification_report(y_test_t, y_nachalnaya_T, zero_division=0),
          '\nМатрица неточностей для оценки точности классификации', confusion_matrix(y_test_t, y_nachalnaya_T))
# Рисует тестовые занчения
    Graph_v1 = pd.crosstab(index=y_test_t, columns='% observations')
    plt.pie(Graph_v1['% observations'], labels=Graph_v1['% observations'].index, autopct='%.0f%%')
    plt.title('Массив тестовых значений')
    plt.show()
# Рисует предсказанные значения
    Graph_v2 = pd.crosstab(index=y_nachalnaya_T, columns='% observations')
    plt.pie(Graph_v2['% observations'], labels=Graph_v2['% observations'].index, autopct='%.0f%%')
    plt.title('Массив прогнозируемых значений')
    plt.show()
# Совпадения значений выше
# совпадение тестовых и предсказанных
    plt.plot(y_test_t, label='Тестовое')
    plt.plot(y_nachalnaya_T, label='Предсказанное')
    plt.legend()
    plt.show()
#Качество данных
    KA4estvo_datas = max(y_test_t) - min(y_test_t)
    plt.title('Качество от {} до {}'.format(min(y_test_t), max(y_test_t)))
    plt.hist(y_test_t, bins=KA4estvo_datas)
    plt.show()
# Метод Логистической регрессии
def LogisticRegression_METHOD(X_train, y_train, X_test, y_test):
    Logistic_v1 = Normalizer().fit_transform(X_train)
    Logistic_v2 = Normalizer().fit_transform(X_test)
    model = LogisticRegression()
    model.fit(Logistic_v1, y_train)
    Vozmozhnaya_model = model.predict(Logistic_v2)
# Вывод метрик классификации
# Вывод матрицы ошибок для оценки точности
# ВЫвод точности метода логистической регрессии
    print('Основные метрики классификации\n', classification_report(y_test, Vozmozhnaya_model, zero_division=0),
          '\nМатрица ошибок для оценки точности', confusion_matrix(y_test, Vozmozhnaya_model), '\nСчет X-train с Y-train: ',
          model.score(Logistic_v1, y_train), '\nСчет X-test  с Y-test  : ', model.score(Logistic_v2, y_test),
          '\nТочность метода логистической регрессии ', accuracy_score(y_test, Vozmozhnaya_model))
    point=300
# Рисует признаки
    v1, orig_v1 = plt.subplots(2, 1, figsize=(12, 12))
    orig_v1[0].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=Vozmozhnaya_model[:point])
    orig_v1[0].set_title('Прогнозироваемая зависимость качества красного вина')
    orig_v1[1].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=y_test[:point])
    orig_v1[1].set_title('Фактическая зависимость качества красного вина')
    plt.show()
# Набор массива данных Red Wine Quality https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
dataset = pd.read_csv('winequality-red.csv')
# Содержимое dataset (0 - 5)
print('Просмотр пяти рядов\n', dataset.head())
# Вывод размерности
print('Формирование набора данных : ', dataset.shape)
# Количество пропусков
print('Количество отсутствующих значений во всем наборе данных\n', dataset.isnull().sum())
# Описательная статистика
print('Статистика\n', dataset.describe().round(2))
# Удаление столбца
dataset.drop(columns='Id', inplace=True)
# Уникальные значения качества (quality)
print('\nКачество ценности : ', dataset['quality'].unique())
# Группирование по значениям quality
ave_qu = dataset.groupby('quality').mean()
print('Группировка по качеству\n', ave_qu.round(2))
# Формирования набора данных
X = dataset.drop(columns='quality').values
y = dataset['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# Вывод размера массива
# Вывод переменных
# Вывод метода KNN
# Вывод метода Древа решений
# Вывод метода Логистическая регрессия
print('\nРазмер массивов\nX train : ', X_train.shape, '\nX test  : ', X_test.shape, '\nY train : ', y_train.shape,
      '\nY test  : ', y_test.shape, '\nМетод K-ближайших соседей\n', knn_METHOD(X_train, X_test, y_train, y_test),
      '\nМетод Классификатор дерева решений\n', Decision_Tree_METHOD(X_train, X_test, y_train, y_test),
      '\nЛогистическая регрессия: \n', LogisticRegression_METHOD(X_train, y_train, X_test, y_test))

