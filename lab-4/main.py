# Лабораторная работа №4
# Метод к – ближайших соседей (k-NN)
# Выполнил студент группы ИСТбд-41
# Калашников Михаил Андреевич
#imports
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pylab
# Задание класcификаций для продуктов
Parametrs = ['Продукт', 'Сладость', 'Хруст', 'Класс']
# Создание списка с продуктами и их параметрами
Eda1 = [['Яблоко', '7', '7', '0'],
        ['Салат', '2', '5', '1'],
        ['Бекон', '1', '2', '2'],
        ['Банан', '9', '1', '0'],
        ['Орехи', '1', '5', '2'],
        ['Рыба', '1', '1', '2'],
        ['Сыр', '1', '1', '2'],
        ['Виноград', '8', '1', '0'],
        ['Морковь', '2', '8', '1'],
        ['Апельсин', '6', '1', '0'],
        ['Мандарин', '9', '1', '0'],
        ['Арбуз', '3', '7', '1'],
        ['Тыква', '3', '1', '2'],
        ['Репка', '5', '3', '0'],
        ['Подсолнух', '1', '7', '1']]
#Создание разметки
def Razmetka(x1, y1, x2, y2):
    return (abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2) ** 0.5
# Реализация классификатора по методу k-NN
def knnKLASS(products, OriginalSIZE, pars_window, Resultats):
# Индексация переменных
    Materials = np.array(products)
    Size1 = len(Materials) - OriginalSIZE
    Distanses = np.zeros((Size1, OriginalSIZE))
    All_class = [0] * Size1
# Реализация классификатора №1
    for i in range(Size1):
        for j in range(OriginalSIZE):
            distans = Razmetka(int(Materials[OriginalSIZE + i][1]),
                            int(Materials[OriginalSIZE + i][2]), int(Materials[j + 1][1]),
                            int(Materials[j + 1][2]))
            Distanses[i][j] = distans if distans < pars_window else 1000
# Реализация классификатора №2
    for i in range(Size1):
        print(str(i) + ') ' + Materials[OriginalSIZE + i][0])
        Massa = [0] * products.iloc[:]['Класс'].nunique()
        neighbor = np.sum(Distanses[i] != 1000)
        for j in range(neighbor + 1):
            min = Distanses[i].argmin()
            Massa[int(Materials[min + 1][3])] += ((neighbor - j + 1) / neighbor)
            Distanses[i][min] = 1000
        All_class[i] = np.array(Massa).argmax()
# Проверка классов (Предположительный и Оригинальный)
# !Должны все совпадать!
# Сообщает о правильности проведения эксперемента!
        print('Предположительный класс: ', All_class[i], 'Оригинальный класс: ', Materials[OriginalSIZE + i][3])
        if int(All_class[i]) != int(Materials[OriginalSIZE + i][3]):
            print('нет совпадения')
        else:
            print('совпадение')
            Resultats += 1
    print(All_class)
    print('Совпадений ручного классификатора: ', str(Resultats))
    return All_class
# Реализация классификатора методом k-NN библиотеки sklearn
def sklearnKNN(values, k, y):
    x_train, x_test, y_train, y_test = train_test_split(values, y, test_size=0.3, shuffle=False, stratify=None)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return x_train, x_test, y_train, y_test, predictions
def Demonstration(data, window, colors):
    x = data['Сладость']
    y = data['Хруст']
    colors_for_Demonstration = [colors[str(i)] for i in data['Класс']]
    pylab.subplot(2, 1, window)
    plt.scatter(x, y, c=colors_for_Demonstration)
    plt.xlabel('Сладость')
    plt.ylabel('Хруст')
# запись данных в файл
with open('File_s_datas.csv', 'w', encoding='utf8') as file:
    writer = csv.writer(file, lineterminator="\r")
    writer.writerow(Parametrs)
    for row in Eda1:
        writer.writerow(row)
# чтение данных из файла
Materials = pd.read_csv('File_s_datas.csv')
train = 10
k = 4
X_for_knn1 = Materials.iloc[:, 1:3].values
Y_for_knn1 = Materials.iloc[:, 3].values
All_class = Materials[:train]['Класс']
# Использование KNN для данных номер 1!
knn_result = pd.Series(knnKLASS(Materials, train, 4, 0))
All_class = pd.concat([All_class, knn_result])
color_for_knn = {'0': 'green', '1': 'gray', '2': 'pink'}
Demonstration(Materials, 1, color_for_knn)
colors_spinner = [color_for_knn[str(i)] for i in All_class]
plt.show()
# ИСпользование knn1 с библиотекой sklearn
X_train, X_test, y_train, y_test, predictions = sklearnKNN(X_for_knn1, k, Y_for_knn1)
# вывод статистики качества прогнозирования
print('Cтатистика качества прогнозирования\nс использованием scikit-learn', classification_report(y_test, predictions),
      confusion_matrix(y_test, predictions))
Eda2 = [['Яблоко', '7', '7', '0'],
            ['Салат', '2', '5', '1'],
            ['Арбуз', '10', '8', '3'],
            ['Яйца', '1', '2', '2'],
            ['Банан', '9', '1', '0'],
            ['Фундук', '10', '9', '3'],
            ['Орехи', '1', '5', '2'],
            ['Рыба', '1', '1', '2'],
            ['Сыр', '1', '1', '2'],
            ['Виноград', '8', '1', '0'],
            ['Шипучка', '8', '10', '3'],
            ['Морковь', '2', '8', '1'],
            ['Апельсин', '6', '1', '0'],
            ['конфета', '6', '9', '3'],
            ['Малина', '9', '1', '0'],
            ['Капуста', '3', '7', '1'],
            ['Халва', '6', '7', '3'],
            ['Говядина', '3', '1', '2'],
            ['Свиника', '5', '3', '0'],
            ['Свекла', '1', '7', '1'],
            ['Мармелад', '7', '8', '3']]
# запись новых данных во второй файл
with open('File_s_newDatas.csv', 'w', encoding='utf8') as f:
    writer = csv.writer(f, lineterminator="\r")
    writer.writerow(Parametrs)
    for row in Eda2:
        writer.writerow(row)
# загрузка новых данных
newDatas = pd.read_csv('File_s_newDatas.csv')
train = 14
# Использование KNN для данных номер 2!
knn_result = pd.Series(knnKLASS(newDatas, train, k, 0))
oldDatas = pd.concat([All_class, knn_result])
X_for_knn2 = newDatas.iloc[:, 1:3].values
Y_for_knn2 = newDatas.iloc[:, 3].values
# Использование knn2 с библиотекой sklearn
X_train, X_test, y_train, y_test, predictions = sklearnKNN(X_for_knn2, k, Y_for_knn2)
# вывод статистики качества прогнозирования
print('Cтатистика качества прогнозирования\nс использованием scikit-learn', classification_report(y_test, predictions),
      confusion_matrix(y_test, predictions))
color_for_knn = {'0': 'green', '1': 'gray', '2': 'pink', '3': 'blue'}
Demonstration(newDatas, 1, color_for_knn)
plt.show()