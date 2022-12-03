import numpy as np
import pandas as pd
import csv
import random
from datetime import date
import matplotlib.pyplot as plt

def table():
    with open('Maleling.txt', 'r', encoding='utf-8') as f:
        Male = [i.rstrip() for i in f]

    with open('Femaling.txt', 'r', encoding='utf-8') as f:
        Female = [i.rstrip() for i in f]

    OligarchList = {
        "Ульяновская область": ["Руб Миллионер", "Руб Миллиардер",
                                "Долл Миллионер", "Долл Миллиардер"],
        "Республика Татарстан": ["Руб Миллионер", "Руб Миллиардер",
                                 "Долл Миллионер", "Долл Миллиардер"],
        "Казанская область": ["Руб Миллионер", "Руб Миллиардер",
                              "Долл Миллионер", "Долл Миллиардер"],
    }
    # Заполнение CSV
    with open("Oligarchs_spinner.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            (
                "Табельный номер", "Фамилия.И.О.", "Пол", "Год рождения", "Год начала работы",
                "Подразделение экономической деятельности", "Должность", "Состояние", "Колличество предприятий"
            )
        )
    # Генерация Имени и Отчества
    def First_and_middle_name():
        return (random.choice('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ') + ". " + random.choice(
            'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯ') + ".")
    # Генерация таблицы олигархов
    for Employees in range(random.randint(1000, 1111)):
        Oligarchs = random.choice(list(OligarchList))
        sex = random.choice(["Мужчина", "Женщина"])
        Birthday = random.randint(1950, 2010)
        StartingYear = min(date.today().year, Birthday + random.randint(12, 99))
        MoneyCondition = random.randrange(1000000, 1000000000, 1000000)
        EnterprisesAmount = random.randint(0, 15)

        if (sex == "Мужчина"):
            Name = Male[random.randint(0, len(Male) - 1)]
        else:
            Name = Female[random.randint(0, 111)]
        # Заполнение CSV
        with open("Oligarchs_spinner.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    Employees + 1,
                    (Name + " " + First_and_middle_name()), sex, Birthday, StartingYear,
                    Oligarchs, random.choice(OligarchList[Oligarchs]),
                    MoneyCondition, EnterprisesAmount
                ]
            )
# Работа с помощью NUMPY
def numpy():
    with open("Oligarchs_spinner.csv") as csvfile:
        metadata = [list(row) for row in csv.reader(csvfile)]
    data = np.array(metadata)
    sexs = data[:, 2]
    Oligarchs = data[:, 5]
    MoneyCondition = data[1:, 7].astype("int32")
    StartingYear = data[1:, 4].astype("int32")
    count_MoneyCondition = np.bincount(MoneyCondition)
    count_StartingYear = np.bincount(StartingYear)
    # Вывод данных в консоль(Numpy)
    print("Задание по NUMPY \n ---------------- \n Статистика по миллионерам и миллиардерам \n "
          "Количество олигархов всего:", np.count_nonzero(MoneyCondition), "\nПодъём экономики был в",
          np.argmax(count_StartingYear), "году. \nТак-как колличество новых олигархов больше в этом году"
          "\n----------------" "\nДоля олигархов мужского пола:", round(np.sum(sexs == "Мужчина") / np.size(sexs), 3),
          "\nДоля олигархов женского пола:", round(np.sum(sexs == "Женщина") / np.size(sexs), 3), "\n----------------"
          "\nДенежное состояние олигархов \n Максимальное состояние олигархов:", np.max(MoneyCondition),
          "\nМинимальная состояние олигархов:", np.min(MoneyCondition), "\nСреднее денежное состояние олигархов:",
          round(np.average(MoneyCondition), 3), "\nДисперсия денежного состояния олигархов:",
          round(np.var(MoneyCondition), 3), "\nСт. откл. состояния олигархов:", round(np.std(MoneyCondition), 3),
          "\nМедиана денежного состояния олигархов:",np.median(MoneyCondition), "\nМода денежного состояния олигархов:",
          np.argmax(count_MoneyCondition), "\n----------------\nГруппировка олигархов по регионам"
                                           "\nКоличество олигархов в Ульяновске:",
          np.count_nonzero(Oligarchs == "Ульяновская область"), "\nКоличество олигархов в Республике Татарстан:",
          np.count_nonzero(Oligarchs == "Республика Татарстан"), "\nКоличество олигархов в Казанской области:",
          np.count_nonzero(Oligarchs == "Казанская область"), "\nБольше всего сотрудников в:",
          "Ульяновской области" if ((np.count_nonzero(Oligarchs == "Ульяновская область") \
                                            > np.count_nonzero(Oligarchs == "Республика Татарстан")) and (
                                                       np.count_nonzero(Oligarchs == "Ульяновская область") \
                                                       > np.count_nonzero(Oligarchs == "Казанская область")))
          else ("Республике Татарстан" if (np.count_nonzero(Oligarchs == "Республика Татарстан") \
                                         > np.count_nonzero(Oligarchs == "Казанская область"))
                else "Казанской области"), "\n----------------")


# Работа с помощью PANDAS
def pandas():
    data = pd.read_csv("Oligarchs_spinner.csv", encoding='cp1251')
    # Вывод данных в консоль(Pandas)
    print("Задание PANDAS\n----------------\nСтатистика олигархов\nОбщее колличество олигархов:", data["Табельный номер"].count(),
          "\nДоля олигархов мужского пола:", round(data["Пол"].value_counts()["Мужчина"] / data["Пол"].shape[0], 3),
          "\nДоля олигархов женского пола:", round(data["Пол"].value_counts()["Женщина"] / data["Пол"].shape[0], 3),
          "\nБольше всего олигархов появилось в", data["Год начала работы"].mode()[0], "году\n----------------"
          "\nДенежное состояние\nМаксимальное денежное состояние олигархов:", data["Состояние"].max(),
          "\nМинимальное денежное состояние олигархов:", data["Состояние"].min(), "\nДисперсия денежного состояния:",
          round(data["Состояние"].var(), 3), "\nСт. откл. денежного состояния олгархов:", round(data["Состояние"].std(), 3),
          "\nМедиана кол-ва денежного состояния олигархов:", data["Состояние"].median(), "\nМода денежного состояния",
          data["Состояние"].mode()[0], "\n----------------\nГруппировка олигархов"
                                       "\nКоличество олигархов в Ульяновской области:",
          (data["Подразделение экономической деятельности"].value_counts()["Ульяновская область"]),
          "\nКоличество олигархов в Республике Татарстан:",
          (data["Подразделение экономической деятельности"].value_counts()["Республика Татарстан"]),
          "\nКоличество олигархов в Казанской области:",
          (data["Подразделение экономической деятельности"].value_counts()["Казанская область"]),
          "\nБольше всего олигархов в:",
          "Ульяновской области" if (
                      (data["Подразделение экономической деятельности"].value_counts()["Ульяновская область"] \
                       > data["Подразделение экономической деятельности"].value_counts()["Республика Татарстан"]) and (
                              data["Подразделение экономической деятельности"].value_counts()[
                                  "Ульяновская область"] \
                              > data["Подразделение экономической деятельности"].value_counts()[
                                  "Республика Татарстан"]))
          else ("Республике Татарстан" if (data["Подразделение экономической деятельности"].value_counts()
                                           ["Республика Татарстан"] \
                          > data["Подразделение экономической деятельности"].value_counts()[
                              "Казанская область"]) else "Казанской области"))
# Работа с графиками(3шт)
def graphics():
    data = pd.read_csv("Oligarchs_spinner.csv", encoding='cp1251')
    proportion = {}
    positions = data['Должность'].unique()

    for item in positions:
        pos = data[data['Должность'] == item]
        proportion[item] = round(pos['Колличество предприятий'].sum() / pos['Колличество предприятий'].count(), 2)
    # График распределения соотношения по полу(мужской или женский)
    sex = [data["Пол"].value_counts()["Мужчина"], data["Пол"].value_counts()["Женщина"]]
    plt.pie(sex, labels=["Мужчины", "Женщины"])
    plt.title('Соотношение олигархов по полу', loc='center')
    plt.show()
    # График группировки по области нахождения(Ульяновская область, Казанская область, Республика Татарстан)
    plt.hist(data['Подразделение экономической деятельности'], bins=8, color='y')
    plt.title('Группировка олигархов', loc='center')
    plt.show()
    # График по колличеству олигархов в обределённой должности(миллионер или миллиардер, рублейвой или доллоровый)
    plt.bar(proportion.keys(), proportion.values(), color='r')
    plt.title('Соотношение количества предприятий по должностям', loc='right')
    plt.show()

table()
numpy()
pandas()
graphics()
