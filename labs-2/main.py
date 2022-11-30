#Лабораторная работа №2
#Выполнил студент группы ИСТбд-41
#Калашников Михаил Андреевич
  import numpy as np
  import matplotlib.pyplot as mpl
#Ввод значений N и K
  N=int(input("Введите четное N,которое >3: "))
  K=int(input("Введите K: "))
#Создание матриц
  matricaB=np.random.randint(-10,10,(N//2,N//2))
  matricaC=np.random.randint(-10,10,(N//2,N//2))
  matricaD=np.random.randint(-10,10,(N//2,N//2))
  matricaE=np.random.randint(-10,10,(N//2,N//2))
    matricaA=np.array([[matricaE, matricaB],[matricaD, matricaC]])
      matricaF=matricaA
arrNat=np.array([-7,-5,-3,-2,2,3,5,7])
#Вывод матриц
  print("Матрица "B"")
   print(matricaB)
  print("Матрица "C"")
    print(matricaC)
  print("Матрица "D"")
   print(matricaD)
  print("Матрица "E"")
   print(matricaE)
  print("Матрица "A"")
   print(matricaA)
countNatC=0
flag=0
#Просчитываем количество натуральных чисел в нечетных столбцах матрицы "С"
for row in range(0, N//2):
    for col in range(1, N//2, 2):
        for nat in range(0, 7):
            if matricaC[row][col] == arrNat[nat]:
                flag=1
        if flag == 1:
            countNatC+=1
            flag=0
print("Количество простых чисел в нечетных столбцах: ", countNatC)

countZeroC=0
#Считаем количество нулевых элементов в четных строках matrС3
for row in range(0, N//2, 2):
    for col in range(0, N//2):
        if matricaC[row][col] == 0:
            countZeroC+=1
print("Количество нулевых элементов в четных строках: ", countZeroC)

cont=0
r=N//2-1
c=N//2-1
print("Матрица F перед преобразованием:")
print(matricaF)

if countNatC > countZeroC:
    print("Меняем E и C симметрично")
    for row in range(0, N//2):
        for col in range(0, N//2):
            cont = matricaF[0][0][row][col]
            matricaF[0][0][row][col] = matricaF[1][1][r][c]
            matricaF[1][1][r][c] = cont
            r -= 1
        c -= 1
        r = N//2-1
else:
    print("Меняем B и C несимметрично")
    for row in range(0, N//2):
        for col in range(0, N//2):
            cont=matricaF[0][1][row][col]
            matricaF[0][1][row][col]=matricaF[1][1][row][col]
            matricaF[1][1][row][col]=cont

print("Матрица F после преобразования:")
print(matricaF)

#Объединение подматриц A в одну
pod_matrixA1=np.concatenate((matricaE,matricaB), axis=1)
pod_matrixA2=np.concatenate((matricaD,matricaC), axis=1)
pod_matrixA=np.concatenate((pod_matrixA1, pod_matrixA2), axis=0)

#Объединение подматриц F в одну
pod_matrixF1=np.concatenate((matricaF[0][0],matricaF[0][1]), axis=1)
pod_matrixF2=np.concatenate((matricaF[1][0],matricaF[1][1]), axis=1)
pod_matrixF=np.concatenate((pod_matrixF1, pod_matrixF2), axis=0)

detA=np.linalg.det(pod_matrixA)
print("Определитель матрицы А:", detA)
sumDiagF=np.trace(matricaF[0][0])+np.trace(matricaF[1][1])
print("Сумма диагонали матрицы F:", sumDiagF)
matricaAT = np.transpose(pod_matrixA)
print("Транспонированная матрица А:", matricaAT)

if detA>sumDiagF:
    #Вычисляем A^(-1)*A^T-K*F
    matricaAInv=np.linalg.inv(pod_matrixA)
    out=matricaAInv*matricaAT-K*pod_matrixF
    print("Определитель больше суммы, результат выражения A^(-1)*A^T-K*F: ", out)
else:
    #Вычисляем (A^T+G^(-1)-F^(-1))*K
    matricaG=np.tril(pod_matrixA)
    matricaGInv=np.linalg.inv(matricaG)
    matricaFInv=np.linalg.inv(pod_matrixF)
    out=(matricaAT+matricaGInv-matricaFInv)*K
    print("Определитель меньше или равен сумме, результат выражения (A^T+G^(-1)-F^(-1))*K: ", out)

print("График значений строк матрицы F")
x = np.linspace(0, N, N)
y = np.asarray(pod_matrixF)
mpl.plot(x, y)
mpl.title("График значений строк матрицы F")
mpl.show()

print("Гистограмма распределения значений матрицы F")
x = np.linspace(-10, 10, 21)
y = np.asarray(pod_matrixF).reshape(-1)
mpl.hist(y, x, rwidth=0.9)
mpl.title("Гистограмма распределения значений матрицы F")
mpl.show()

print("Диаграмма распределения значений подматрицы B матрицы F")
y = np.asarray(matricaF[0][1]).reshape(-1)
#Считаем количество значений
arrCount=np.zeros(21)
for i in range(0,len(y)):
    arrCount[y[i]+10]+=1
chisls=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
#Отсеиваем нулевые значения
chislsB = [value for value in chisls if arrCount[value+10]!=0]
valuesB = [value for value in arrCount if value!=0]
mpl.pie(valuesB,labels=chislsB)
mpl.title("Диаграмма распределения в подматрице B")
mpl.show()
