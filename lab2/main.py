import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

# зчитування даних з csv
def read_data(filename):
    # oтримуємо точний шлях до папки, де лежить поточний скрипт main.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # зліплюємо шлях до папки з назвою нашого файлу
    file_path = os.path.join(current_dir, filename)
    x, y = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Dataset size'])) 
            y.append(float(row['Train time (sec)']))
    return np.array(x), np.array(y)

x_data, y_data = read_data('data.csv')

# oбчислення розділених різниць
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

# інтерполяційний многочлен Ньютона
def newton_poly(coef, x_data, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n-k] + (x - x_data[n-k]) * p
    return p

coef = divided_differences(x_data, y_data)

# прогноз часу тренування для 120000
x_target = 120000
y_pred = newton_poly(coef, x_data, x_target)
print(f"Прогнозований час для датасету {x_target}: {y_pred:.2f} сек")

# побудова основного графіка
x_vals = np.linspace(min(x_data), max(x_data), 500)
y_vals = [newton_poly(coef, x_data, xi) for xi in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Поліном Ньютона", color='blue')
plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label="Експериментальні дані")
plt.scatter(x_target, y_pred, color='green', marker='X', s=100, zorder=5, label=f"Прогноз (x={x_target})")
plt.title("Прогнозування часу тренування моделі")
plt.xlabel("Розмір датасету")
plt.ylabel("Час (сек)")
plt.legend()
plt.grid(True)
plt.show()

# дослідження ефекту Рунге та похибок (n=5, 10, 20)
# використовуємо кубічний сплайн як "еталонну" криву для генерації вузлів
true_func = CubicSpline(x_data, y_data)

plt.figure(figsize=(10, 6))
for n in [5, 10, 20]:
    x_nodes = np.linspace(min(x_data), max(x_data), n)
    y_nodes = true_func(x_nodes)
    
    coef_n = divided_differences(x_nodes, y_nodes)
    y_interp = [newton_poly(coef_n, x_nodes, xi) for xi in x_vals]
    
    error = np.abs(true_func(x_vals) - y_interp)
    plt.plot(x_vals, error, label=f"Похибка при n={n}")

plt.title("Дослідження похибки інтерполяції (Ефект Рунге)")
plt.xlabel("Розмір датасету")
plt.ylabel("Абсолютна похибка")
plt.yscale("log") 
plt.legend()
plt.grid(True)
plt.show()