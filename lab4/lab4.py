import math
import numpy as np
import matplotlib.pyplot as plt

# функція вологості
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

# точне значення похідної
def exact_derivative(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

# чисельна похідна (центральна різниця)
def y_prime(t, h):
    return (M(t + h) - M(t - h)) / (2 * h)

# точка для обчислення похідної
x0 = 1.0
exact_val = exact_derivative(x0)

# 1. малюємо графік самої функції
t_vals = np.linspace(0, 20, 500)
plt.figure(figsize=(10, 5))
plt.plot(t_vals, M(t_vals), label='M(t)', color='teal')
plt.title('модель вологості ґрунту')
plt.xlabel('час (t)')
plt.ylabel('вологість')
plt.grid(True)
plt.legend()
plt.show()

# 2. шукаємо оптимальний крок h
h_values = []
errors = []
best_h = None
min_error = float('inf')

# перебираємо степені від -20 до 3
for p in range(-20, 4):
    h = 10**p
    if h == 0: continue
    
    val = y_prime(x0, h)
    err = abs(val - exact_val)
    
    h_values.append(h)
    errors.append(err if err > 0 else 1e-16) # щоб графік не впав через нуль
    
    # запам'ятовуємо найкращий результат
    if err < min_error:
        min_error = err
        best_h = h

# малюємо графік похибки
plt.figure(figsize=(10, 5))
plt.loglog(h_values, errors, marker='o', color='crimson')
plt.axvline(best_h, color='black', linestyle='--', label=f'h0 = {best_h:.0e}')
plt.title('залежність похибки від кроку h')
plt.xlabel('крок h (логарифм)')
plt.ylabel('похибка (логарифм)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.gca().invert_xaxis() # розвертаємо вісь, щоб бачити зменшення кроку
plt.show()

print(f"оптимальний крок: {best_h:.0e}")
print(f"мінімальна похибка R0: {min_error:.10e}\n")

# метод рунге-ромберга
h = 1e-3 # базовий крок
y_h = y_prime(x0, h)
y_2h = y_prime(x0, 2 * h)

R1 = abs(y_h - exact_val)
print(f"похибка R1 (крок h): {R1:.10e}")

# уточнюємо за рунге-ромбергом
y_R = y_h + (y_h - y_2h) / 3
R2 = abs(y_R - exact_val)
print(f"уточнене значення (рунге-ромберг): {y_R:.10f}")
print(f"похибка R2: {R2:.10e}\n")

# метод ейткена
y_4h = y_prime(x0, 4 * h)

numerator = y_2h**2 - y_4h * y_h
denominator = 2 * y_2h - (y_4h + y_h)

if denominator != 0:
    # уточнюємо за ейткеном
    y_E = numerator / denominator
    R3 = abs(y_E - exact_val)
    
    # рахуємо порядок точності
    ratio = abs((y_4h - y_2h) / (y_2h - y_h))
    p_order = (1 / math.log(2)) * math.log(ratio)
    
    print(f"уточнене значення (ейткен): {y_E:.10f}")
    print(f"порядок точності p: {p_order:.4f}")
    print(f"похибка R3: {R3:.10e}")