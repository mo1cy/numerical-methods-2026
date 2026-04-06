import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#задана функція навантаження
def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

a, b = 0, 24

#точне значення інтегралу
I0, _ = quad(f, a, b, epsabs=1e-14, epsrel=1e-14)
print(f"Точне значення інтегралу I0 = {I0:.12f}")

#складова квадратурна формула Сімпсона
def simpson(f, a, b, N):
    if N % 2 != 0:
        raise ValueError("N має бути парним")
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])

#дослідження залежності точності від N
N_values = np.arange(10, 1002, 2)
errors = [abs(simpson(f, a, b, n) - I0) for n in N_values]

target_eps = 1e-12
N_opt = None
eps_opt = None

for n, err in zip(N_values, errors):
    if err <= target_eps:
        N_opt = n
        eps_opt = err
        break

if N_opt is None:
    N_opt = N_values[-1]
    eps_opt = errors[-1]

print(f"Задана точність досягається при N_opt = {N_opt}, eps_opt = {eps_opt:.2e}")

#побудова графіка залежності похибки від N
plt.figure(figsize=(10, 5))
plt.plot(N_values, errors, label=r'$\epsilon(N) = |I(N) - I_0|$', color='red')
plt.axhline(target_eps, color='blue', linestyle='--', label='Задана точність $10^{-12}$')
plt.yscale('log')
plt.title('Залежність похибки формули Сімпсона від кількості розбиттів N')
plt.xlabel('Число розбиттів, N')
plt.ylabel('Похибка $\epsilon$ (логарифмічний масштаб)')
plt.grid(True)
plt.legend()
plt.show()

#похибка при N0
N0 = int(N_opt / 10)
if N0 % 8 != 0:
    N0 += 8 - (N0 % 8) #кратне 8
if N0 == 0:
    N0 = 8

I_N0 = simpson(f, a, b, N0)
eps0 = abs(I_N0 - I0)
print(f"\nВибране N0 = {N0}. Значення I(N0) = {I_N0:.12f}, eps0 = {eps0:.2e}")

#метод Рунге-Ромберга
I_N0_2 = simpson(f, a, b, N0 // 2)
I_R = I_N0 + (I_N0 - I_N0_2) / 15
eps_R = abs(I_R - I0)
print(f"Метод Рунге-Ромберга: I_R = {I_R:.12f}, eps_R = {eps_R:.2e}")

#метод Ейткена
I_N0_4 = simpson(f, a, b, N0 // 4)

#обчислення порядку методу p
numerator = abs(I_N0_4 - I_N0_2)
denominator = abs(I_N0_2 - I_N0)
p = (1 / np.log(2)) * np.log(numerator / denominator)

#уточнене значення
I_E = (I_N0_2**2 - I_N0 * I_N0_4) / (2 * I_N0_2 - (I_N0 + I_N0_4))
eps_E = abs(I_E - I0)
print(f"Метод Ейткена: Порядок методу p = {p:.4f}")
print(f"Метод Ейткена: I_E = {I_E:.12f}, eps_E = {eps_E:.2e}")

#адаптивний алгоритм
def adaptive_simpson(f, a, b, delta, calc_count=0):
    c = (a + b) / 2
    h = b - a
    
    #інтеграл на одному відрізку (1 крок)
    I1 = (h / 6) * (f(a) + 4 * f(c) + f(b))
    calc_count += 3
    
    #інтеграл на двох половинках (2 кроки)
    d = (a + c) / 2
    e = (c + b) / 2
    I2 = (h / 12) * (f(a) + 4 * f(d) + f(c)) + (h / 12) * (f(c) + 4 * f(e) + f(b))
    calc_count += 4 # f(a), f(c), f(b) вже є, додаємо f(d), f(e)

    if abs(I1 - I2) <= 15 * delta: #15 з'являється з оцінки Рунге
        return I2 + (I2 - I1) / 15, calc_count
    else:
        left_I, left_calc = adaptive_simpson(f, a, c, delta / 2, 0)
        right_I, right_calc = adaptive_simpson(f, c, b, delta / 2, 0)
        return left_I + right_I, calc_count + left_calc + right_calc

delta_target = 1e-8
I_adapt, func_evals = adaptive_simpson(f, a, b, delta_target)
eps_adapt = abs(I_adapt - I0)
print(f"\nАдаптивний алгоритм (delta={delta_target}):")
print(f"I_adapt = {I_adapt:.12f}, eps_adapt = {eps_adapt:.2e}, Викликів функції: {func_evals}")

# Задана функція 
def f(x): 
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2) 
 
# Інтервал 
x = np.linspace(0, 24, 1000) 
y = f(x) 
 
# Побудова графіка 
plt.figure(figsize=(10, 6)) 
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$') 
plt.title('Графік функції навантаження на сервер') 
plt.xlabel('Час, x (год)') 
plt.ylabel('Навантаження, f(x)') 
plt.grid(True) 
plt.legend() 
plt.show()