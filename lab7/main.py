import random

def generate_and_write_data(n, exact_x):
    with open("A.txt", "w") as file_a, open("B.txt", "w") as file_b:
        A = []
        for i in range(n):
            row = []
            sum_row_abs = 0
            #спочатку генеруємо випадкові числа
            for j in range(n):
                if i != j:
                    val = random.uniform(-10.0, 10.0)
                    row.append(val)
                    sum_row_abs += abs(val)
                else:
                    row.append(0.0)
            
            #робимо строге діагональне переважання: діагональ > суми інших елементів
            diag_val = sum_row_abs + random.uniform(10.0, 50.0)
            row[i] = diag_val
            
            #обчислюємо вектор B
            sum_b = sum(row[j] * exact_x for j in range(n))
            A.append(row)
            
            #запис у файли
            file_a.write(" ".join(f"{v:.6f}" for v in row) + "\n")
            file_b.write(f"{sum_b:.6f}\n")
            
    print("Файли A.txt та B.txt успішно згенеровано (з діагональним переважанням).")


def read_matrix(filename):
    A = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                A.append([float(x) for x in line.split()])
    return A

def read_vector(filename):
    B = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                B.append(float(line.strip()))
    return B

def multiply_mat_vec(A, X, n):
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(n))
    return res

def calculate_vector_norm(V):
    return max(abs(v) for v in V)

def calculate_matrix_norm(A, n):
    return max(sum(abs(x) for x in row) for row in A)


#метод простої ітерації
def solve_simple_iteration(A, B, n, eps):
    X = [1.0] * n #початкове наближення
    iterations = 0
    
    #знаходимо параметр tau: 0 < tau < 2/||A||
    tau = 1.0 / calculate_matrix_norm(A, n) 
    
    while True:
        AX = multiply_mat_vec(A, X, n)
        max_diff = 0.0
        X_new = [0.0] * n
        
        for i in range(n):
            X_new[i] = X[i] - tau * (AX[i] - B[i])
            
            diff = abs(X_new[i] - X[i])
            if diff > max_diff:
                max_diff = diff
                
        X = X_new
        iterations += 1
        
        if max_diff <= eps:
            break
    return X, iterations

#метод Якобі
def solve_jacobi(A, B, n, eps):
    X = [1.0] * n 
    iterations = 0
    
    while True:
        max_diff = 0.0
        X_new = [0.0] * n
        
        for i in range(n):
            #сума всіх елементів рядка, крім діагонального
            s = sum(A[i][j] * X[j] for j in range(n) if j != i)
            X_new[i] = (B[i] - s) / A[i][i]
            
            diff = abs(X_new[i] - X[i])
            if diff > max_diff:
                max_diff = diff
                
        X = X_new
        iterations += 1
        
        if max_diff <= eps:
            break
    return X, iterations

#метод Гауса-Зейделя
def solve_seidel(A, B, n, eps):
    X = [1.0] * n
    iterations = 0
    
    while True:
        max_diff = 0.0
        
        for i in range(n):
            old_xi = X[i]
            
            # В методі Зейделя ми одразу використовуємо вже оновлені значення X[j] для j < i
            s1 = sum(A[i][j] * X[j] for j in range(i))
            s2 = sum(A[i][j] * X[j] for j in range(i + 1, n))
            
            X[i] = (B[i] - s1 - s2) / A[i][i]
            
            diff = abs(X[i] - old_xi)
            if diff > max_diff:
                max_diff = diff
                
        iterations += 1
        if max_diff <= eps:
            break
    return X, iterations


def main():
    n = 100
    exact_x = 2.5
    eps0 = 1e-14 #задана точність
    
    #зчитування даних
    generate_and_write_data(n, exact_x)
    A = read_matrix("A.txt")
    B = read_vector("B.txt")
    
    print(f"\nРозмірність матриці: {n}x{n}")
    print(f"Початкове наближення x_i = 1.0, Точність eps = {eps0}")
    print("-" * 50)
    
    #знаходження уточненого розв'язку кожним із методів
    # Метод простої ітерації
    X_si, iters_si = solve_simple_iteration(A, B, n, eps0)
    print(f"1. Метод простої ітерації:")
    print(f"   Ітерацій: {iters_si}")
    print(f"   Перевірка X[0] = {X_si[0]:.5f}, X[{n-1}] = {X_si[-1]:.5f}\n")

    #метод Якобі
    X_jacobi, iters_jacobi = solve_jacobi(A, B, n, eps0)
    print(f"2. Метод Якобі:")
    print(f"   Ітерацій: {iters_jacobi}")
    print(f"   Перевірка X[0] = {X_jacobi[0]:.5f}, X[{n-1}] = {X_jacobi[-1]:.5f}\n")

    #метод Зейделя
    X_seidel, iters_seidel = solve_seidel(A, B, n, eps0)
    print(f"3. Метод Зейделя:")
    print(f"   Ітерацій: {iters_seidel}")
    print(f"   Перевірка X[0] = {X_seidel[0]:.5f}, X[{n-1}] = {X_seidel[-1]:.5f}\n")

if __name__ == "__main__":
    main()