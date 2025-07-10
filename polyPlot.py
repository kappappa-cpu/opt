import matplotlib.pyplot as plt
import sys
import numpy as np

"""
1変数多項式をグラフで表示するモジュール

引数1: 次元数
引数2以下、各a_i
ex) polyPlot.py 2 1 2 3
    → 1*x**2 + 2*x + 3
"""

def plot_polynomial(degree, coeffs):
    x = np.linspace(-100, 100, 400)
    y = np.polyval(coeffs, x)
    plt.plot(x, y)
    plt.title(f"Polynomial: degree {degree}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    degree = int(sys.argv[1])
    coeffs = []
    for i in range(degree + 1):
        coeffs.append(int(sys.argv[i+2]))
    plot_polynomial(degree, coeffs)