import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import fsolve
import sympy

"""
2変数多項式をグラフで表示するモジュール

引数例:
twoValPolyPlot.py "-x**3-3*x*y**2+y**3+3*x"
"""

def parse_poly(expr):
    x, y = sympy.symbols('x y')
    poly = sympy.sympify(expr)
    poly = sympy.expand(poly)
    coeffs_dict = {}
    max_x = 0
    max_y = 0
    # 多項式の項ごとに係数と次数を取得
    for term in poly.as_ordered_terms():
        coef = term.as_coeff_Mul()[0]
        powers = sympy.Poly(term, x, y).monoms()[0]
        x_deg, y_deg = powers
        coeffs_dict[(x_deg, y_deg)] = float(coef)
        max_x = max(max_x, x_deg)
        max_y = max(max_y, y_deg)
    # 係数リストを x^max_x→0, y^max_y→0 の順で並べる
    coeffs = []
    for i in range(max_x, -1, -1):
        for j in range(max_y, -1, -1):
            coeffs.append(coeffs_dict.get((i, j), 0.0))
    return max_x, max_y, coeffs, coeffs_dict

def poly_func(XY, coeffs_dict):
    x, y = XY
    val = 0
    for (i, j), c in coeffs_dict.items():
        val += c * (x ** i) * (y ** j)
    return val

def grad_poly(XY, coeffs_dict):
    x, y = XY
    dx = 0
    dy = 0
    for (i, j), c in coeffs_dict.items():
        if i > 0:
            dx += c * i * (x ** (i - 1)) * (y ** j)
        if j > 0:
            dy += c * j * (x ** i) * (y ** (j - 1))
    return np.array([dx, dy])

def plot_two_var_polynomial(max_x, max_y, coeffs, coeffs_dict):
    # --- 極値点を先に探索 ---
    critical_points = []
    for x0 in np.linspace(-10, 10, 5):
        for y0 in np.linspace(-10, 10, 5):
            try:
                sol, info, ier, msg = fsolve(lambda XY: grad_poly(XY, coeffs_dict), [x0, y0], full_output=True)
                if ier == 1:
                    if not any(np.allclose(sol, cp, atol=1e-3) for cp in critical_points):
                        critical_points.append(sol)
            except Exception:
                pass

    # 極値点や原点を含む範囲を決定
    points = np.array(critical_points) if critical_points else np.zeros((1,2))
    points = np.vstack([points, [0,0]])
    x_min, x_max = points[:,0].min()-2, points[:,0].max()+2
    y_min, y_max = points[:,1].min()-2, points[:,1].max()+2

    # 範囲が狭すぎる場合はデフォルト値
    if x_max - x_min < 2:
        x_min, x_max = -2, 2
    if y_max - y_min < 2:
        y_min, y_max = -2, 2

    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    idx = 0
    for i in range(max_x, -1, -1):
        for j in range(max_y, -1, -1):
            Z += coeffs[idx] * (X ** i) * (Y ** j)
            idx += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 表面のplot
    mappable = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    # mapper関数のカラーバー
    cbar = plt.colorbar(mappable, ax=ax)
    cbar.set_label('Value')
    # 等高線
    ax.contour(X, Y, Z, levels=20)

    ax.set_title(f"2-variable Polynomial: degree x={max_x}, y={max_y}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # 極値点をプロット
    if critical_points:
        cps = np.array(critical_points)
        zs = [poly_func(cp, coeffs_dict) for cp in critical_points]
        ax.scatter(cps[:,0], cps[:,1], zs, color='red', s=50, label='∇f=0')
        ax.legend()

    plt.show()

if __name__ == "__main__":
    expr = sys.argv[1]
    max_x, max_y, coeffs, coeffs_dict = parse_poly(expr)
    plot_two_var_polynomial(max_x, max_y, coeffs, coeffs_dict)