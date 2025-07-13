import sympy as sp

x, y = sp.symbols('x y')
expr = sp.sympify("-x**3-3*x*y**2+y**3+3*x")  # 文字列 → 式

# lambdify で関数化（NumPyベース）
f = sp.lambdify((x, y), expr, "numpy")

print(f(1, 2))  # 例: x=1, y=2