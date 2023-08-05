import matplotlib.pyplot as plt


def Sum_XY(X, Y):
    return (sum(X[i] * Y[i] for i in range(len(X))))


def product_X_Y(X, Y):
    return sum(X) * sum(Y)


def sum_x_square(X):
    return sum([x ** 2 for x in X])


def cal_b1(X, Y):
    b1 = ((len(X) * Sum_XY(X, Y)) - (product_X_Y(X, Y))) / (len(X) * sum_x_square(X) - sum(X) ** 2)
    return b1


def cal_b0(X, Y):
    return round((sum(Y) - (cal_b1(X, Y) * sum(X))) / len(X), 2)


def equation(x, X, Y):
    return cal_b0(X, Y) + (cal_b1(X, Y) * x)


def print_equation(X, Y):
    print("The Equation is :")
    print("Y = ", cal_b0(X, Y), "+", round(cal_b1(X, Y), 3), "X")
    print()


def predict_Y(X, Y):
    list = []
    for i in X:
        list.append(equation(i, X, Y))
    return list


def least_square(X, Y):
    Y0 = predict_Y(X, Y)
    return [Y[i] - Y0[i] for i in range(len(Y))]


def table(X, Y):
    Y0 = predict_Y(X, Y)
    error = least_square(X, Y)
    print("--" * 15)
    print("X\t  Y\t\t  Y0\t  Error")
    print("--" * 15)
    for i in range(len(X)):
        print(X[i], "\t", Y[i], "\t", round(Y0[i], 1), "\t", round(error[i], 3))


x = [1, 2, 3, 4, 5, 6, 7]
y = [1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16]

print_equation(x, y)
print()
table(x, y)
