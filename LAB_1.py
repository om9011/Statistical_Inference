def mean(data):
    return sum(data) / len(data)


def variance(data):
    n = len(data)
    var = (1 / (n - 1)) * sum((x - mean(data)) ** 2 for x in data)
    return var


def COV(X, Y):
    n = len(X)
    mean_x = mean(X)
    mean_y = mean(Y)
    cov = (1 / (n - 1)) * sum((X[i] - mean_x) * (Y[i] - mean_y) for i in range(n))
    return cov


def cal_b(X, Y):
    return COV(X, Y) / variance(X)


def cal_a(X, Y):
    a = mean(Y) - cal_b(X, Y) * mean(X)
    return round(a, 4)


def linear_regression(X, Y):
    print("a : ", cal_a(X, Y))
    print("b: ", cal_b(X, Y))
    print()
    print("Equation Will Be :")
    print("Y =", cal_b(X, Y), "X +", cal_a(X, Y))


def predict(x):
    return cal_a(X, Y) + (cal_b(X, Y) * x)


X = [1, 2, 3, 4, 5]
Y = [1, 2, 1.3, 3.75, 2.25]

linear_regression(X, Y)
print(predict(6))
