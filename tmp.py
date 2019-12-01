a = [1, 1]


def fun(a):
    print a
    return a


for i in range(5):
    print len(a)
    b = fun(a+a)
