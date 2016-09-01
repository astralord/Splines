class Point:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w


class Points:
    def __init__(self, x, y, w):
        min_len = min(len(x), len(y), len(w))
        self.x = x[0:min_len]
        self.y = y[0:min_len]
        self.w = w[0:min_len]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return Point(self.x[item], self.y[item], self.w[item])
