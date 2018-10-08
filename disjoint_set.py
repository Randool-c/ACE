class Disjoint:
    def __init__(self, laplacian):
        self.size = laplacian.shape[0]
        self.parent = [-1] * self.size
        self.length = self.size
        self.__contruct(laplacian)

    def find(self, i):
        if self.parent[i] < 0:
            return i
        else:
            tmp = self.find(self.parent[i])
            self.parent[i] = tmp
            return tmp

    def union(self, root1, root2):
        if self.parent[root1] < self.parent[root2]:
            self.parent[root1] += self.parent[root2]
            self.parent[root2] = root1
        else:
            self.parent[root2] += self.parent[root1]
            self.parent[root1] = root2
        self.length -= 1

    def __contruct(self, laplacian):
        n = laplacian.shape[0]
        for i in range(n):
            for j in range(i, n):
                if laplacian[i, j] != 0 and self.find(i) != self.find(j):
                    self.union(self.find(i), self.find(j))

    def get_length(self):
        return self.length
