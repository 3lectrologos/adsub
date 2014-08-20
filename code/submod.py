import abc
import random
import Queue


DEBUG = False

class AdaptiveMax:
    def __init__(self, E, f=None, data=None):
        self.E = E
        if f: self.f = f
        if data: self.data = data

    def init_f_hook(self):
        return

    def update_f_hook(self):
        return

    def greedy(self, k):
        q = Queue.PriorityQueue()
        for v in self.E:
            q.put((-self.f([v]), v, 1))
        i = 1
        self.sol = []
        self.fsol = 0
        while i <= k:
            e = q.get()
            fv, v, iv = e
            assert fv <= 0, 'non-monotone function: negative marginal gain'
            if iv < i:
                q.put((-self.f(self.sol + [v]) + self.fsol, v, i))
            else:
                self.sol.append(v)
                self.fsol = self.fsol - fv
                i = i + 1
        return (self.sol, self.fsol)

    def random_greedy(self, k):
        self.init_f_hook()
        q = Queue.PriorityQueue()
        for v in self.E:
            q.put((-self.f([v]), v, 1))
        i = 1
        sk = []
        self.sol = []
        self.fsol = 0
        while i <= k:
            if len(sk) == k or (len(sk) > 0 and q.empty()):
                j = random.randint(0, len(sk)-1)
                fv, v, iv = sk.pop(j)
                self.sol.append(v)
                self.fsol = self.fsol - fv
                i = i + 1
                for e in sk:
                    q.put(e)
                sk = []
            if q.empty():
                break
            e = q.get()
            fv, v, iv = e
            if -fv <= 0:
                continue
            if iv < i:
                q.put((-self.f(self.sol + [v]) + self.fsol, v, i))
            else:
                sk.append(e)
        return (self.sol, self.fsol)
