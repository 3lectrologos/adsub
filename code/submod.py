import abc
import random
import Queue


DEBUG = False

class AdaptiveMax(object):
    def __init__(self, E, f=None):
        self.E = E
        if f: self.f = lambda v, a: f(a + [v])

    def init_f_hook(self):
        self.sol = []
        self.fsol = 0

    def update_f_hook(self):
        self.fsol = self.f(self.sol[-1], self.sol[:-1])

    def greedy(self, k):
        self.init_f_hook()
        q = Queue.PriorityQueue()
        for v in self.E:
            df = self.f(v, [])
            q.put((-df, v, 1))
        i = 1
        while i <= k:
            e = q.get()
            fv, v, iv = e
            if -fv <= 0:
                print 'Warning: Greedy terminated due to negative marginal gain.'
                break
            if iv < i:
                df = self.f(v, self.sol) - self.fsol
                q.put((-df, v, i))
            else:
                self.sol.append(v)
                self.update_f_hook()
                i = i + 1
        return (self.sol, self.fsol)

    def random_greedy(self, k):
        self.init_f_hook()
        q = Queue.PriorityQueue()
        for v in self.E:
            df = self.f(v, [])
            q.put((-df, v, 1))
        i = 1
        sk = []
        #foobar = 0
        while i <= k:
            if len(sk) == k or (len(sk) > 0 and q.empty()):
                j = random.randint(0, len(sk)-1)
                fv, v, iv = sk.pop(j)
                self.sol.append(v)
                self.update_f_hook()
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
                #print 'computing', v, 'foobar =', foobar
                #foobar += 1
                df = self.f(v, self.sol) - self.fsol
                q.put((-df, v, i))
            else:
                sk.append(e)
        return (self.sol, self.fsol)
