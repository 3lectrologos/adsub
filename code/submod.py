import random
import Queue


def greedy(f, s, k):
    q = Queue.PriorityQueue()
    for v in s:
        q.put((-f([v]), v, 1))
    i = 1
    sol = []
    fsol = 0
    while i <= k:
        e = q.get()
        fv, v, iv = e
        assert fv >= 0, 'negative marginal gain'
        if iv < i:
            q.put((-f(sol + [v]) + fsol, v, i))
        else:
            sol = sol + [v]
            fsol = fsol - fv
            i = i + 1
    return (sol, fsol)

def random_greedy(f, s, k):
    q = Queue.PriorityQueue()
    for v in s:
        q.put((-f([v]), v, 1))
    i = 1
    sk = []
    sol = []
    fsol = 0
    while i <= k:
        if len(sk) == k or (len(sk) > 0 and q.empty()):
            j = random.randint(0, len(sk)-1)
            fv, v, iv = sk.pop(j)
            sol = sol + [v]
            fsol = fsol - fv
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
            q.put((-f(sol + [v]) + fsol, v, i))
        else:
            sk.append(e)
    return (sol, fsol)
