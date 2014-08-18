import Queue


def greedy_max(f, s, k):
    q = Queue.PriorityQueue()
    for v in s:
        q.put((-f([v]), v, 1))
    i = 1
    sol = []
    fsol = 0
    while i <= k:
        e = q.get()
        fv, v, iv = e
        if iv < i:
            q.put((-f(sol + [v]) + fsol, v, i))
        else:
            sol = sol + [v]
            fsol = fsol - fv
            i = i + 1
    return (sol, fsol)
