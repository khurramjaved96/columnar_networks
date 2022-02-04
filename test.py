import numpy as np
avg = 0
steps = 100
for a in range(0, steps):
    vec = np.random.choice([0, 1], 4)
    unit = np.random.normal(0, 1)
    # print(unit)
    avg += (np.sum(vec) + (unit))**2

print(avg/steps)