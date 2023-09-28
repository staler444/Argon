import pandas as pd
import matplotlib.pyplot as plt
jd = pd.read_csv("argon_energia.csv")
jd = jd.values.tolist()

def gen(co, labx, laby):
    en = []
    for it in jd:
        en.append(it[co])
    plt.plot(range(0, 20000, 100), en)
    plt.xlabel(labx)
    plt.ylabel(laby)
    plt.show()

# example use
gen(6, "Krok symulacji", "Temperatura (K)")
