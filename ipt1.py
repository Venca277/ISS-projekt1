import numpy as np
import scipy.stats as stats
from scipy.stats import rankdata

income = [56.1, 41.1, 61.5,	47.7, 58.2, 48,	47.9, 55.1, 43,	46.9, 42.6, 51.2]
expenses = [18.6, 16.7, 23.1, 16.6, 20.7, 14.4, 14.6, 18.7, 13.2, 18.5, 13.5, 17.4]

print(f"Příjmy: {income}")
print(f"Výdaje: {expenses}")

# kontrola, jestli je každá hodnota v polích jednou
d = {}
for key in income:
    d[key] = d.get(key, 0) + 1

for i in d:
    if (d[i] > 1):
        print(f"Chyba: Hodnota {i} se vyskytuje {d[i]} krát")
        exit()

d = {}
for key in expenses:
    d[key] = d.get(key, 0) + 1

for i in d:
    if (d[i] > 1):
        print(f"Chyba: Hodnota {i} se vyskytuje {d[i]} krát")
        exit()

n = len(income)
e_len = len(expenses)

assert(n == e_len)

rho, p_value = stats.spearmanr(np.array(income), np.array(expenses))

t = (rho * np.sqrt(n - 2)) / np.sqrt(1 - rho ** 2)

alpha = 0.05
critical_value = stats.t.ppf(1 - alpha / 2, df=n - 2)

ri = rankdata(income, method='ordinal')
qi = rankdata(expenses, method='ordinal')

print(f"Pořadí příjmů r_i: {list(ri)}")
print(f"Pořadí výdajů q_i: {list(qi)}")

print(f"Spearmanův korelační koeficient (rho): {rho}")
print(f"    > {rho:.3f}")
print(f"Testovací statistika (T): {t}")
print(f"    > {t:.3f}")
print(f"W = (-inf, -c) U (c, inf)")
print(f"c = Kritická hodnota: ±{critical_value}")
print(f"    > {critical_value:.3f}")

if abs(t) > critical_value:
    print("Zamítáme nulovou hypotézu. Podařilo se prokázat, že...")
else:
    print("Nezamítáme nulovou hypotézu. Nepodařilo se prokázat, že...")