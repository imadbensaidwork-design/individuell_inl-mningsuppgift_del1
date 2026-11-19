# DENNA MODUL INNEHÅLLER..

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)


# En enkel linjär regression för att förutsäga blodtryck från ålder eller vikt.


def linear_regression(df):
    plt.figure(figsize=(6, 4))
plt.scatter(df["study_hours"], df["exam_score"], alpha=0.7)
plt.xlabel("Studietimmar (X)")
plt.ylabel("Provpoäng (Y)")
plt.title("Samband mellan X och Y")
plt.show()
