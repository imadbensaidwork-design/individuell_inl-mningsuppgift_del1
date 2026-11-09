# FUNKTIONER FÖR GRAFISK VISUALISERING AV ANALYS

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# FUNKTION 1: HISTOGRAM ÖVER BLODTRYCK

def hist(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["systolic_bp"], bins=25) # väljer kolumnen systolic_bp (blodtryck)
    ax.set_title("Frekvensfördelning av blodtryck")
    ax.set_xlabel("Blodtryck")
    ax.set_ylabel("Antal") # antal individere
    ax.grid(True, axis="y") # lägger till rutnät längs Y-axeln (vågräta linjer)
    plt.tight_layout()
    return fig

# FUNKTION 2: # LÅDDIAGRAM ÖVER VIKT PER KÖN

def box(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    weight_m = df.loc[df["sex"] == "M", "weight"] # hämtar vikterna för män
    weight_f = df.loc[df["sex"] == "F", "weight"] # hämtar vikterna för kvinnor

    ax.boxplot( # tar fram ett låddiagram
    [weight_m, weight_f], # skickar in vikter för män och kvinnor
    patch_artist=True, # gör lådorna möjliga att färglägga
    boxprops=dict(facecolor="lightblue"), # färgar lådorna ljusblå
    medianprops=dict(color="red") # färgar medianlinjen röd
    )

    ax.set_title("Vikt per kön")
    ax.set_xlabel("Kön")
    ax.set_ylabel("Vikt") # antal individere
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

# FUNKTION 3: # STAPELDIAGRAM ÖVER ANDELEN RÖKARE PER KÖN

def bar(df):
    fig, ax = plt.subplots(figsize=(8, 5))

    # 1. Filtrerar DataFrame med ett villkor
    # df["smoker"] == "Yes" skapar en True/False-kontroll för varje rad
    # df.loc använder kontrollen för att välja endast rader där villkoret är True
    # "sex" efter kommatecknet anger att vi vill hämta just den kolumnen
    smoker_yes = df.loc[df["smoker"] == "Yes", "sex"]

    # 2. Väljer ut två kolumner från ursprungliga DataFrame
    # Dubbla hakparenteser används eftersom vi väljer flera kolumner
    # Resultatet blir en ny, mindre DataFrame
    smoker_total = df[["smoker", "sex"]]

    # 3. Räknar hur många gånger varje unikt värde förekommer
    # value_counts() räknar hur många M respektive F som finns bland rökarna
    smoker_yes = smoker_yes.value_counts()

    # 4. Grupperar efter kön och räknar antalet rader i varje grupp
    # groupby("sex") delar upp personerna i grupperna M och F
    # size() räknar antalet rader i varje grupp
    smoker_total = smoker_total.groupby("sex").size()
    # Exempel på resultat:
    # M    214  # Totalt antal män
    # F    183  # Totalt antal kvinnor
   
    smoker_rate = (smoker_yes / smoker_total * 100) # jag delar antalet rökare med totalen (smoker == Yes och == No)
                                                    # Pandas matchar automatiskt könen via deras index, inte via radposition

    # smoker_rate.index (x-axeln) hämtar indexet från Series, alltså kategorierna (M och F)
    # smoker_rate.values (y-axeln) hämtar själva värdena, alltså andelen rökare i procent för respektive kön
    # ax.bar använder kategorierna på X-axeln och procentvärdena som stapelhöjd på Y-axeln
    ax.bar(smoker_rate.index, smoker_rate.values)
    ax.set_title("Andel rökare per kön")
    ax.set_xlabel("Kön") 
    ax.set_ylabel("Rökare (%)")
    ax.grid(True, axis="y")
    plt.tight_layout()  
    return fig