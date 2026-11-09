import numpy as np


# FUNKTION 1: BESKRIVANDE ANALYS AV NUMERISKA KOLUMNER

def summary_statistics(df): # funktion som tar emot DataFrame
    """Beräknar statistiska mått för utvalda numeriska kolumner"""

    columns = ["age", "weight", "height", "systolic_bp", "cholesterol"] # anger vilka kolumner som ska undersökas

    summary = df[columns].describe() # väljer kolumnerna och beräknar beskrivande statistik

    summary = summary.loc[["mean", "50%", "min", "max"]]  # väljer ut medelvärde, median, minimum och maximum
                                                          # .loc som väljer ut specifika rader i DataFrame:en

    summary["age"] = summary["age"].round(0) # avrundar värdena i ålderskolumnen till heltal
    summary["weight"] = summary["weight"].round(1) # avrundar vikt till en decimal
    summary["height"] = summary["height"].round(1) # avrundar längd till en decimal
    summary["systolic_bp"] = summary["systolic_bp"].round(0) # avrundar blodtryck till heltal
    summary["cholesterol"] = summary["cholesterol"].round(2) # avrundar kolesterol till två decimaler

    return summary # returnerar sammanställningen som en DataFrame

# FUNKTION 2: SIMULERING AV SJUKDOMSCASE

def simulation(df):

    # 1. Beräkna andelen personer i datasetet som har sjukdomen

        # # antal_sjuka = int(df["disease"].sum()) # Räknar antalet sjuka (sum() räknar alla ettor)
        # # totalt_antal_personer = len(df) # Räknar det totala antalet personer
        # # andel_sjukaa = (antal_sjuka / totalt_antal_personer * 100)

    antal_sjuka = df["disease"].sum() # beräknar antalet sjuka i originaldatasetet (alla ettor multipliceras)
    andel_sjuka = df["disease"].mean() # beräknar andelen sjuka i originaldatasetet (alla ettor mupltipliceras och delas på antalet rader totalt)

    # 2. Använd NumPy för att simulera 1000 slumpade personer med samma sannolikhet för sjukdom
    slumpgenerator = np.random.default_rng(42) # Skapar en slumptalsgenerator
    # default_rng() = NumPys rekommenderade metod för att skapa en slumptalsgenerator
    # 42 = startvärde som gör att vi får samma slumpmässiga resultat vid varje körning
    # slumpgenerator = namnet vi ger generatorn så att vi kan använda den senare i koden

    antal_personer = 800 # Anger antalet personer vi ska simulera

    simulerade_utfall = slumpgenerator.binomial( # Simulerar slumpmässiga sjukdomsutfall
        n=1, # Anger ett försök per person (personen blir antingen sjuk eller frisk)
        p=andel_sjuka, # Anger sannolikheten att bli sjuk, baserat på originaldatasetet
        size=antal_personer # Anger hur många personer som ska simuleras
    )

    # 3. Jämför den simulerade andelen med den verkliga andelen i datasetet

    antal_simulerat_sjuka = simulerade_utfall.sum() # Räknar antalet sjuka i simuleringen
    andel_simulerat_sjuka = simulerade_utfall.mean() # Beräknar andelen sjuka i simuleringen

    print(f"Antal sjuka i originaldatasetet: {antal_sjuka}")
    print(f"Andel sjuka i originaldatasetet: {andel_sjuka * 100:.2f}%")
    print(f"Antal sjuka i simuleringen: {antal_simulerat_sjuka}")
    print(f"Andel sjuka i simuleringen: {andel_simulerat_sjuka * 100:.2f}%")

    return