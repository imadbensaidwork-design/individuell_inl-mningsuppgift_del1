import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# FUNKTION 1: BESKRIVANDE HÄLSOANALYS AV NUMERISKA KOLUMNER

def summary_statistics(df): # funktion tar emot DataFrame
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


# FUNKTION 2: SIMULERING AV SJUKDOM

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

    antal_personer = 1000 # Anger antalet personer vi ska simulera

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


# FUNKTION 3: KONFIDENSINTERVALL NORMALAPPROXIMATION

def normal_approximation_ci(df):
    np.random.seed(42)   
    """
    95%-CI för medel med normal-approximation
    medel +- 1.96 *(std / sqrt(n))
    """
    # 1. ingredienser
    medelvärde = df["systolic_bp"].mean()
    standardavvikelse = df["systolic_bp"].std()
    antal_mätningar = len(df["systolic_bp"])

    standardfel = standardavvikelse / math.sqrt(antal_mätningar)

    # 2. Räkna ut felmarginal (för 95% konfidensnivå använder vi kritiskt z-värde 1,96 eller 2)
    z = 1.96
    felmarginal = z * standardfel

    # 3. bygg det färdiga intervallet (nedre och övre gräns)
    nedre_gräns = medelvärde - felmarginal
    övre_gräns = medelvärde + felmarginal 

    # 4. Resultat: Vi är 95% säkra på att det sanna medelvärde ligger mellan dessa värden
    # print(f"95% Konfidensintervall: Mellan {nedre_gräns:.1f} och {övre_gräns:.1f}")

    return f"Med normalapproximationen får vi konfidensintervallet {nedre_gräns:.4f} - {övre_gräns:.4f}"


# FUNKTION 4: KONFIDENSINTERVALL BOOTSTRAP

def bootstrap_ci(df, B=5000, confidence=0.95):
    np.random.seed(42)   
    # 1. Konvertera kolumnen till en ren NumPy-array och räkna stickprovsstorleken (n = 800)
    urval = np.asarray(df["systolic_bp"])
    n = len(urval)
    
    # Förhandsboka en tom array med plats för B antal medelvärden (för struktur och minneseffektivitet)
    boot_means = np.empty(B)
    
    # 2. Starta simuleringsloopen som körs B gånger (t.ex. 5000 gånger)
    for b in range(B):
        # Dra n stycken slumpmässiga värden med återläggning för att skapa ett nytt stickprov
        boot_sample = np.random.choice(urval, size=n, replace=True)
        
        # Räkna ut medelvärdet för detta stickprov och spara på indexplats 'b' i arrayen
        boot_means[b] = np.mean(boot_sample)
        
    # 3. Sista steget: Sortera de 5000 medelvärdena och kapa de yttersta 2.5% i varje ände 
    # för att få fram ett 95%-igt konfidensintervall (lämnar de 95% i mitten)
    ci = np.percentile(boot_means, [2.5, 97.5])
    
    # Returnera det färdiga intervallet
    return f"Med bootstrapmetoden får vi konfidensintervallet {ci[0]:.4f} - {ci[1]:4f}"

# FUNKTION 5: KORRELATIONSANALYS (ÅLDER & SYSTOLISKT BLODTRYCK)
def correlation(df):
    X = df["age"].values # hämtar värdena från "age" och omvandlar dem till en np-array
    Y = df["systolic_bp"].values  # hämtar värdena från "systolic_bp" och omvandlar dem till en np-array
    r = np.corrcoef(X, Y) # beräknas korrelationsmatris
    return (f"Korrelationen mellan ålder och systoliskt blodtryck är: r = {r[0, 1]:.1f}") # returnerar korrelationen mellan X och Y


# FUNKTION 6: MULTIPEL REGRESSIONSANALYS

# Källa: https://www.youtube.com/watch?v=KYNuzfn5Fx0
# Source - https://stackoverflow.com/a/40902719
# Posted by piRSquared
# Retrieved 2026-11-20, License - CC BY-SA 3.0

def multiple_regression(df):
   
    # (Y) beroende variabel: systolic_bp
    # (X) oberoende variabler: age, weight, cholesterol, smoking, disease, sex

    # samtliga oberoende variabler är numeriska/kontinuerliga utom sex (M/F) och smoking (Yes/No)
    # dessa två behöver konverteras från strings till numeriska variabler (1/0)

    c_smoker = df["smoker"].map({"Yes": 1, "No": 0})    # omvandlar (convert) smoker från Yes/No till 1/0
    c_sex = df["sex"].map({"M": 1, "F": 0})             # omvandlar (convert) sex från M/F till 1/0
    df_drop = df.drop(columns=["id", "smoker","sex"])   # tar bort originalkolumnerna
    df_assign = df_drop.assign(smoker=c_smoker, sex=c_sex)  # lägger till den omkodade kolumnerna
    df_processed = df_assign

    # plockar ut variablerna till X (oberoende)
    X = df_processed[["age", "height", "weight", "cholesterol", "disease", "smoker", "sex"]]

    # plockar ut variablerna till Y (beroende)
    Y = df_processed["systolic_bp"]

    # lägger till en konstantkolumn (med värdet 1) i X
    # konstanten gör att modellen kan beräkna ett grundvärde (intercept)
    # som blir modellens konstant = fasta stardvärde: Blodtryck = 99.9181
    # + (0.5235 * ålder)
    # + (0.0533 * längd)
    # + (0.1603 * vikt)
    X_const = sm.add_constant(X)

    # sm.OLS() = Ordinary Least Squares (minsta kvadratmetoden)
    # Skapar en linjär regressionsmodell som beräknar vilka koefficienter som bäst passar datan
    # .fit() = genomför beräkningen och anpassar modellen till datan genom att hitta de bästa koefficienterna
    model = sm.OLS(Y, X_const).fit()

    # beräknar modellens koefficienter och andra statistiska resultat
    # visar bland annat koefficienter, p-värden och förklaringsgrad
    return (model.summary())