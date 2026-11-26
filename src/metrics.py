import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize  # beräknar effektstorlek mellan andelar
from statsmodels.stats.proportion import proportions_ztest      # utför z-test för att jämföra andelar
from statsmodels.stats.multitest import multipletests           # korrigerar p-värden vid flerfaldig testning

# FUNKTION 1: BESKRIVANDE HÄLSOANALYS AV NUMERISKA KOLUMNER
def summary_statistics(df): # funktion tar emot DataFrame
    """   
    Beräknar beskrivande statistiska mått för utvalda numeriska kolumner
    i datasetet.

    Funktionen sammanställer medelvärde, median, minimum och maximum
    för ålder, vikt, längd, systoliskt blodtryck och kolesterol.
    Resultaten avrundas till ett lämpligt antal decimaler för varje kolumn.
    """
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
    """ 
    Simulerar sjukdomsutfall för 1000 slumpmässiga personer baserat
    på sjukdomsförekomsten i originaldatasetet.

    Funktionen beräknar först antalet och andelen personer med sjukdomen
    i originaldatasetet. Därefter används NumPy för att simulera
    sjukdomsutfall där sannolikheten att bli sjuk motsvarar den
    observerade andelen i datasetet.

    Slutligen jämförs antalet och andelen sjuka i simuleringen
    med originaldatasetet.
    """
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
    Beräknar ett 95-procentigt konfidensintervall för medelvärdet
    av det systoliska blodtrycket med hjälp av normalapproximation.

    Funktionen beräknar medelvärde, standardavvikelse och standardfel.
    Därefter används det kritiska z-värdet 1,96 för att beräkna
    felmarginalen och konfidensintervallets nedre och övre gräns.
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
    """
    Beräknar ett konfidensintervall för medelvärdet av det systoliska
    blodtrycket med hjälp av bootstrapmetoden.

    Funktionen drar upprepade stickprov med återläggning från
    originaldatasetet och beräknar medelvärdet för varje stickprov.
    De simulerade medelvärdena används sedan för att beräkna
    konfidensintervallet med hjälp av percentilmetoden.
    """
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

# FUNKTION 5: HYPOTESTPRÖVNING: ”Rökare har högre medel-blodtryck än icke-rökare”
def hypothesis(df):
    """
    Undersöker om rökare har ett högre genomsnittligt systoliskt
    blodtryck än icke-rökare genom att utföra ett ensidigt Welch t-test.

    Funktionen delar upp datasetet i två oberoende grupper:
    rökare och icke-rökare. Därefter jämförs gruppernas medelvärden
    för systoliskt blodtryck med hjälp av Welch t-test, som inte
    förutsätter att grupperna har lika varians.

    Hypotesprövningen använder en signifikansriktning där alternativ-
    hypotesen är att rökare har högre genomsnittligt systoliskt
    blodtryck än icke-rökare.
    """
    # 1. ta reda på gruppernas skillnad i medelvärde
    smoker_yes = df.loc[df["smoker"] == "Yes", "systolic_bp"]
    smoker_no = df.loc[df["smoker"] == "No", "systolic_bp"]

    smoker_yes.mean() # 149.524882629108 # rökare
    smoker_no.mean() # 149.05298126064736 # icke-rökare

    # 2. testa hypotesen: "rökare har högre medel-blodtryck än icke-rökare"

    # 2.1 welch t-test (jämför två oberoende grupper utan krav på lika varians)

    # stats.ttest_ind = utför ett t-test för två oberoende grupper
    # smoker_yes och smoker_no = grupperna som jämförs (rökare och icke-rökare)
    # equal_var=False = tillåter olika varians (spridning) i grupperna

    # t_stat = t-statistiken som visar skillnaden mellan gruppernas medelvärden i förhållande till osäkerheten

        # Osäkerheten i skillnaden mellan gruppernas medelvärden påverkas av:
        # 1.  Spridningen: Större variation i blodtrycket ger större osäkerhet
        # 1.1 Mindre variation i blodtrycket ger mindre osäkerhet
        # 2.  Antalet personer: Fler personer ger mindre osäkerhet
        # 2.1 Färre personer ger större osäkerhet

    # Ett högt absolut t-värde innebär att skillnaden är stor i förhållande till osäkerheten
    # Ett lågt absolut t-värde innebär motsatsen
    # Det finns ingen exakt gräns för vad som räknas som ett högt eller lågt t-värde
    
    # Grov tumregel:
    # t < 1 = lågt t-värde
    # t ≈ 2 = börjar ofta ge statistiskt signifikant resultat
    # t > 2 = kan ge statistiskt signifikant resultat
    # t > 3 = relativt högt t-värde

    # p_val = p-värdet, används för att bedöma om resultatet är statistiskt signifikant
    # Lågt p-värde: Resultatet vore ovanligt om nollhypotesen stämde, vilket ger stöd för att förkasta den
    # Högt p-värde: Resultatet är inte särskilt ovanligt om nollhypotesen stämde, vilket innebär att vi saknar tillräckligt stöd för att förkasta den

    # En vanlig signifikansnivå är 0.05
    # p < 0.05 = statistiskt signifikant
    # p >= 0.05 = inte statistiskt signifikant
    t_stat, p_val = stats.ttest_ind(smoker_yes, smoker_no, equal_var=False, alternative="greater")

    return f"Welch t-test: t = {t_stat:.3f}, p = {p_val:.4f}"


# FUNKTION 6: KORRELATIONSANALYS (ÅLDER & SYSTOLISKT BLODTRYCK)
# källa: https://realpython.com/numpy-scipy-pandas-correlation-python/#example-numpy-correlation-calculation
def correlation(df):
    """
    Beräknar Pearsons korrelationskoefficient mellan ålder
    och systoliskt blodtryck.

    Funktionen hämtar värdena från de två kolumnerna och använder
    NumPy för att beräkna korrelationen. Resultatet visar styrkan
    och riktningen på det linjära sambandet mellan variablerna.
    """
    X = df["age"].values # hämtar värdena från "age" och omvandlar dem till en np-array
    Y = df["systolic_bp"].values  # hämtar värdena från "systolic_bp" och omvandlar dem till en np-array
    r = np.corrcoef(X, Y) # beräknas korrelationsmatris
    return (f"Korrelationen mellan ålder och systoliskt blodtryck är: r = {r[0, 1]:.1f}") # returnerar korrelationen mellan X och Y

# FUNKTION 7: MULTIPEL REGRESSIONSANALYS
# Källa: https://www.youtube.com/watch?v=KYNuzfn5Fx0
# Source - https://stackoverflow.com/a/40902719
# Posted by piRSquared
# Retrieved 2026-11-20, License - CC BY-SA 3.0

def multiple_regression(df):
    """
    Genomför en multipel linjär regressionsanalys för att undersöka
    sambandet mellan flera förklarande variabler och systoliskt blodtryck.

    Funktionen omkodar först de kategoriska variablerna 'smoker'
    och 'sex' till numeriska värden. Därefter används ålder, längd,
    vikt, kolesterol, sjukdom, rökning och kön som oberoende variabler.
    Systoliskt blodtryck används som beroende variabel.

    Med hjälp av statsmodels och minsta kvadratmetoden (OLS) anpassas
    en multipel linjär regressionsmodell. Funktionen sammanställer
    modellens resultat, bland annat regressionskoefficienter,
    p-värden och förklaringsgrad.
    """
   
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

# FUNKTION 8: STATISTISK POWER-SIMULERING
# Källa: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html?utm_source=chatgpt.com
# Källa: Joakim Lind, Power och flerfaldig testning https://hermods.omniway.se/studentcourselessons/34430261

def simulation(df):
    """
    Beräknar teoretisk och simulerad statistisk power för att undersöka
    hur ofta ett hypotesprövningstest upptäcker en skillnad i genomsnittligt
    systoliskt blodtryck mellan rökare och icke-rökare.

    Först beräknas gruppernas medelvärden, den observerade skillnaden
    och effektstorleken (Cohen's d). Därefter beräknas den teoretiska
    powern med hjälp av TTestIndPower().

    Slutligen genomförs 5000 simuleringar där slumpmässiga blodtrycksvärden
    genereras utifrån gruppernas medelvärden och standardavvikelser.
    För varje simulering utförs Welch's t-test för att uppskatta den
    simulerade powern.
    
    """
    # 1. BASELINE
    np.random.seed(42)                                  # Gör slumpresultaten reproducerbara
    Y = df.loc[df["smoker"] == "Yes", "systolic_bp"]    # Y = rökare
    N = df.loc[df["smoker"] == "No", "systolic_bp"]     # N = icke-rökare

    mean_Y = Y.mean()                          # Medelvärdet för rökare
    mean_N = N.mean()                          # Medelvärdet för icke-rökare
    min_detectable_diff = mean_Y - mean_N      # Faktisk skillnad mellan grupperna som vi vill upptäcka: 0.4719
    alpha = 0.05                               # Signifikansnivå: 5 %

    # 2. BERÄKNA EFFEKTSTORLEK (COHEN'S D, FÖR KONTINUERLIGA VÄRDEN)
    # Dividerar skillnaden mellan gruppernas medelvärden med gemensam standardavvikelse
    pooled_sd = np.sqrt((Y.std()**2 + N.std()**2) / 2)  # Beräknar en gemensam standardavvikelse för grupperna
    effect_size = min_detectable_diff / pooled_sd       # Effektstorlek (Cohen's d)
    # Cohen's d = 0.036

    # 3. BERÄKNA TEORETISK POWER (MATEMATISK BERÄKNING)
    # Matematisk beräkning av power med hjälp av TTestIndPower().
    # Här används: Effektstorleken (Cohen's d), antal personer i grupperna och signifikansnivån (alpha)

    # TTestIndPower() hör till statsmodels (scipy) och skapar ett beräkningsverktyg för att beräkna power för ett t-test mellan två oberoende grupper
    analysis = TTestIndPower()

    # power = variabeln med resultatet av power-beräkningen
    # .power() beräknar matematiskt sannolikheten att testet upptäcker en skillnad
    # om den antagna effektstorleken faktiskt finns i populationen
    power = analysis.power(
        # effect_size = Cohen's d = 0.036
        effect_size=effect_size,

        # nobs1 = number of observations
        # Anger antalet observationer i grupp Y (rökare)
        nobs1=len(Y),

        # ratio = förhållandet mellan gruppernas stickprovsstorlekar
        # len(N) / len(Y) anger hur stor grupp N är i förhållande till grupp Y
        # Rökare (Y): 213 personer
        # Icke-rökare (N): 587 personer
        # ration är 2,75. Det betyder att det finns ungefär 2,76 gånger så många icke-rökare som rökare i datan
        ratio=len(N) / len(Y),

        # alpha = signifikansnivå
        alpha=alpha
    )

    print(f"Observerad skillnad i medelvärden mellan grupperna: {min_detectable_diff:.4f}")
    print(f"Effektstorlek (Cohen's d): {effect_size:.4f}")
    print(f"Teoretisk Power: {power * 100:.1f}%")

    # 4. SIMULERA POWER GENOM ATT UPPREPA HYPOTESTESTET
    n_sims = 5000
    rejections = 0           # Räknare för hur många gånger vi förkastar nollhypotesen (H0) = när vår simulering upptäcker faktisk skillnad
    for _ in range(n_sims):  # repetera 5000 gånger

        # Skapar slumpmässiga blodtrycksvärden för varje person i respektive grupp, baserat på våra faktiska gruppstorlekar =  len(Y), len(N)
        # Värdena dras från en normalfördelning (np.random.normal) med gruppernas antagna medelvärden
        # och den gemensamma standardavvikelsen (pooled_sd) från vår ursprungliga data
        Y_sim = np.random.normal(mean_Y, Y.std(), len(Y))
        N_sim = np.random.normal(mean_N, N.std(), len(N))

        _, p_val = stats.ttest_ind(     # Utför ett tvåsidigt Welch t-test
            Y_sim,
            N_sim,
            equal_var=False,            # Welch t-test, som inte förutsätter lika varians
            alternative="two-sided"     # ger oss ett tvåsidigt test.. som undersöker om gruppernas medelvärden skiljer sig åt
        )

        # Om p-värdet är mindre än alpha (0.05) förkastar vi H0
        # Vi räknar hur många simuleringar som hittar en statistiskt signifikant skillnad
        rejections += (p_val < alpha) # och adderar dem till rejections

    sim_power = rejections / n_sims     # Andelen simuleringar där vi lyckades upptäcka skillnaden
    print(f"Simulerad Power: {sim_power * 100:.1f}%")