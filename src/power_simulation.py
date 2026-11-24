import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize  # beräknar effektstorlek mellan andelar
from statsmodels.stats.proportion import proportions_ztest  # utför z-test för att jämföra andelar
from statsmodels.stats.multitest import multipletests  # korrigerar p-värden vid flerfaldig testning

# FUNKTION 1: STATISTISK POWER-SIMULERING
#### FUNKTIONENS STRUKTUR ####
# 1. BASELINE
# 2. BERÄKNA EFFEKTSTORLEK (COHEN'S D)
# 3. BERÄKNA TEORETISK POWER
# 4. SIMULERA POWER GENOM ATT UPPREPA HYPOTESTESTET
   
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