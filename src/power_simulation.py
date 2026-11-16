import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import NormalIndPower  # beräknar power (teststyrka)
from statsmodels.stats.proportion import proportion_effectsize  # beräknar effektstorlek mellan andelar
from statsmodels.stats.proportion import proportions_ztest  # utför z-test för att jämföra andelar
from statsmodels.stats.multitest import multipletests  # korrigerar p-värden vid flerfaldig testning

# FUNKTION 1: STATISTISK POWER-SIMULERING

#### FUNKTIONENS STRUKTUR ####
# 1. BASELINE = hämtar grupperna ( Y och N) från vår df, bestämmer deras medelvärden och skillnaden vi vill upptäcka
# 2. BERÄKNA EFFEKTSTORLEK (COHEN'S D) = # Cohen's d används i vår kalkylator (solve_power) för att beräkna hur många personer
# som behövs i varje grupp för att kunna upptäcka den antagna skillnaden i blodtryck (min_detectable_diff = 0.4719)
# 3. BERÄKNA NÖDVÄNDIG STICKPROVSSTORLEK = med beräkningsverktyget solver = NormalIndPower() 
# 4. SKAPA EN SIMULERING FÖR ATT BERÄKNA STATISTISK STYRKA (POWER)

def simulation(df):

    # 1. BASELINE
    np.random.seed(42)                                  # gör slumpresultaten reproducerbara
    Y = df.loc[df["smoker"] == "Yes", "systolic_bp"]    # Y = rökare (smoker_yes)
    N = df.loc[df["smoker"] == "No", "systolic_bp"]     # N = icke-rökare (smoker_no)

    baseline_mean_Y = 149.5249      # Medelvärdet för rökare
    baseline_mean_N = 149.0530      # Medelvärdet för icke-rökare
    min_detectable_diff = 0.4719    # Skillnaden vi vill kunna upptäcka
    target_mean_N = baseline_mean_Y - min_detectable_diff
 
    alpha = 0.05  # Signifikansnivå: 5 %
    power_target = 0.80  # Önskad power: 80 %

    # 2. BERÄKNA EFFEKTSTORLEK (COHEN'S D)
    # Cohen's D beräknas genom att dividera skillnaden mellan gruppernas medelvärden
    # med gruppernas gemensamma standardavvikelse (pooled_sd). Det Visar hur stor skillnaden är 
    # i förhållande till spridningen
    pooled_sd = np.sqrt((Y.std()**2 + N.std()**2) / 2)
    effect_size = (baseline_mean_Y - target_mean_N) / pooled_sd # effect_size (Cohen's d) = 0.036


    # 3. BERÄKNA NÖDVÄNDIG STICKPROVSSTORLEK
    solver = NormalIndPower()   # Skapar ett beräkningsverktyg (en "kalkylator") för test med två oberoende grupper
    n_per_group = solver.solve_power(
    effect_size=effect_size,    # Effektstorleken (Cohen's d ≈ 0.036) som stickprovet ska kunna upptäcka
    alpha=alpha,                # Signifikansnivå: 0.05
    power=power_target,         # Träffsäkerhet: (80 % används som en etablerad riktlinje inom statistik)
    ratio=1.0,                  # Fördelning: 1.0 betyder exakt lika många personer i A som i B (50/50)
    alternative="larger"        # "larger" ger oss ett ensidigt test: vi undersöker om rökare (Y) har högre blodtryck än icke-rökare (N)
    )                                   

    n_per_group_rounded = int(np.ceil(n_per_group))    # avrundar uppåt till närmaste heltal, eftersom antal personer måste vara heltal

    # 4. SKAPA EN SIMULERING FÖR ATT BERÄKNA STATISTISK STYRKA (POWER)
    n_sims = 5000
    rejections = 0           # Räknare för hur många gånger vi förkastar nollhypotesen (H0) = när vår simulering upptäcker faktisk skillnad
    for _ in range(n_sims):  # repetera 5000 gånger

        # Skapar slumpmässiga blodtrycksvärden för varje person i respektive grupp
        # Värdena dras från en normalfördelning (np.random.normal) med gruppens antagna medelvärde
        # och den gemensamma standardavvikelsen (pooled_sd) från vår ursprungliga data
        Y_sim = np.random.normal(baseline_mean_Y, pooled_sd, n_per_group_rounded)
        N_sim = np.random.normal(target_mean_N, pooled_sd, n_per_group_rounded)

        _, p_val = stats.ttest_ind(     # Utför ett ensidigt Welch t-test
            Y_sim,
            N_sim,
            equal_var=False,            # var=False ger oss Welch t-test, eftersom det inte antar lika varians i grupperna
            alternative="greater"
        )

        # Om p-värdet är mindre än alpha (0.05) förkastar vi H0
        # Vi räknar hur många simuleringar som hittar en statistiskt signifikant skillnad
        rejections += (p_val < alpha) # och adderar dem till rejections

    power = rejections / n_sims     # Andelen simuleringar där vi lyckades upptäcka skillnaden
    return f"Simulerad power: {power:.4f}"