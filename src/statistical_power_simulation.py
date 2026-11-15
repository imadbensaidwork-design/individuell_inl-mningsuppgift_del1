import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.power import NormalIndPower  # beräknar power (teststyrka)
from statsmodels.stats.proportion import proportion_effectsize  # beräknar effektstorlek mellan andelar
from statsmodels.stats.proportion import proportions_ztest  # utför z-test för att jämföra andelar
from statsmodels.stats.multitest import multipletests  # korrigerar p-värden vid flerfaldig testning

# FUNKTIONENS ÖVERGRIPANDE UPPGIFT OCH SYFTE:
# Vi vill undersöka hur säkert vårt hypotesprov är när vi jämför blodtrycket
# mellan rökare (Y) och icke-rökare (N)
#
# Vi utgår från den observerade skillnaden mellan gruppernas medelvärden (0.4719 mmHg)
# och undersöker om vårt stickprov är tillräckligt stort för att upptäcka en sådan skillnad
#
# Vi beräknar först effektstorleken (Cohen's d) och därefter hur många personer
# som behövs i varje grupp för att uppnå 80 % power vid 5 % signifikansnivå
#
# Slutligen simulerar vi hypotesprovet för att undersöka hur ofta det upptäcker
# en verklig skillnad mellan grupperna

# FUNKTION 1: STATISTISK POWER-SIMULERING

def power_simulation(df):

    # BASELINE
    np.random.seed(42)                                  # gör slumpresultaten reproducerbara
    Y = df.loc[df["smoker"] == "Yes", "systolic_bp"]    # Y = rökare (smoker_yes)
    N = df.loc[df["smoker"] == "No", "systolic_bp"]     # N = icke-rökare (smoker_no)

    baseline_mean_Y = 149.5249      # Medelvärdet för rökare
    baseline_mean_N = 149.0530      # Medelvärdet för icke-rökare
    min_detectable_diff = 0.4719    # Skillnaden vi vill kunna upptäcka

    target_mean_N = baseline_mean_Y - min_detectable_diff   # skapar en antagen skillnad mellan grupperna som vi sedan testar
                                                            # hur ofta vårt statistiska test kan upptäcka (power)
    alpha = 0.05  # Signifikansnivå: 5 %
    power_target = 0.80  # Önskad power: 80 %

    # BERÄKNA EFFEKTSTORLEK (COHEN'S D) OCH NÖDVÄNDIG STICKPROVSSTORLEK
 
    # 1. BERÄKNA EFFEKTSTORLEKEN (COHEN'S D)
    # Cohen's D beräknas genom att dividera skillnaden mellan gruppernas medelvärden
    # med gruppernas gemensamma standardavvikelse (pooled_sd). Det Visar hur stor skillnaden är 
    # i förhållande till spridningen
    pooled_sd = np.sqrt((Y.std()**2 + N.std()**2) / 2)
    effect_size = (baseline_mean_Y - baseline_mean_N) / pooled_sd
    # effect_size (Cohen's d) = 0.036

    # 2: BERÄKNA ANTALET PERSONER SOM KRÄVS I VÅRT STICKPROV FÖR ATT UPPTÄCKA SKILLNADEN (min_detectable_diff = 0.4719)
    # Syfte: Ta reda på hur mycket data som behövs för att inte bli lurad av slumpen

    solver = NormalIndPower()   # Skapar ett beräkningsverktyg (en "kalkylator") för test med två oberoende grupper
    n_per_group = solver.solve_power(
    effect_size=effect_size,    # Effektstorleken (Cohen's d ≈ 0.036) som stickprovet ska kunna upptäcka
    alpha=alpha,                # Signifikansnivå: 0.05
    power=power_target,         # Träffsäkerhet: (80 % används som en etablerad riktlinje inom statistik)
    ratio=1.0,                  # Fördelning: 1.0 betyder exakt lika många personer i A som i B (50/50)
    alternative="larger"        # "larger" ger oss ett ensidigt test: vi undersöker om rökare (Y) har högre blodtryck än icke-rökare (N)
    )                                   

    n_per_group_rounded = int(np.ceil(n_per_group))    # avrundar uppåt till närmaste heltal, eftersom antal personer måste vara heltal