import scipy.stats as stats

# FUNKTION 1: HYPOTESTPRÖVNING: ”Rökare har högre medel-blodtryck än icke-rökare”

def hypothesis(df):

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
    # |t| < 1 = lågt t-värde
    # |t| ≈ 2 = börjar ofta ge statistiskt signifikant resultat
    # |t| > 2 = kan ge statistiskt signifikant resultat
    # |t| > 3 = relativt högt t-värde

    # p_val = p-värdet, används för att bedöma om resultatet är statistiskt signifikant
    # Lågt p-värde: Resultatet vore ovanligt om nollhypotesen stämde, vilket ger stöd för att förkasta den
    # Högt p-värde: Resultatet är inte särskilt ovanligt om nollhypotesen stämde, vilket innebär att vi saknar tillräckligt stöd för att förkasta den

    # En vanlig signifikansnivå är 0.05
    # p < 0.05 = statistiskt signifikant
    # p >= 0.05 = inte statistiskt signifikant

    t_stat, p_val = stats.ttest_ind(smoker_yes, smoker_no, equal_var=False)

    return f"Standard t-test: t = {t_stat:.3f}, p = {p_val:.4f}"
