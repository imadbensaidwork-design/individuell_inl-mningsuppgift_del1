# FUNKTION FÖR BESKRIVANDE ANALYS AV NUMERISKA KOLUMNER

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