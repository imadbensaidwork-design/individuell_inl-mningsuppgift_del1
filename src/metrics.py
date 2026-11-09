from src.io_utils import load_data

# FUNKTION FÖR BESKRIVANDE ANALYS AV NUMERISKA KOLUMNER
def summary_statistics(df):
    """Beräknar statistiska mått för utvalda numeriska kolumner"""

    columns = ["age", "weight", "height", "systolic_bp", "cholesterol"] # samlar namnen på kolumnerna vi vill undersöka

    summary = df[columns].describe() # väljer kolumnerna och beräknar statistiska mått för dem

    return summary # returnerar sammanställning i form av en dataframe