import pandas as pd

# FUNKTION FÖR ATT LÄSA IN CSV-FIL
def load_data(file_path):  # Tar emot sökvägen till CSV-filen
    """Läser in en CSV-fil och returnerar en pandas DataFrame.

    Parametrar:
    file_path (str): Sökvägen till CSV-filen.
    """
    df = pd.read_csv(file_path)  # Läser in CSV-filen till en DataFrame
    return df  # Returnerar färdig DataFrames