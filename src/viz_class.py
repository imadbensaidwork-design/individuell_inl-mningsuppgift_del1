# DENNA MODUL INNEHÅLLER EN KLASS (GRAPHS) MED METODER FÖR GRAFISK VISUALISERING

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Graphs:
    """
    Denna klass innehåller metoder för grafisk visualisering av analyser
    baserade på vårt hälsodata, anpassade för det aktuella datasetet.
    DataFrame skickas in när ett objekt av klassen skapas och sparas
    för att kunna användas av klassens olika metoder.

    Metoder:
    - hist(): Skapar ett histogram som visar frekvensfördelningen av
      systoliskt blodtryck (systolic_bp).

    - box(): Skapar ett låddiagram som jämför viktfördelningen
      mellan män och kvinnor.

    - bar(): Skapar ett stapeldiagram som visar andelen rökare
      bland män respektive kvinnor.
    """

    def __init__(self, df):
        self.df = df

    def hist(self): # METOD 1: HISTOGRAM ÖVER BLODTRYCK
        """
        Denna metod tar fram ett histogram
        som visar frekvensfördelningen över blodtryck
        bland personerna i vårt dataset.
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.df["systolic_bp"], bins=25) # väljer kolumnen systolic_bp (blodtryck)
        ax.set_title("Frekvensfördelning av blodtryck")
        ax.set_xlabel("Blodtryck")
        ax.set_ylabel("Antal") # antal individere
        ax.grid(True, axis="y") # lägger till rutnät längs Y-axeln (vågräta linjer)
        plt.tight_layout()
        return fig
    
    def box(self): # METOD 2: # LÅDDIAGRAM ÖVER VIKT PER KÖN
        """
        Denna metod tar fram ett låddiagram
        över vikt per kön, inkl. median, min- och maxvikt
        samt potentiella extremvärden (outliers).
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        weight_m = self.df.loc[self.df["sex"] == "M", "weight"] # hämtar vikterna för män
        weight_f = self.df.loc[self.df["sex"] == "F", "weight"] # hämtar vikterna för kvinnor
        ax.boxplot( # tar fram ett låddiagram
        [weight_m, weight_f], # skickar in vikter för män och kvinnor
        patch_artist=True, # gör lådorna möjliga att färglägga
        boxprops=dict(facecolor="lightblue"), # färgar lådorna ljusblå
        medianprops=dict(color="red") # färgar medianlinjen röd
        )

        ax.set_title("Vikt per kön")
        ax.set_xlabel("Kön")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["M", "F"])
        ax.set_ylabel("Vikt (kg)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        return fig

    def bar(self): # METOD 3: STAPELDIAGRAM ÖVER ANDELEN RÖKARE PER KÖN
        """
        Denna metod tar fram ett stapeldiagram
        som visar andelen rökare per kön i vårt dataset.
        """
    
        fig, ax = plt.subplots(figsize=(8, 5))

        # 1. Filtrerar DataFrame med ett villkor
        # self.df["smoker"] == "Yes" skapar en True/False-kontroll för varje rad
        # self.df.loc använder kontrollen för att välja endast rader där villkoret är True
        # "sex" efter kommatecknet anger att vi vill hämta just den kolumnen
        smoker_yes = self.df.loc[self.df["smoker"] == "Yes", "sex"]

        # 2. Väljer ut två kolumner från ursprungliga DataFrame
        # Dubbla hakparenteser används eftersom vi väljer flera kolumner
        # Resultatet blir en ny, mindre DataFrame
        smoker_total = self.df[["smoker", "sex"]]

        # 3. Räknar hur många gånger varje unikt värde förekommer
        # value_counts() räknar hur många M respektive F som finns bland rökarna
        smoker_yes = smoker_yes.value_counts()

        # 4. Grupperar efter kön och räknar antalet rader i varje grupp
        # groupby("sex") delar upp personerna i grupperna M och F
        # size() räknar antalet rader i varje grupp
        smoker_total = smoker_total.groupby("sex").size()
        # Exempel på resultat:
        # M    214  # Totalt antal män
        # F    183  # Totalt antal kvinnor
    
        smoker_rate = (smoker_yes / smoker_total * 100) # jag delar antalet rökare med totalen (smoker == Yes och == No)
                                                        # Pandas matchar automatiskt könen via deras index, inte via radposition

        # smoker_rate.index (x-axeln) hämtar indexet från Series, alltså kategorierna (M och F)
        # smoker_rate.values (y-axeln) hämtar själva värdena, alltså andelen rökare i procent för respektive kön
        # ax.bar använder kategorierna på X-axeln och procentvärdena som stapelhöjd på Y-axeln
        ax.bar(smoker_rate.index, smoker_rate.values)
        ax.set_title("Andel rökare per kön")
        ax.set_xlabel("Kön") 
        ax.set_ylabel("Rökare (%)")
        ax.grid(True, axis="y")
        plt.tight_layout()  
        return fig

    def linear_regression(self):
        """
        Denna metod tar fram ett spridningsdiagram samt en regressionslinje
        som visar sambandet mellan ålder och systoliskt blodtryck
        """
        np.random.seed(42)
        # 1. beräkna intercept och lutning
        X = self.df["age"].values # hämtar värdena från "age" och omvandlar dem till en np-array
        Y = self.df["systolic_bp"].values  # hämtar värdena från "systolic_bp" och omvandlar dem till en np-array

        # 1.1 bygg upp designmatrisen (design_matrix): [1, x]
        ones_column = np.ones_like(X) # skapar en np array fylld med ettor, med samma form och storlek som X
        design_matrix = np.column_stack([ones_column, X]) # np.column_stack() staplar ihop flera arrayer som kolumner bredvid varandra
        transposed_design = design_matrix.T # transponerar design_matrix: rader blir kolumner
        # [1, 2],       [1, 1, 1]
        # [1, 5],  -->  [2, 5, 8]
        # [1, 8]]
        gram_matrix = transposed_design @ design_matrix # multiplicerar transponerade designmatrisen med originalmatrisen (X.T @ X)
        # [1, 1, 1]     [1, 2]     [3, 15]
        # [2, 5, 8]   @ [1, 5]  =  [15, 93]
        #               [1, 8]
        design_response = transposed_design @ Y # multiplicerar transponerade designmatrisen (X.T) med Y (systolic_bp)
        beta_hat_vector = np.linalg.inv(gram_matrix) @ design_response # centrala beräkningen där vi faktiskt räknar fram regressionsmodellens koefficienter (intercept och lutning)
        intercept_hat = float(beta_hat_vector[0]) # Hämtar det första värdet (index 0), alltså interceptet (Y när X = 0)
        slope_hat = float(beta_hat_vector[1]) # Hämtar det andra värdet (index 1), alltså lutningen (k)

        print(f" Skattning av hur mycket systoliskt blodtryck (Y) ökar med åldern i år (X): {slope_hat:.2f} mmHg") # modellens lutning: hur mycket blodtrycket (Y) ökar med åldern(X)

        # 2. ta fram figur
        plt.figure(figsize=(8, 6))
        plt.scatter(X, Y, alpha=0.7, label="Data")

        grid_X = np.linspace(18, 90, 200) # skapar 200 nya, jämnt fördelade åldersvärden för att rita en jämn regressionslinje
        # 18 och 90 är datasetets min- och maxåldrar
                                          
        grid_Y_manual = intercept_hat + slope_hat * grid_X
        # Räknar ut vilket Y-värde regressionslinjen har vid vart och ett av X-värdena
        # Använder interceptet och lutningen som vi räknade fram tidigare
    
        # 2.1 ta fram linje
        plt.plot(grid_X, grid_Y_manual, color="red", linewidth=2, label="Regressionslinje")
        plt.xlabel("Ålder (X)")
        plt.ylabel("Systoliskt blodtryck (Y)")
        plt.title("Samband mellan ålder och systoliskt blodtryck (linjär regression)")
        plt.legend()
        plt.show()