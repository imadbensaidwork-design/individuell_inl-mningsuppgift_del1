# DENNA MODUL INNEHÅLLER EN KLASS (GRAPHS) MED METODER FÖR GRAFISK VISUALISERING

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Graphs:
    """
    Denna klass innehåller metoder för grafisk visualisering av analyser
    baserade på vårt hälsodata, anpassade för det aktuella datasetet.
    DataFrame skickas in när ett objekt av klassen skapas och sparas
    för att kunna användas av klassens olika metoder.
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
        ax.set_title("1. Frekvensfördelning av blodtryck")
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

        ax.set_title("2. Vikt per kön")
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
        ax.set_title("3. Andel rökare per kön")
        ax.set_xlabel("Kön") 
        ax.set_ylabel("Rökare (%)")
        ax.grid(True, axis="y")
        plt.tight_layout()  
        return fig

    def linear_regression(self): # METOD 4: SPRIDNIGNSDIAGRAMM OCH REGRESSIONSLINJE
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

    def correlation_matrix(self): # METOD 5: KORRELATIONSMATRIS
        # Källa: https://stackoverflow.com/a/29432741
        """
        Denna metod tar fram en korrelationsmatris som visar sambandet mellan
        datasetets numeriska variabler: ålder, längd, vikt, systoliskt blodtryck 
        och kolesterol.
        """

        korrelationsdata = self.df[["age", "height", "weight", "systolic_bp", "cholesterol"]]

        figur = plt.figure(figsize=(8,6))
        plt.matshow(korrelationsdata.corr(), fignum=figur.number) # skapar korrelationsmatrisen
        korrelationsmatris = korrelationsdata.corr() # hämtar korrelationsvärdena från korrelationsdata.corr()

        for i in range(korrelationsmatris.shape[0]): # Skriver ut korrelationsvärdet i varje ruta
            for j in range(korrelationsmatris.shape[1]):
                plt.text(j, i, f"{korrelationsmatris.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black")

        plt.xticks(range(korrelationsdata.shape[1]), korrelationsdata.columns, fontsize=14, rotation=45) # namnger x-axeln
        plt.yticks(range(korrelationsdata.shape[1]), korrelationsdata.columns, fontsize=14) # namnger y-axeln

        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title("Korrelationsmatris", fontsize=16);
        plt.show();


    def coefplot(self): # METOD 6: MULTIPEL REGRESSIONSANALYS OCH KOEFFICIENTDIAGRAM
        # källa: https://medium.com/analytics-vidhya/create-your-own-coefficient-plot-function-in-python-aadb9fe27a77
        """
        Denna metod utför en multipel regressionsanalys och presenterar sedan resultaten i en
        tabell och i ett kompletterande koefficientdiagram. Den beroende variabeln i analysen 
        är systoliskt blodtryck. De oberoende variablerna är ålder, längd, vikt, kolesterol,
        sjukdom, rökare och kön.
            
        """
        c_smoker = self.df["smoker"].map({"Yes": 1, "No": 0}) # omvandlar (convert) smoker från Yes/No till 1/0
        c_sex = self.df["sex"].map({"M": 1, "F": 0})
        self.df_drop = self.df.drop(columns=["id", "smoker","sex"])      # tar bort originalkolumnen
        self.df_assign = self.df_drop.assign(smoker=c_smoker, sex=c_sex) # lägger till den omkodade kolumnen
        self.df_processed = self.df_assign

        # plockar ut variablerna til X och Y
        X = self.df_processed[["age", "height", "weight", "cholesterol", "disease", "smoker", "sex"]] # oberoende variabler
        Y = self.df_processed["systolic_bp"] # beroende variabel

        # lägger till en konstantkolumn (med värdet 1) i X
        # konstanten gör att modellen kan beräkna ett grundvärde (intercept)
        # som blir modellens konstant = fasta stardvärde: Blodtryck = 99.9181
        X = sm.add_constant(X)

        # sm.OLS() = Ordinary Least Squares (minsta kvadratmetoden)
        # Skapar en linjär regressionsmodell som beräknar vilka koefficienter som bäst passar datan
        # .fit() = genomför beräkningen och anpassar modellen till datan genom att hitta de bästa koefficienterna
        model = sm.OLS(Y, X).fit()

        # meräknar och skriver ut modellens MSE (Mean Squared Error)
        # MSE visar det genomsnittliga kvadrerade felet mellan modellens
        # förutsagda blodtrycksvärden och de faktiska blodtrycksvärdena
        model.mse_resid

        # tabellens resultat
        model.summary()

        # förbered figuren

        coef_df = pd.DataFrame(model.summary().tables[1].data) # hämtar koefficienttabellen från regressionen och gör om den till en Pandas-tabell
        coef_df.columns = coef_df.iloc[0]                 # tar första raden och gör den till kolumnnamn, till exempel coef, std err och P>|t|
        coef_df=coef_df.drop(0)                           # tar bort första raden eftersom den nu redan används som kolumnnamn
        coef_df = coef_df.set_index(coef_df.columns[0])   # gör första kolumnen till radnamn, så att variablerna (age, weight osv.) blir index
        coef_df = coef_df.astype(float)                   # gör om tabellens värden till decimaltal så att Python kan räkna med dem
        errors_lower = coef_df["coef"] - coef_df["[0.025"] # räknar ut hur långt det är från koefficienten till den nedre konfidensgränsen
        errors_upper = coef_df["0.975]"] - coef_df["coef"]  # räknar ut hur långt det är från koefficienten till den övre konfidensgränsen
        coef_df["errors_lower"] = errors_lower               # skapar en ny kolumn som lägger till de uträknade avstånden i DataFrame
        coef_df["errors_upper"] = errors_upper               # skapar en ny kolumn som lägger till de uträknade avstånden i DataFrame
        coef_df = coef_df.drop(["const"])                    # tar bort konstanten (interceptet) från tabellen eftersom vi bara vill visa våra
                                                            # förklarande variabler i koefficientdiagrammet, inte modellens startvärde.
        coef_df = coef_df.sort_values(by=["coef"])        # sorterar variablerna efter deras koefficientvärde, från lägsta till högsta, så att 
                                                        # de visas i den ordningen i koefficientdiagrammet
        # skriv ut koefficienterna 
        variables = list(coef_df.index.values)            # hämtar alla variabelnamn från tabellen och sparar dem i en lista
        coef_df['variables'] = variables                  # lägger till en kolumn med variables (etiketter smoker, sex, osv..)
        sns.set_context("poster")                         # Gör text och diagrammets detaljer större - ta ev. bort..

        fig, ax = plt.subplots(figsize=(8, 6))           # ta fram figuren

        # error-stapel för 95% konfidensintervall (tas fram med pandas(plot))
        coef_df.plot(
            x="variables",       # Visar variabelnamnen på X-axeln
            y="coef",             # Visar koefficientvärdena på Y-axeln
            kind="bar",           # Väljer stapeldiagram
            ax=ax,                # Placerar diagrammet på en befintlig Matplotlib-axel
            color="none",         # Gör staplarna genomskinliga
            fontsize=16,          # Bestämmer textstorleken
            ecolor="steelblue",   # Gör felstaplarna blå
            capsize=3,            # Tar bort små tvärstreck i felstaplarnas ändar
            yerr=[coef_df["errors_lower"], coef_df["errors_upper"]], # Anger felmarginalerna som ska visas
            legend=False          # döljer legend
            )

        # titlar och etiketter
        plt.title("Koefficienter av hälsodata / 95% Konfidensintervall")
        ax.set_ylabel("Koefficienter")
        ax.set_xlabel("Variabler")

        # koefficienter
        ax.scatter(
            x=np.arange(coef_df.shape[0]),  # skapar en position på X-axeln för varje variabel i tabellen
            marker="o",                     # visar varje koefficient som en rund punkt
            s=16,                           # bestämmer punktens storlek
            y=coef_df["coef"],              # placerar punkterna utifrån variablernas koefficientvärden på Y-axeln
            color="steelblue"               # Gör punkterna blå
        )

        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # ritar en referenslinje vid värdet 0 på Y-axeln
        ax.axhline(y=0, linestyle='--', color='red', linewidth=1)

        # skriver ut tabellen och grafen
        print(model.summary())
        plt.show();