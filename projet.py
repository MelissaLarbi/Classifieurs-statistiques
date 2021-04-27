# *************************************************************************************************************************
# Projet 2: APPRENTISSAGE STATISTIQUE
# Réalisé par les étudiants:
#                           LARBI Melissa 3971141
#                           ZHOU Jean-Marc 28008879
# *************************************************************************************************************************

import utils
from utils import AbstractClassifier
import math
from collections import defaultdict
from graphviz import Digraph
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


# ******************************** Question 01 *****************************************

def getPrior(train):
    """
    Calcule la probabilité a priori de la classe 1 
    ainsi que l'intervalle de confiance à 95% 
    pour l'estimation de cette probabilité.
    
    Args:
        train (DataFrame): notre base de donnés.

    Returns:
        dict: estimation, estimation, max5pourcent
    """
    estimation = train['target'].sum() / train['target'].size  # la moyenne
    variance = estimation * (1 - estimation)  # la variance de l'échantillion
    # result ={
    #         'estimation': estimation, 
    #         'min5pourcent': estimation - 1.96 * math.sqrt(variance/train['target'].size), 
    #         'max5pourcent': estimation + 1.96 * math.sqrt(variance/train['target'].size)
    #     }
    return {'estimation': estimation,'min5pourcent': estimation - 1.96 * math.sqrt(variance/train['target'].size),'max5pourcent': estimation + 1.96 * math.sqrt(variance/train['target'].size)
        }

# ******************************** Question 02  ********************************************

class APrioriClassifier(AbstractClassifier):
    """
    Ce classifieur est naif il retourn toujours la classe majouritaire (a priori).

    Args:
        AbstractClassifier (object): Un classifier abtrait.

    """
    def __init__(self):
        AbstractClassifier.__init__(
            self)  # comme c'est une classe fille on se contante d'appeller le constructeur père

    def estimClass(self, vecteur):
        """
        Estime la classe majoritaire.

        Args:
            vecteur (dict): Représente un individu avec les attributs et les valeur associées il peut être n'import quoi.

        Returns:
            int: soit 0 ou 1 qui est la classe auquel appartien l'individu.
        """
        return 1

    def statsOnDF(self, train):
        """
        Renvoie les valeurs VP, VN, FP, FN, la précision le rappel.

        Args:
            train (DataFrame): notre base de données.

        Returns:
            dict: VP : nombre d'individus avec target=1 et classe prévue=1 
                  VN : nombre d'individus avec target=0 et classe prévue=0
                  FP : nombre d'individus avec target=0 et classe prévue=1
                  FN : nombre d'individus avec target=1 et classe prévue=0
                  Précision : combien de candidats sélectionnés sont pertinents (VP/(VP+FP))
                  Rappel : combien d'éléments pertinents sont sélectionnés (VP/(VP+FN))
        """
        dico = {}
        dico["VP"] = dico["VN"] = dico["FP"] = dico["FN"] = 0
        for _, t in train.iterrows():
            res = self.estimClass(dict(t))
            if(t.target == 1):
                if(res == 1):
                    dico["VP"] += 1
                else:
                    dico["FN"] += 1

            if(t.target == 0):
                if(res == 1):
                    dico["FP"] += 1
                else:
                    dico["VN"] += 1

        dico["Précision"] = dico["VP"]/(dico["VP"] + dico["FP"])
        dico["Rappel"] = dico["VP"] / (dico["VP"] + dico["FN"])
        return dico


# *********************************** Question 03 ******************************************************

# ******* Question 3.a **********************************************************************************

def probaXsachatTarget(df, valeur_attr, target, attr):
    """
    Renvoie la valeur de P(attr=valeur_attr|target)

    Args:
        df (DataFrame): notre base de donnees. 
        valeur_attr (int):la valeur de l'attribut. 
        target (int): la classe voulu. 
        attr (str): l'attribut voulu.
    
    Returns:
        float: La probabilité, entre 0 et 1
    """

    
    n = df.shape[0]  # la taille de df
    nb = 0  # P(attr=X and t)
    nbT = 0
    for tuple in df.itertuples():
        dictio = tuple._asdict()
        if(dictio['target'] == target):
            nbT = nbT+1
            if(dictio[attr] == valeur_attr):
                nb = nb+1
    p_X_sachat_t = nb/n
    p_t = nbT/n
    return p_X_sachat_t/p_t


def P2D_l(df, attr):
    """
    Calcule P(attr=A|target) pour tous les A unique de l'attrbut.

    Args:
        df (DataFrame): notre base de données.
        attr (str): L'attribut du df pour qui on charche la probabilté sachant target.

    Returns:
        dict: dict associe à toute les valeur de attr leur proba sachant target.
    """
    ens = set(
        df[attr])  # on calcule l'ensemble des valeur distincte de l'attribut dans le dataframe
    dict = {}  # un dictionnaire dont les valeurs est des dictionnaires
    for t in {1, 0}:
        dict[t] = {x: probaXsachatTarget(df, x, t, attr) for x in ens}
    return dict


def P2D_p(df, attr):
    """
    Calcule 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡=𝑡|𝑎𝑡𝑡𝑟=𝑎) pour différentes valeur de l'attribut.

    Args:
        df (DataFrame): notre base de données.
        attr (str): L'attribut consérné pour la probabilité.  

    Returns:
        dict: dict associe à toute les valeur de target leur proba sachant les valeur de l'attribut.
    """
    # tirer les valeurs unique que puet prendre l'attribut attr du dataframe
    values = pd.crosstab(df["target"], df[attr]).values
    N_0 = np.sum(values[0])
    N_1 = np.sum(values[1])
    dicG = dict()
    for i in np.unique(df[attr].values):
        dat = df[(df[attr] == i)]['target'].values
        N = len(dat)
        N_0 = len(np.where(dat == 0)[0])
        N_1 = len(np.where(dat == 1)[0])
        dic = dict()
        dic[1] = N_1/(1.0*N)
        dic[0] = N_0/(1.0*N)
        dicG[i] = dic
    return dicG

# ******* Question 3.b **************************************************************************************


class ML2DClassifier(APrioriClassifier):
    """
    Classe de classfieur qui utilise le principe de maximum de vraissemblace pour estimer la classe
    d'un individu
    """

    def __init__(self, train, attr):
        """
        Le constructeur de la classe.

        Args:
            df (DataFrame): notre base de données.
            attr : L'attribu sur lequelle se base la classification.

        """
        APrioriClassifier.__init__(self)
        self.attr = attr  # l'attribut du classifieur
        # la table de classifieur dont on trouve les proba; il est sous la forme d'un dictionnaire dont les clé est les valeurs de target les donnée c'est les proba de attribut selon ses différentes valeurs
        self.table_P2DL = P2D_l(train, attr)

    def estimClass(self, vecteur):
        """
        Estime la classe majoritaire.

        Args:
            vecteur (dict): Représente un individu avec les attributs et les valeur associées il peut être n'import quoi.

        Returns:
            int: soit 0 ou 1 qui est la classe auquel appartien l'individu.
        """

        value_of_att = vecteur[self.attr]
        if(self.table_P2DL[1][value_of_att] > self.table_P2DL[0][value_of_att]):
            return 1
        else:
            return 0


# ***************** Question 3.c **************************************************************************

class MAP2DClassifier(APrioriClassifier):
    """
    un classifieur utilise le principe du maximum a posteriori
    """

    def __init__(self, train, attr):
        """
        le constructeur de la classe
        en entrée:
            train: le dataset
            attr: l'attribu de la classe sur lequel se fait la classification
        """
        APrioriClassifier.__init__(
            self)  # appel du constructeur de la classe mère
        self.attribut = attr  # initialisation de l'attribut de la classe ainsi que sa table P2Dp
        self.table_P2Dp = P2D_p(train, attr)

    def estimClass(self, vecteur):
        """
        Estime la classe majoritaire.

        Args:
            vecteur (dict): Représente un individu avec les attributs et les valeur associées il peut être n'import quoi.

        Returns:
            int: soit 0 ou 1 qui est la classe auquel appartien l'individu.
        """
        if(self.table_P2Dp[vecteur[self.attribut]][1] > self.table_P2Dp[vecteur[self.attribut]][0]):
            return 1
        else:
            return 0

# *****************************************************************************************************
#                               Question 04
# *****************************************************************************************************


# *****************************************************************************************************
#                               question 4.1
# *******************************************************************************************************

def nbParams(df, liste=[]):
    """
    Calcule le nombre d'octer d'occupe les tables 𝑃(𝑡𝑎𝑟𝑔𝑒𝑡|𝑎𝑡𝑡𝑟1,..,𝑎𝑡𝑡𝑟𝑘)
    Args:
        df (DataFrame):contient les données.
        liste (list, optional):la liste des attributs. Defaults to [].
    """
    n = 8  # la taille d'un float
    l = df.columns  # pour prendre en considération le cas ou la liste des attributs donnée est vide
    if(len(liste) == 0):  # pour prendre on considération si la liste donnée est vide
        liste = l  # si la liste est vide on considère tous les attributs

    for attr in liste:
        n = n*len(set(df[attr]))
    res = str(len(liste)) + " variable(s) : " + str(n) + " octets"
    print(res)


# *******************************************************************************************************
#                       question 4.2
# *******************************************************************************************************

def nbParamsIndep(df, attrs=[]):
    """
     calcule la taille mémoire nécessaire pour représenter les tables de probabilité
     étant donné un dataframe, en supposant qu'un float est représenté sur 8octets
     et en supposant l'indépendance des variables.en supposant l'indépendance des
     variables.
    Args:
        train ([dataframe]): contient les données
        attrs (list, optional):contient les attributs. Defaults to [].
    """
    nbP = 0
    if(attrs == []):
        attrs = list(df.keys())
    for i in attrs:
        l = list(np.unique(df[i]))
        nbP += len(l)
    taille = 8*nbP
    print(len(attrs), "variable(s):", taille, "octets")


# ***********************************************************************************************************
#                             Question 05
# **********************************************************************************************************

# **********************************************************************************************************
#                             Question 5.3
# ***********************************************************************************************************

def drawNaiveBayes(df, attr):
    """
    Une fonction qui dessine un graphe
    Args:
        df (dataframe): contient les données
        attr (liste ): la liste des attribut.
    """
    liste_attributs = df.columns
    x = ""
    for s in liste_attributs:
        if(s != attr):
            x = x+attr+"->"+s+";"
    return utils.drawGraph(x)


def nbParamsNaiveBayes(df, target, attributs=None):
    """
    Renvoie la taille mémoire nécessaire pour représenter les tables de probabilité étant
    donné un dataframe,en supposant qu'un float est représenté sur 8octets et en utilisant
    l'hypothèse du Naive Bayes.
    Args:
        df (dataframe): contient les données
        target (str):on le mit toujours à 'target'
        attributs (list): une liste d'attributs

    Returns:
        int:la taille mémoire en octets
    contraintes:
        target et la liste des attributs ont des clés valide pour le dataframe

    """
    taille = np.unique(df[target].values).size * 8

    if attributs is None:  # on test si les attributs ne nous sont pas donnée
        # on considère tous les attributs du dataframe
        attributs = list(df.columns)

    if attributs != []:  # on test si la liste des attributs est vide
        # on retire de la liste des attributs 'target'
        attributs.remove(target)

    for attr in attributs:  # pour chaque attributs dans la liste des attributs
        temp = (np.unique(df[target].values).size *
                np.unique(df[attr].values).size) * 8
        taille += temp

    print(str(len(attributs)) + " variable(s) : " + str(taille) + " octets")


# ******************************************************************************************************************
#                          Question 5.4
# ******************************************************************************************************************


class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Une classe de classification qui utilise le maximum de vraisemblance (ML)
    ainsi que l'hypothèse du Naïve Bayes, elle hérite de la classe APrioriClassifier
    """

    def __init__(self, df):
        """
        le constructeur de la classe
        Args:
            df (dataframe):contient les données
        sortie:
            un classifier MLNaiveBayesClassifier
        """
        APrioriClassifier.__init__(
            self)  # appel au constructeur de la classe mère
        # self.P2D_l2(train) # initialisatin de la table des probabilités
        self.table_MLNaive = {}
        attributs = list(df.columns)  # on récupère la liste des attributs
        attributs.remove("target")   # on supprime 'target' de la liste
        for attr in attributs:       # pour tous les attributs dans la liste des attributs
            self.table_MLNaive[attr] = P2D_l(
                df, attr)  # on ajoute au dictionnair

    def estimProbas(self, vecteur):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target)
        args:
            vecteur: un dictionnaire nom:valeur d'un individu
        renvoie:
            un dictionnaire classe:proba la probabilité que l'individu soit dansla classe 1 ou 0
        """
        res1 = 1  # la proba de la classe 1
        res0 = 1  # la proba de la classe 0
        for key in self.table_MLNaive:  # pour chaque attributs dont la table de classifier
            # on récupère le sous dictionnair associé à chaque attribut
            p = self.table_MLNaive[key]
            if vecteur[key] in p[0]:  # si la valeur de cet attribut est dans les sous dictionnair
                # on multiplie les probabilites des attributs
                res1 = res1 * p[1][vecteur[key]]
                res0 = res0 * p[0][vecteur[key]]
            else:  # la valuer de l'attribut ne se trouve dans l'ensemble
                return {0: 0.0, 1: 0.0}
        return {0: res0, 1: res1}

    def estimClass(self, vecteur):
        """
        Estime la classe majoritaire.

        Args:
            vecteur (dict): Représente un individu avec les attributs et les valeur associées il peut être n'import quoi.

        Returns:
            int: soit 0 ou 1 qui est la classe auquel appartien l'individu.
        """
        dict = self.estimProbas(
            vecteur)  # on récupère l'estimation des probabilité
        return int(dict[1] > dict[0])  # on test et on revoie le résultat


class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes.
    """

    def __init__(self, train):
        """
        le constructeur de la classe
        Args:
            train (dataframe):contient les données
        """
        APrioriClassifier.__init__(
            self)  # on appel le constructeur de la classe mère
        liste = list(train.columns)  # on récupère la liste des attribut
        liste.remove('target')  # on enlève l'attribut 'target' de la liste
        self.table_MAPNaive = {}  # initialisation de la table du classifier
        for attr in liste:
            self.table_MAPNaive[attr] = P2D_l(train, attr)

        self.pTarget = {1: train['target'].mean()}  # la proba de target=1
        self.pTarget[0] = 1 - self.pTarget[1]  # la proba de target=0

    def estimProbas(self, vecteur):
        """
        renvoie pour un individu la proba d'appartenir à la classe 0 et la classe 1
        Args:
            vecteur (dict):représente un individu
        Returns:
            (dict):les proba des classe 1 et 0 pour cet individu
        """
        P_0 = self.pTarget[0]  # on récupère la probabilité de target=0
        P_1 = self.pTarget[1]  # on récupère la probabilité de target=1
        for key in self.table_MAPNaive:  # pour tous les attributs
            # on récupère le sous dictionnaire le cet attribut
            p = self.table_MAPNaive[key]
            # on test si la valeur donné de l'invidu est dans le sous dictionnaire
            if vecteur[key] in p[0]:
                P_0 *= p[0][vecteur[key]]  # on multiplie les probabilité
                P_1 *= p[1][vecteur[key]]
            else:  # si la valuer de l'attribut dans le vecteur donné n'est pas dans la dataframe
                return {0: 0.0, 1: 0.0}
        # normaliser les valeurs les renvoyer
        return {0: P_0/(P_0 + P_1), 1: P_1 / (P_0 + P_1)}

    def estimClass(self, vecteur):
        """
        Estime la classe majoritaire.

        Args:
            vecteur (dict): Représente un individu avec les attributs et les valeur associées il peut être n'import quoi.

        Returns:
            int: soit 0 ou 1 qui est la classe auquel appartien l'individu.
        """
        res = self.estimProbas(
            vecteur)  # récuper l'estimaition des probabilité
        if res[0] >= res[1]:  # on test et en renvoie la classe qui possède la plus grande probabilité
            return 0
        return 1


# *************************************************************************************************************
#                                      Question 6
# *************************************************************************************************************

def isIndepFromTarget(df, attr, x):
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    Args:
        df (dataframe): contient les données
        attr (str):le nom de l'attribut
        x (int): le seuil

    Returns:
        [str]: yes ou non

    contraintes:
        _ le dataframe doit contenir les attribut: attr et target
        _ les seul valeur de target dans le dataframe est 0 ou 1
    """
    """
    la méthode de travail:
        la création du tableu de contingence S
        la méthode pivot_table pour obtenir notre tableau de contingence.
        J’en profite au passage pour remplacer les valeurs nulles par des zéro,
        je crée une copie du dataframe original et je m’assure de tout convertir en int.
        avec ca on aura pour chaque valuer de attr le nombre de fois qu'il figure avec target=0 et target=1
    """
    cont = df[['target', attr]].pivot_table(
        index='target', columns=attr, aggfunc=len).fillna(0).copy().astype(int)
    _, p, _, _ = chi2_contingency(cont)
    return p > x


class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    """
    Un classifier tilisent le maximum de vraisemblance (ML)
    pour estimer la classe d'un individu sur un modèle Naïve Bayes qu'ils auront
    préalablement optimisé grâce à des tests d'indépendance au seuil de 𝑥%
    """

    def __init__(self, df, x):
        """
        le constructeur de la clsse
        Args:
            df (dataframe): contient les données
            x (int):le sueil d'indépendance
        contrainte:
            x est une clé valide pour df
        """
        MLNaiveBayesClassifier.__init__(
            self, df)  # on appelle le constructeur de la classe mère
        self.df = df  # on initialise les variable de la classe
        # on récupère la liste des attributs de dataframe
        liste = list(df.columns)
        liste.remove('target')  # on enleve de cette liste l'attribut target
        # on initialise la liste des attributs qui ne sont indépendant de target
        self.liste_attributsDependant = []
        self.table_ReduceMLNaive = {}  # la table des proba de la clase
        for attr in liste:  # on chaque attribut
            if(not isIndepFromTarget(df, attr, x)):  # si il est indépendant donc on l'exclui de la table
                # si il est dépendant de target on l'ajoute à la liste des attributs dépendants
                self.liste_attributsDependant.append(attr)
                # on ajoute les proba de cet attribut ç la table des proba
                self.table_ReduceMLNaive[attr] = P2D_l(df, attr)

    def draw(self):
        """
        une fonction qui permet de dessiner un arbre dont la racine est target
        les feuille est la liste des attribut qui dépendent de target i.e on élimine les autre attributes
        qui sont indépendants de target
        """
        liste_attributs = self.liste_attributsDependant  # on récupère la liste des attributs de la classe
        x = ""  # on initialise une chaine vide
        for s in liste_attributs:  # pour chaque attributs
            if(s != 'target'):  # on reteste qu'il est différent de target
                x = x+'target'+"->"+s+";"  # on l'ajoute à la chaine
        # on appelle la fonction utils.drawGraph(x) avec la chaine formée
        return utils.drawGraph(x)


class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df, x):
        """
        le constructeur de la classe
        Args:
            df (dataframe): contient les données
            x (int):le seuil d'indépendance
        """
        MAPNaiveBayesClassifier.__init__(
            self, df)  # on appelle le constructeur de la classe mère
        self.df = df  # on initialise le dataframe de la classe
        self.pTarget = {1: df["target"].mean()}  # on calcule P(target=1)
        self.pTarget[0] = 1 - self.pTarget[1]  # on calcule P(target=0)
        # on récupère la liste des attributs du dataframe
        liste = list(df.columns)
        liste.remove('target')  # on enlève target
        # on initialise la liste des attributs dépendants de target
        self.liste_attributsDependant = []
        self.table_ReduceMAPNaive = {}  # on initilaise la table des proba de la classe
        for attr in liste:  # pour chaque attribut de la liste
            if(not isIndepFromTarget(df, attr, x)):  # si il est indépendant donc on l'exclui de la table
                # si il est dépendant de target on l'ajoute à la liste des attributs dépendants
                self.liste_attributsDependant.append(attr)
                # on ajoute les proba de cet attribut à la table des proba
                self.table_ReduceMAPNaive[attr] = P2D_l(df, attr)

    def estimProbas(self, vecteur):
        """
        Calcule la probabilité à posteriori par naïve Bayes réduit:
        P(target | attr1, ..., attrk).
        param:
            vecteur : le dictionnaire(présente un individu) nom-valeur pour qui on estime la classe
        """
        P_0 = self.pTarget[0]  # on récupère P(target=0)
        P_1 = self.pTarget[1]  # on récupère P(target=1)
        # on appelle la fonction estime proba de la classe mère car elle fait la meme chose
        dic = super().estimProbas(vecteur)
        # on renvoie les valeur sans normalisation
        return {0: dic[0]*(P_0+P_1), 1: dic[1]*(P_0+P_1)}

    def draw(self):
        """
         une fonction qui permet de dessiner un arbre dont la racine est target
         les feuille est la liste des attribut qui dépendent de target i.e on élimine les autre attributes
         qui sont indépendants de target
         """
        liste_attributs = self.liste_attributsDependant  # on récupère la liste des attributs dépendant de target
        x = ""  # on initialise une chaine vide qui va contenir la structure de notre arbre
        for s in liste_attributs:  # pour chaque attribut
            if(s != 'target'):  # on retest s'il est diférent de target
                x = x+'target'+"->"+s+";"  # on ajoute à la chaine comme un fils de target
        return utils.drawGraph(x)  # on désine le résultat

# ********************************************************************************************
#                                   question 7
# ********************************************************************************************


# ********************************************************************************************
#                                   question 7.2
# ********************************************************************************************


def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifiers à partir d'un dictionnaire dic de 
    {nom:instance de classifier} et d'un dataframe df, dans l'espace (précision,rappel). 

    :param dic: dictionnaire {nom:instance de classifier}
    :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    """
    import matplotlib.pyplot as plt  # on import matplotlib
    # on déclare un bableau de la taille de dataframe(le nombre de ligne)
    precision = np.empty(len(dic))
    rappel = np.empty(len(dic))

    # transforme le dictionnaire donnée en un objet itérable
    for i, nom in enumerate(dic):
        dico_stats = dic[nom].statsOnDF(df)
        precision[i] = dico_stats["Précision"]
        rappel[i] = dico_stats["Rappel"]

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(precision, rappel, marker='x', c='red')

    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))

    plt.show()

# *************************************************************************************
#                   Question 08
# *************************************************************************************

# *************************************************************************************
#                   Question 08.01
# *************************************************************************************


def MutualInformation(df, x, y):
    """
    Calcule l'information mutuelle entre les colonnes x et y du dataframe.
    Args:
        train (dataframe):contient les données
        attr (list):la liste des attributs 
    """
    list_x = np.unique(df[x].values)  # Valeurs possibles de x.
    list_y = np.unique(df[y].values)  # Valeurs possibles de y.

    dico_x = {list_x[i]: i for i in range(list_x.size)}
    # un dictionnaire associant chaque valeur a leur indice en list_x.

    dico_y = {list_y[i]: i for i in range(list_y.size)}
    # un dictionnaire associant chaque valeur a leur indice en list_y.

    mat_xy = np.zeros((list_x.size, list_y.size), dtype=int)
    # matrice des valeurs P(x,y) initialement à 0

    # on génére à l'aide de la méthode groupby les liste des index
    # des individu qui verifie x=valeur et y=valeur
    group = df.groupby([x, y]).groups

    for i, j in group:
        mat_xy[dico_x[i], dico_y[j]] = len(group[(i, j)])

    mat_xy = mat_xy / mat_xy.sum()

    mat_x = mat_xy.sum(1)
    # matrice des P(x)
    mat_y = mat_xy.sum(0)
    # matrice des P(y)
    mat_px_py = np.dot(mat_x.reshape((mat_x.size, 1)),
                       mat_y.reshape((1, mat_y.size)))
    # matrice des P(x)P(y)

    mat_res = mat_xy / mat_px_py
    mat_res[mat_res == 0] = 1
    # pour traiter log 0
    mat_res = np.log2(mat_res)
    mat_res *= mat_xy

    return mat_res.sum()


def ConditionalMutualInformation(df, x, y, z):
    list_x = np.unique(df[x].values)  # Valeurs possibles de x.
    list_y = np.unique(df[y].values)  # Valeurs possibles de y.
    list_z = np.unique(df[z].values)  # Valeurs possibles de z.

    dico_x = {list_x[i]: i for i in range(list_x.size)}
    # un dictionnaire associant chaque valeur a leur indice en list_x.

    dico_y = {list_y[i]: i for i in range(list_y.size)}
    # un dictionnaire associant chaque valeur a leur indice en list_y.

    dico_z = {list_z[i]: i for i in range(list_z.size)}
    # un dictionnaire associant chaque valeur a leur indice en list_z.

    mat_xyz = np.zeros((list_x.size, list_y.size, list_z.size), dtype=int)
    # matrice des valeurs P(x,y,z)

    group = df.groupby([x, y, z]).groups

    for i, j, k in group:
        mat_xyz[dico_x[i], dico_y[j], dico_z[k]] = len(group[(i, j, k)])

    mat_xyz = mat_xyz / mat_xyz.sum()  # on calcule les probas

    mat_xz = mat_xyz.sum(1)
    # matrice des P(x, z)

    mat_yz = mat_xyz.sum(0)
    # matrice des P(y, z)

    mat_z = mat_xz.sum(0)
    # matrice des P(z)

    mat_pxz_pyz = mat_xz.reshape(
        (list_x.size, 1, list_z.size)) * mat_yz.reshape((1, list_y.size, list_z.size))
    # matrice des P(x, z)P(y, z)

    mat_pxz_pyz[mat_pxz_pyz == 0] = 1
    mat_pz_pxyz = mat_z.reshape((1, 1, list_z.size)) * mat_xyz
    # matrice des P(z)P(x, y, z)

    mat_res = mat_pz_pxyz / mat_pxz_pyz
    mat_res[mat_res == 0] = 1
    # pour éviter des problèmes avec le log de zero
    mat_res = np.log2(mat_res)
    mat_res *= mat_xyz

    return mat_res.sum()

# ************************************************************************************************
#                               Question 08.02
# ************************************************************************************************


def MeanForSymetricWeights(a):
    """
    Calcule la moyenne des poids pour une matrice symétrique de diagonale nulle.
    La diagonale n'est pas prise en compte pour le calcul de la moyenne.
    en entré: 
        a: une matrice
    en sortie: 
        la moyenne
    """
    return a.sum()/(a.size - a.shape[0])  # car il ya autant de ligne que des valeurs dans la diagonal


def SimplifyConditionalMutualInformationMatrix(a):
    """
    annule toutes les valeurs plus petites que cette moyenne dans une matrice
    symétrique de diagonale nulle.
    Args:
        a (matrcice): la matrice pour dans on veut annule les valeur les plus petites que la moyenne de cette matrice
    Effet de bord : 
        la matrice avec des 0 à la place des valeurs inférieures à la moyenne
    """
    moyenne = MeanForSymetricWeights(
        a)  # on récupère la moyenne de la matrice a
    # on remplace dans a toute les valeurs inférieure à cette moyenne par 0
    a[a < moyenne] = 0

# ******************************************************************************************************************
#                                       Question 08.03
# ******************************************************************************************************************

def Kruskal(df, a):
    """
    Propose la liste des arcs à ajouter dans notre classifieur sous 
    la forme d'une liste de triplet (𝑎𝑡𝑡𝑟1,𝑎𝑡𝑡𝑟2,𝑝𝑜𝑖𝑑𝑠). 
    Args:
        df (dataframe): contient les données 
        a (matrice): Matrice symétrique de diagonale nulle.
    return:
        list[tuple]: pour dire que le résultat est une liste des tuple (attr1,attr2,poid)
    """
    liste_sommet = [x for x in df.keys() if x !="target"]  # la liste des attribut différent de target dans le dataframe 
    # on génére une liste avec toutes les configuration d'arrêtes possible et la matrice 0 nous donne le poid de chaque arrête
    list_arretes = [(liste_sommet[i],liste_sommet[j], a[i, j]) for i in range(a.shape[0]) for j in range(i + 1, a.shape[0]) if a[i, j] != 0] 

    list_arretes.sort(key=lambda x: x[2], reverse=True) # on trie la liste des arrêtes selon le poind et selon l'ordre croissant
    dict_sommet_parent={s:s for s in liste_sommet} # un dictionnair qui associe à chaque sommet son parent 
    dict_taille={s:1 for s in liste_sommet}
    res=[] # le résulat: la liste des arrêtes 
    for (u,v,poind) in list_arretes: # pour chaque tuple dans la liste des somment possible
        if(find(u,dict_sommet_parent)!=find(v,dict_sommet_parent)):
            res.append((u,v,poind))
            union(u, v,dict_taille,dict_sommet_parent)
    return res # la liste des arcs 


def find(u,dict_sommet_parent):
    """
    Trouve la racine du sommet u dans la forêt utilisée par l'algorithme de
    kruskal.
    Args:
        u (str):un sommet du graphe
        dict_sommet_parent(dict): un dictionnair qui associe à chaque sommet son parent  
    """
    racine=u
    #la recherche de la racine 
    while(racine!=dict_sommet_parent[u]): 
        racine=dict_sommet_parent[u] # on met la racine au parent de u

    #compression du chamin 
    while(u!=racine):
        v=dict_sommet_parent[u] # on stocke le parent de u dans v 
        dict_sommet_parent[u]=racine 
        u=v
    return racine

def union(u, v,dict_taille,dict_sommet_parent):
    """
    Union des deux arbres contenant u et v. Doivent être dans deux
    arbres differents.
    Args:
        u (str): un sommet
        v (str): un sommet
    """
    racine_u=find(u,dict_sommet_parent)
    racine_v=find(v,dict_sommet_parent)
    if(dict_taille[racine_u] < dict_taille[racine_v]): 
            dict_sommet_parent[racine_v] = racine_v 
            dict_taille[racine_v] += dict_taille[racine_u] 
    else: 
        dict_sommet_parent[racine_v] = racine_u 
        dict_taille[racine_u] += dict_taille[racine_v]  

# ********************************************************************************************************************
#                                       Question 8.4
# *********************************************************************************************************************



def ConnexSets(liste):
    """
    Costruit une liste des composantes connexes du graphe dont la liste d'aretes
    est list.
    Args:
        liste (list):une liste des acrs
    """
    res = []    # la liste des composantes connexes
    for (u, v,_) in liste: # pour chaque tuple(arc) dans la liste des arcs 
        ensemble_u = None
        ensemble_v = None
        for s in res:
            if u in s:
                ensemble_u = s
            if v in s:
                ensemble_v = s
        if ensemble_u is None and ensemble_v is None:
            res.append({u, v})
        elif ensemble_u is None:
            ensemble_u=set()
            ensemble_u.add(u)
        elif ensemble_v is None:
            ensemble_v=set()
            ensemble_v.add(v)
        elif ensemble_u != ensemble_v:
            res.remove(ensemble_u)
            ensemble_v = ensemble_v.union(ensemble_u)
    return res

def OrientConnexSets(df, arcs, attr_ref):
    """
    Une fonction qui propose pour chaque ensemble d'attributs connexes une racine et qui rend 
    la liste des arcs orientés.
    Args:
        df: Dataframe contenant les données. 
        arcs: liste d'ensembles d'arcs connexes.
        classe: l'attribut réference pour le calcul de l'information mutuelle.
    """
    arcs_copy = arcs.copy() # on sauvegarde la liste des arcs
    list_set = ConnexSets(arcs_copy) # la liste des composantes connexes du graphe
    list_ori = [] # liste des arcs orientés
    for s in list_set: # pour chaque composante connexe 
        attr_max = "" # on initialise la colonne max à une chaine vide 
        i_max = -float("inf") # on initialise le max à moins l'infinie 
        for attr in s: 
            i = MutualInformation(df, attr, attr_ref)
            if i > i_max:
                i_max = i
                attr_max=attr
        list_ori += ArbreOri(arcs_copy, attr_max)
    return list_ori

def ArbreOri(arcs,racine):
    """
    Renvoie l'arbre orienté depuis la racine 
    Args:
        arcs (list):la liste des arcs 
        racine (str):la racine de l'arbre 
    """
    arb= [] # on initialise l'arbre à liste vide 
    f = [racine] # au départ ya que la racine  
    while f != []:
        sommet = f.pop(0) # on renvoie le premier élement et on le supprime de f
        arcs_copy = arcs.copy() #  on copie la liste des arcs
        for (u, v, poids) in arcs_copy: # pour chaque tuple dans la liste des arcs
            if sommet == u: 
                arb.append((u, v)) # on l'ajoute à l'arbre contruite 
                arcs_copy.remove((u, v, poids))
                f.append(v)
            elif sommet == v:
                arb.append((v, u))
                arcs_copy.remove((u, v, poids))
                f.append(u)
    return arb





# **************************************************************************************************************************
#                                          Question 8.5
# ***************************************************************************************************************************


class MAPTANClassifier(APrioriClassifier):
    """
    Une classe qui construit un modèle TAN donnée  
    Args:
        APrioriClassifier (objet): Le classfier apriori 
    """

    def __init__(self,df):
        """
        Le constructeur de la classe
        Args:
            df (dataframe): contient les données
        """

