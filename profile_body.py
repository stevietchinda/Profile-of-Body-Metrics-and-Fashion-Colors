import pandas as pd

# Chargement du fichier en spécifiant le séparateur
df = pd.read_csv('profile_body_metric_and_fashion_color.csv', delimiter=';', engine='python')

# Nettoyage des espaces en trop dans chaque colonne
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Affichage  de l'entete du DataFrame propre
print(df.head())
print(df.info())

# Suppression des lignes vides
df.dropna(how='all', inplace=True)  # où toutes les valeurs sont NaN

# Suppression des doublons
df.drop_duplicates(inplace=True)
print(df.info())

#**************ARBRE DE DECISION************************

# arbre de decision
from sklearn.tree import DecisionTreeClassifier
# Sélectionner les caractéristiques

X = df[['Height(Centimeter)', 'Weight(Kilograms)', 'Gender']]
y = df['BMI']  # label, c.-à-d. la colonne à prédire

# Convertir la variable catégorielle 'Gender' en numérique
X['Gender'] = X['Gender'].map({'male': 0, 'female': 1})

# Initialiser et entraîner le classificateur
max_depth = 4  # Profondeur maximale de l'arbre
clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X, y)  # Entraînement du programme
print(clf)


# Visualisation de l'arbre de décision
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))

# Passer class_names=True pour générer automatiquement les classes
plt.figure(figsize=(30, 12))  # Agrandir la figure pour plus d'espace
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Ideal', 'Underweight', 'Overweight'],
    filled=True,
    rounded=True,
    fontsize=10,
    precision=2
)
plt.show()