import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# Tworzenie przykładowego zbioru danych
np.random.seed(42)
n_samples = 300

# Generowanie danych
age = np.random.normal(55, 10, n_samples).astype(int)
age = np.clip(age, 30, 80)

sex = np.random.binomial(1, 0.6, n_samples)  # 1 = mężczyzna, 0 = kobieta
cp = np.random.randint(0, 4, n_samples)      # typ bólu w klatce (0-3)
trestbps = np.random.normal(130, 20, n_samples).astype(int)  # ciśnienie
trestbps = np.clip(trestbps, 90, 200)

chol = np.random.normal(220, 40, n_samples).astype(int)  # cholesterol
chol = np.clip(chol, 120, 400)

# Tworzenie DataFrame
data = pd.DataFrame({
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
})

# Dodanie target na podstawie warunków
conditions = [
    (data['age'] > 60) & (data['chol'] > 240),
    (data['cp'] > 2) & (data['trestbps'] > 140),
    (data['age'] > 50) & (data['chol'] > 300),
    (data['trestbps'] > 160) & (data['chol'] > 280)
]
data['target'] = np.where(np.any(conditions, axis=0), 1, 0)

# Analiza statystyczna danych
print("Podstawowe statystyki:")
print(data.describe())

# Sprawdzenie brakujących wartości
print("\nBrakujące wartości:")
print(data.isnull().sum())

# Wizualizacja korelacji między zmiennymi
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Macierz korelacji parametrów zdrowotnych')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Podział na cechy i etykiety
X = data.drop('target', axis=1)
y = data['target']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utworzenie i trenowanie modelu
clf = DecisionTreeClassifier(random_state=42, max_depth=4)
clf.fit(X_train, y_train)

# Predykcja i ocena modelu
y_pred = clf.predict(X_test)

print("\nDokładność modelu:", accuracy_score(y_test, y_pred))
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Wizualizacja macierzy pomyłek
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Macierz pomyłek')
plt.ylabel('Wartości rzeczywiste')
plt.xlabel('Wartości przewidziane')
plt.savefig('confusion_matrix.png')
plt.close()

# Wizualizacja drzewa decyzyjnego
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, 
               class_names=['Brak choroby serca', 'Choroba serca'], 
               filled=True, rounded=True)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Analiza ważności cech
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Ważność cech w modelu')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Funkcja do predykcji dla nowego pacjenta
def predict_heart_disease(age, sex, cp, trestbps, chol):
    new_patient = np.array([[age, sex, cp, trestbps, chol]])
    prediction = clf.predict(new_patient)
    probability = clf.predict_proba(new_patient)
    return prediction[0], probability[0]

# Przykład użycia
example_patient = predict_heart_disease(
    age=65,
    sex=1,  # mężczyzna
    cp=2,   # typ bólu w klatce
    trestbps=145,  # ciśnienie krwi
    chol=250,      # cholesterol
)

print("\nPrzykładowa predykcja dla pacjenta:")
print(f"Przewidywanie: {'Ryzyko choroby serca' if example_patient[0] == 1 else 'Brak ryzyka choroby serca'}")
print(f"Prawdopodobieństwo: {example_patient[1][1]:.2%}")

# Zapisanie wyników do pliku
with open('heart_disease_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("RAPORT Z ANALIZY RYZYKA CHORÓB SERCA\n")
    f.write("=====================================\n\n")
    f.write("1. OPIS DANYCH\n")
    f.write("Zbiór danych zawiera następujące parametry:\n")
    f.write("- Wiek (age): 30-80 lat\n")
    f.write("- Płeć (sex): 0 = kobieta, 1 = mężczyzna\n")
    f.write("- Typ bólu w klatce piersiowej (cp): 0-3\n")
    f.write("- Ciśnienie krwi (trestbps): 90-200 mmHg\n")
    f.write("- Cholesterol (chol): 120-400 mg/dl\n\n")
    
    f.write("2. DOKŁADNOŚĆ MODELU\n")
    f.write(f"Dokładność: {accuracy_score(y_test, y_pred):.2%}\n\n")
    
    f.write("3. RAPORT KLASYFIKACJI\n")
    f.write(classification_report(y_test, y_pred))
    
    f.write("\n4. NAJWAŻNIEJSZE CECHY\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    f.write("\n5. PRZYKŁADOWE REGUŁY DECYZYJNE\n")
    f.write("- Wysokie ryzyko przy wieku > 60 lat i cholesterolu > 240\n")
    f.write("- Wysokie ryzyko przy silnym bólu w klatce (cp > 2) i ciśnieniu > 140\n")
    f.write("- Wysokie ryzyko przy wieku > 50 lat i cholesterolu > 300\n") 