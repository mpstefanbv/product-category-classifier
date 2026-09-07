# product-category-classifier
ML model for automatic product category prediction
Un proiect de clasificare automata a produselor pe baza titlului, folosind TF‑IDF si Logistic Regression.
Modelul primeste un titlu de produs si returneaza categoria corecta.

1. Structura proiectului
product-category-classifier/
│
├── data/
│   └── products.csv
│
├── src/
│   ├── train_model.py
│   └── predict_category.py
│
├── models/
│   ├── product_category_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── products_eda_and_models.ipynb
│
└── README.md

2. Notebook EDA
Notebook-ul products_eda_and_models.ipynb contine:
-analiza datasetului
-verificarea valorilor lipsa
-distributia categoriilor
-grafice (histograme, bar chart)
-curatarea textului
-vectorizare TF‑IDF
-impartirea datelor in train/test
-antrenarea modelului
-evaluare (classification report + confusion matrix)
-testare manuala a predictiilor
-salvarea modelului .pkl
Notebook-ul este complet si poate fi rulat direct in VS Code.

3. Cum rulezi modelul
- Antrenarea modelului
Din terminal: python src/train_model.py
Modelul antrenat va fi salvat in: models/product_category_model.pkl
                                  models/tfidf_vectorizer.pkl
- Predictii
python src/predict_category.py
Scriptul va cere un titlu de produs si va afisa categoria prezisa.

4. Tehnologii folosite
-Python 3
-Pandas
-Scikit-learn
-Matplotlib
-Seaborn
-Jupyter Notebook
-TF‑IDF Vectorizer
-Logistic Regression

5. Scopul proiectului
Proiectul demonstrează:
-procesul complet de analiza a datelor (EDA)
-antrenarea unui model de clasificare text
-evaluarea performantei
-integrarea modelului intr-un script de predictții
-organizarea corecta a unui proiect ML
 