import pickle

with open("models/product_category_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

while True:
    title = input("Product title: ")
    if title.lower() == "exit":
        break

    X = vectorizer.transform([title.lower()])
    pred = model.predict(X)[0]
    print("Predicted category:", pred)
