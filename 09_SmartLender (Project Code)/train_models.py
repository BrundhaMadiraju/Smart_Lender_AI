import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from preprocess import preprocess

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"  Train Accuracy: {train_acc*100:.2f}%")
    print(f"  Test  Accuracy: {test_acc*100:.2f}%")
    print(classification_report(y_test, model.predict(X_test),
                                 target_names=['Rejected', 'Approved']))

    # Confusion matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Rejected', 'Approved'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix — {name}')
    plt.savefig(f'static/cm_{name.replace(" ", "_")}.png')
    plt.close()

    return train_acc, test_acc, model


def main():
    X_train, X_test, y_train, y_test, _ = preprocess()

    models = {
        'Decision Tree':   DecisionTreeClassifier(random_state=42),
        'Random Forest':   RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN':             KNeighborsClassifier(n_neighbors=5),
        'XGBoost':         XGBClassifier(n_estimators=100, learning_rate=0.1,
                                         use_label_encoder=False,
                                         eval_metric='logloss', random_state=42),
    }

    results = {}
    best_name, best_acc, best_model = '', 0, None

    for name, model in models.items():
        train_acc, test_acc, trained = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results[name] = {'train': train_acc, 'test': test_acc}
        if test_acc > best_acc:
            best_acc   = test_acc
            best_name  = name
            best_model = trained

    # ── Comparison chart ──
    names = list(results.keys())
    train_scores = [results[n]['train']*100 for n in names]
    test_scores  = [results[n]['test']*100  for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - 0.2, train_scores, 0.4, label='Train', color='steelblue')
    bars2 = ax.bar(x + 0.2, test_scores,  0.4, label='Test',  color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Comparison — Train vs Test Accuracy')
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f%%', padding=3)
    ax.bar_label(bars2, fmt='%.1f%%', padding=3)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig('static/model_comparison.png')
    plt.show()

    print(f"\n✅ Best model: {best_name} (Test Accuracy: {best_acc*100:.2f}%)")
    joblib.dump(best_model, 'model/best_model.pkl')
    print("✅ Model saved to model/best_model.pkl")


if __name__ == '__main__':
    main()