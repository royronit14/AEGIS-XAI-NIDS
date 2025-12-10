import sys
from pathlib import Path

# enable imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from aegis.data.preprocess.preprocess import preprocess_dataframe
from aegis.models.baseline import BaselineModels
from scripts.run_preprocess import load_dataset


def main():
    # dataset options: "cicids", "botiot", "unsw"
    DATASET = "unsw"

    print(f"\n🔥 Running baseline for: {DATASET.upper()}")

    # Load raw dataset (chunked for BoT-IoT)
    df = load_dataset(DATASET)
    print(f"[+] Dataset shape: {df.shape}")

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_dataframe(df)
    print("[+] Preprocessing done.")

    # Initialize baseline model manager
    bm = BaselineModels()

    # Train models
    trained = bm.train(X_train, y_train)

    # Evaluate each model
    for name, model in trained.items():
        print(f"\n📊 Evaluating: {name}")
        metrics = bm.evaluate(model, X_test, y_test)

        print("   Accuracy:", metrics["accuracy"])
        print("   Precision:", metrics["precision"])
        print("   Recall:", metrics["recall"])
        print("   F1 Score:", metrics["f1_score"])
        print("   Confusion Matrix:\n", metrics["confusion_matrix"])

        # save model
        bm.save_model(model, name=f"{DATASET}_{name}")


if __name__ == "__main__":
    main()
