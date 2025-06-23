import subprocess

def run_pipeline():
    print("🚀 Running full Titanic survival prediction pipeline...")

    steps = [
        "python src/data/preprocess.py",
        "python src/models/train_model.py",
        "python src/models/evaluate_test.py"
    ]

    for step in steps:
        print(f"\n➡️ {step}")
        result = subprocess.run(step, shell=True)
        if result.returncode != 0:
            print(f"❌ Failed at: {step}")
            break
    else:
        print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()