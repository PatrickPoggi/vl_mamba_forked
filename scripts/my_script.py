import os

BASE_DIR = "checkpoints"
model_dirs = os.listdir(BASE_DIR)

print(f"model_dirs = {model_dirs}")

'''
NOTE: I will exclude some models from the reproducibility analysis for resource constraints purposes (cases "*2.8b*") and for impossibility in loading them due to out-of-date model files ("*mambavl_790*")
'''
models_to_exclude = [ "finetuned_mambavl_790m/", "finetuned_mambavl_790m_old/", "finetuned_mambavl_2.8b", "finetuned_pythiavl_2.8b"]

model_dirs = [md for md in model_dirs if md not in models_to_exclude]
print(f"Excluding some models from analysis...{models_to_exclude}\nFinal model_dirs: {model_dirs}")

for model_dir in model_dirs:
    model_path = os.path.join(BASE_DIR, model_dir)
    if os.path.isdir(model_path):
        command = f"./scripts/evaluate_on_all.sh {model_path} ./results/{model_dir}/ 0 ./cache/datasets/ ./storage/datasets"
        print(f"Executing: {command}")
        os.system(command)
	