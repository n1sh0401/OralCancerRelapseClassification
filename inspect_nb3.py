import json
from pathlib import Path
p = Path("c:/Downloads/vscode/Codings/OralCancerRelapseClassification/DanishOralCancerRelapse_v2.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if any(k in src for k in ["preprocessor =", "LR =", "rf_model =", "LogisticRegression(", "RandomForestClassifier("]):
            print("CELL", idx)
            print(src)
            print("---")
