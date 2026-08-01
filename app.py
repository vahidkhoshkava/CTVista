
import os
import pandas as pd
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing. Add it under Space Settings > Secrets."
    )

CT_PATH = hf_hub_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    filename="dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz",
    token=HF_TOKEN,
)

results_df = pd.read_csv("results.csv")


from pathlib import Path

# Find the previously executed CTVista dashboard code.
dashboard_source = None

for cell_source in reversed(In):
    if (
        "CTVista — Bilingual AI Chest CT Review Dashboard" in cell_source
        and "with gr.Blocks" in cell_source
    ):
        dashboard_source = cell_source
        break

if dashboard_source is None:
    raise RuntimeError(
        "I could not find the CTVista dashboard cell in this runtime. "
        "Run the full CTVista dashboard cell once, then run this cell again."
    )

# Remove Colab-only installation commands.
clean_lines = []

for line in dashboard_source.splitlines():
    if not line.strip().startswith("!pip"):
        clean_lines.append(line)

dashboard_source = "\n".join(clean_lines)

# A Hugging Face Space must define the CT and results itself.
startup_code = '''
import os
import pandas as pd
from huggingface_hub import hf_hub_download

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing. Add it under Space Settings > Secrets."
    )

CT_PATH = hf_hub_download(
    repo_id="ibrahimhamamci/CT-RATE",
    repo_type="dataset",
    filename="dataset/valid/valid_1/valid_1_a/valid_1_a_1.nii.gz",
    token=HF_TOKEN,
)

results_df = pd.read_csv("results.csv")
'''

# Replace the Colab launch command with the Space launch command.
old_launch = '''demo.launch()'''

dashboard_source = dashboard_source.replace(
    old_launch,
    "demo.launch()",
)

app_source = startup_code + "\n\n" + dashboard_source

Path("/content/app.py").write_text(
    app_source,
    encoding="utf-8",
)

print("Created /content/app.py")
print("Size:", round(Path("/content/app.py").stat().st_size / 1024, 1), "KB")