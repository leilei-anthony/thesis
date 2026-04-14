import json

# Define the notebook structure
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DAiSEE Predictive Modeling & Explainable AI (SHAP)\n",
    "### Thesis Machine Learning Pipeline (Corrected Syntax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import shap\n",
    "import os\n",
    "from sklearn.metrics import classification_report, f1_score, confusion_matrix\n",
    "\n",
    "# Initialize SHAP visualizer\n",
    "shap.initjs()\n",
    "\n",
    "labels = ['Boredom', 'Engagement', 'Confusion', 'Frustration']\n",
    "data_folder = 'Data/Engineered/3_Changepoint'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_and_clean(file_path):\n",
    "    if not os.path.exists(file_path):\n",
    "        print(f'Error: File {file_path} not found.')\n",
    "        return None, None\n",
    "    \n",
    "    df = pd.read_csv(file_path)\n",
    "    \n",
    "    # Rename labels to remove the _first suffix from aggregation\n",
    "    rename_map = {f'{l}_first': l for l in labels}\n",
    "    df = df.rename(columns=rename_map)\n",
    "    \n",
    "    df = df.fillna(0)\n",
    "    \n",
    "    # Identify metadata to drop\n",
    "    drop_cols = ['source_video_path', 'person_id', 'source_video_path_first', 'person_id_first']\n",
    "    actual_labels = [l for l in labels if l in df.columns]\n",
    "    \n",
    "    X = df.drop(columns=actual_labels + [c for c in drop_cols if c in df.columns])\n",
    "    y = df[actual_labels]\n",
    "    \n",
    "    return X, y\n",
    "\n",
    "X_train, y_train = load_and_clean(os.path.join(data_folder, '3_Changepoint_Train.csv'))\n",
    "X_val, y_val = load_and_clean(os.path.join(data_folder, '3_Changepoint_Validation.csv'))\n",
    "X_test, y_test = load_and_clean(os.path.join(data_folder, '3_Changepoint_Test.csv'))\n",
    "print(f'Features loaded: {X_train.shape[1]}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Train and Evaluate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {}\n",
    "\n",
    "for label in labels:\n",
    "    if label not in y_train.columns: continue\n",
    "        \n",
    "    print(f'\\nTraining Model for: {label}')\n",
    "    \n",
    "    model = xgb.XGBClassifier(\n",
    "        n_estimators=500,\n",
    "        learning_rate=0.03,\n",
    "        max_depth=5,\n",
    "        objective='multi:softprob',\n",
    "        num_class=4,\n",
    "        random_state=42,\n",
    "        early_stopping_rounds=10,\n",
    "        eval_metric='mlogloss'\n",
    "    )\n",
    "    \n",
    "    model.fit(\n",
    "        X_train, y_train[label],\n",
    "        eval_set=[(X_val, y_val[label])],\n",
    "        verbose=False\n",
    "    )\n",
    "    models[label] = model\n",
    "    \n",
    "    preds = model.predict(X_test)\n",
    "    print(f'{label} Macro F1: {f1_score(y_test[label], preds, average=\"macro\"):.4f}')\n",
    "    \n",
    "    plt.figure(figsize=(5, 4))\n",
    "    sns.heatmap(confusion_matrix(y_test[label], preds), annot=True, fmt='d', cmap='Blues')\n",
    "    plt.title(f'Confusion Matrix: {label}')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. SHAP Analysis\n",
    "This section uses SHAP to explain why the model predicts High Boredom."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "target = 'Boredom'\n",
    "explainer = shap.TreeExplainer(models[target])\n",
    "shap_values = explainer.shap_values(X_test)\n",
    "\n",
    "print(f'SHAP Summary for Level 2 (High {target}):')\n",
    "shap.summary_plot(shap_values[2], X_test, max_display=12)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

# Write to file
with open('DAiSEE_Model_SHAP_Final.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=4)

print("DAiSEE_Model_SHAP_Final.ipynb has been generated successfully.")