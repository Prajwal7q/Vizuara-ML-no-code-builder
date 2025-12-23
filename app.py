import os
import json
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from models import db, Dataset, Pipeline, Run

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ml_pipeline.sqlite"
app.config["SECRET_KEY"] = "CHANGE_THIS_SECRET_KEY"
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

db.init_app(app)

with app.app_context():
    db.create_all()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("dataset")
    if not file:
        flash("Please select a file", "error")
        return redirect(url_for("index"))

    filename = file.filename
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ["csv", "xlsx", "xls"]:
        flash("Only CSV or Excel files are supported", "error")
        return redirect(url_for("index"))

    unique_name = f"{uuid.uuid4()}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        if ext == "csv":
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception:
        flash("Error reading file. Please check your dataset.", "error")
        return redirect(url_for("index"))

    n_rows, n_cols = df.shape
    columns = df.columns.tolist()

    columns_json = json.dumps(columns)

    dataset = Dataset(
        filename=filename,
        path=save_path,
        n_rows=n_rows,
        n_cols=n_cols,
        columns_json=columns_json
    )
    db.session.add(dataset)
    db.session.commit()

    return render_template(
        "index.html",
        step="preprocess",
        dataset=dataset,
        columns=columns,
        sample=df.head().to_html(classes="table table-striped table-sm", index=False)
    )


@app.route("/configure", methods=["POST"])
def configure():
    dataset_id = request.form.get("dataset_id")
    target_column = request.form.get("target_column")
    split_ratio = float(request.form.get("split_ratio", 0.8))
    model_type = request.form.get("model_type")

    # checkbox values
    standardization = request.form.get("standardization") == "on"
    normalization = request.form.get("normalization") == "on"

    preprocessing = {
        "standardization": standardization,
        "normalization": normalization
    }

    if not dataset_id or not target_column or not model_type:
        flash("Please select all required options", "error")
        return redirect(url_for("index"))

    pipeline = Pipeline(
        dataset_id=int(dataset_id),
        preprocessing_json=json.dumps(preprocessing),
        split_ratio=split_ratio,
        target_column=target_column,
        model_type=model_type
    )
    db.session.add(pipeline)
    db.session.commit()

    return redirect(url_for("run_pipeline", pipeline_id=pipeline.id))


@app.route("/run/<int:pipeline_id>", methods=["GET"])
def run_pipeline(pipeline_id):
    pipeline = Pipeline.query.get_or_404(pipeline_id)
    dataset = pipeline.dataset

    run = Run(pipeline_id=pipeline.id, status="RUNNING")
    db.session.add(run)
    db.session.commit()

    try:
        print(f"DEBUG: Loading dataset {dataset.path}")
        
        ext = dataset.filename.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(dataset.path)
        else:
            df = pd.read_excel(dataset.path)
        
        print(f"DEBUG: Dataset shape: {df.shape}")
        
        target = pipeline.target_column
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")
        
        y = df[target]
        print(f"DEBUG: Target '{target}' unique values: {sorted(y.unique())}")

        # BINARIZE if needed
        if len(y.unique()) > 2:
            print(f"DEBUG: Binarizing '{target}' using median threshold")
            median_val = y.median()
            y = (y > median_val).astype(int)
            print(f"DEBUG: Binarized target unique values: {sorted(y.unique())}")
        
        X = df.drop(columns=[target])
        print(f"DEBUG: Features shape: {X.shape}")

        # Handle preprocessing (only numeric columns)
        preprocessing = json.loads(pipeline.preprocessing_json)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features found for modeling")

        X_numeric = X[numeric_cols]

        if preprocessing.get("standardization"):
            scaler = StandardScaler()
            X_numeric = scaler.fit_transform(X_numeric)
        elif preprocessing.get("normalization"):
            scaler = MinMaxScaler()
            X_numeric = scaler.fit_transform(X_numeric)

        test_size = 1 - pipeline.split_ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=test_size, random_state=42, stratify=y
        )

        if pipeline.model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix chart
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Pred 0', 'Pred 1'], 
               yticklabels=['True 0', 'True 1'],
               title=f'Confusion Matrix (Acc: {acc:.1%})',
               ylabel='True label',
               xlabel='Predicted label')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        chart_filename = f"{uuid.uuid4()}.png"
        chart_path = os.path.join("static", chart_filename)
        plt.savefig(chart_path, bbox_inches='tight', dpi=100)
        plt.close()

        metrics = {"confusion_matrix": cm.tolist(), "accuracy": float(acc)}

        run.status = "DONE"
        run.accuracy = float(acc)
        run.metrics_json = json.dumps(metrics)
        run.chart_path = chart_filename
        db.session.commit()
        
        print(f"SUCCESS: Accuracy = {acc:.3f}")

    except Exception as e:
        run.status = "FAILED"
        run.metrics_json = json.dumps({"error": str(e)})
        db.session.commit()
        print(f"ERROR: {e}")
        flash(f"Pipeline failed: {str(e)}", "error")
        return redirect(url_for("index"))

    return redirect(url_for("results", run_id=run.id))


@app.route("/results/<int:run_id>", methods=["GET"])
def results(run_id):
    run = Run.query.get_or_404(run_id)
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    return render_template("results.html", run=run, metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
