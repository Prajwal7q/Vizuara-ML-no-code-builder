from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(512), nullable=False)
    n_rows = db.Column(db.Integer, nullable=False)
    n_cols = db.Column(db.Integer, nullable=False)
    columns_json = db.Column(db.Text, nullable=False)  # store list of columns as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    pipelines = db.relationship('Pipeline', backref='dataset', lazy=True, cascade='all, delete-orphan')


class Pipeline(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id', ondelete='CASCADE'), nullable=False)

    # preprocessing choices: e.g. {"standardization": true, "normalization": false}
    preprocessing_json = db.Column(db.Text, nullable=False)

    split_ratio = db.Column(db.Float, nullable=False)          # e.g. 0.8 for 80-20
    target_column = db.Column(db.String(100), nullable=False)  # name of target column
    model_type = db.Column(db.String(50), nullable=False)      # "logistic_regression" or "decision_tree"

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    runs = db.relationship('Run', backref='pipeline', lazy=True, cascade='all, delete-orphan')


class Run(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pipeline_id = db.Column(db.Integer, db.ForeignKey('pipeline.id', ondelete='CASCADE'), nullable=False)

    status = db.Column(db.String(20), nullable=False, default='PENDING')   # PENDING/RUNNING/DONE/FAILED
    accuracy = db.Column(db.Float, nullable=True)
    metrics_json = db.Column(db.Text, nullable=True)  # e.g. confusion matrix, classification report snippets

    chart_path = db.Column(db.String(512), nullable=True)  # path to generated chart image in static/
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
