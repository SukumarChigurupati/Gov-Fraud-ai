import pandas as pd
import xgboost as xgb
import shap

TRAIN_FILE = "data/medicare_fraud.csv"
MODEL_FILE = "model.bin"


def load_model_artifacts():
    """
    Loads an already-trained model if available.
    Otherwise trains model fresh using medicare_fraud.csv.
    This function NEVER uses the uploaded dataset.
    """
    # Load the training data
    train_df = pd.read_csv(TRAIN_FILE)

    # ✅ MODEL DOES NOT EXPECT A "fraud" COLUMN
    # Instead, last column = label
    feature_cols = train_df.columns[:-1]
    label_col = train_df.columns[-1]

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    try:
        model.load_model(MODEL_FILE)
    except:
        model.fit(X_train, y_train)
        model.save_model(MODEL_FILE)

    explainer = shap.TreeExplainer(model)
    return model, explainer
