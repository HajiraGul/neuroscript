import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from config import *
from feature_extraction import preprocess_image, extract_features_from_array


def ensure_dirs():
    for d in [NORMAL_DIR, OCD_DIR, ADHD_DIR, MODEL_DIR, OUTPUT_DIR, TEMP_DIR]:
        os.makedirs(d, exist_ok=True)


def load_dataset():
    imgs = []
    feats = []
    labels = []

    for folder, label in [(NORMAL_DIR, 0), (OCD_DIR, 1), (ADHD_DIR, 2)]:
        for fn in os.listdir(folder):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                p = os.path.join(folder, fn)

                try:
                    img = preprocess_image(p, IMAGE_SIZE)
                    imgs.append(img)
                    feats.append(extract_features_from_array(img))
                    labels.append(label)

                except Exception as e:
                    print("Skipped", p, e)

    if not imgs:
        raise ValueError(
            "Add images in dataset/normal, dataset/ocd_like, dataset/adhd_like"
        )

    return (
        np.array(imgs).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1),
        np.array(feats),
        np.array(labels)
    )


def build_cnn():
    m = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax")
    ])

    m.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return m


def ev(name, y, p):
    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y, p), 4),
        "Precision": round(
            precision_score(y, p, average="weighted", zero_division=0),
            4
        ),
        "Recall": round(
            recall_score(y, p, average="weighted", zero_division=0),
            4
        ),
        "F1 Score": round(
            f1_score(y, p, average="weighted", zero_division=0),
            4
        )
    }


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "OCD-like", "ADHD-like"]
    )

    disp.plot(
        cmap="Blues",
        ax=ax,
        values_format="d",
        colorbar=True
    )

    ax.set_title(
        "Confusion Matrix - Handwriting Classification",
        fontsize=14
    )

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("Actual Class", fontsize=12)

    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_PATH,
        bbox_inches="tight",
        dpi=300
    )
    plt.close()


def main():
    ensure_dirs()

    Ximg, Xf, y = load_dataset()

    print("Samples:", len(y), dict(zip(*np.unique(y, return_counts=True))))

    XtrI, XteI, XtrF, XteF, ytr, yte = train_test_split(
        Ximg,
        Xf,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()

    XtrFs = scaler.fit_transform(XtrF)
    XteFs = scaler.transform(XteF)

    rf = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    rf.fit(XtrFs, ytr)
    rfp = rf.predict(XteFs)

    svm = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    svm.fit(XtrFs, ytr)
    svmp = svm.predict(XteFs)

    cnn = build_cnn()

    cnn.fit(
        XtrI,
        ytr,
        validation_data=(XteI, yte),
        epochs=10,
        batch_size=8,
        verbose=1
    )

    cnnp = np.argmax(
        cnn.predict(XteI, verbose=0),
        axis=1
    )

    df = pd.DataFrame([
        ev("Random Forest", yte, rfp),
        ev("SVM", yte, svmp),
        ev("CNN", yte, cnnp)
    ])

    df.to_csv(METRICS_PATH, index=False)

    print(df)

    save_confusion_matrix(yte, cnnp)

    joblib.dump(rf, RF_MODEL_PATH)
    joblib.dump(svm, SVM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    cnn.save(CNN_MODEL_PATH)

    print("Training completed. Run: streamlit run app.py")


if __name__ == "__main__":
    main()