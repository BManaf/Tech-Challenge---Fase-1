import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(img_size=(224, 224)):
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))

    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.05)(x)
    x = layers.RandomZoom(0.10)(x)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def load_breakhis_train_val(data_dir, img_size=(224, 224), batch_size=16, seed=42):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Diretorio nao encontrado: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset="training",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset="validation",
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def evaluate_on_validation(model, val_ds):
    y_true = []
    y_prob = []

    for x_batch, y_batch in val_ds:
        probs = model.predict(x_batch, verbose=0).reshape(-1)
        y_true.extend(y_batch.numpy().reshape(-1).tolist())
        y_prob.extend(probs.tolist())

    y_pred = (np.array(y_prob) >= 0.5).astype(int)
    return confusion_matrix_binary(y_true, y_pred)


def main():
    data_dir = r"C:\Users\dell\Desktop\FIAP\Tech Challenge - Fase1\data\BreaKHis_v1\BreaKHis_v1\histology_slides\breast"

    img_size = (224, 224)
    batch_size = 16
    epochs = 5

    print("Carregando dataset (BreaKHis)...")
    train_ds, val_ds, class_names = load_breakhis_train_val(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=42,
    )

    print("Classes encontradas:", class_names)
    print("Iniciando treinamento...")

    model = build_model(img_size=img_size)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "cnn_cancer_mama_best.keras",
            save_best_only=True
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    print("Avaliando no conjunto de validacao...")
    metrics = model.evaluate(val_ds, verbose=0)
    for name, value in zip(model.metrics_names, metrics):
        print(f"{name}: {value:.4f}")

    cm = evaluate_on_validation(model, val_ds)
    print("Matriz de confusao (validacao, 1 = classe positiva):")
    print(cm)

    model.save("cnn_cancer_mama_final.keras")
    print("Modelo salvo em: cnn_cancer_mama_final.keras")
    print("Checkpoint salvo em: cnn_cancer_mama_best.keras")


if __name__ == "__main__":
    main()
