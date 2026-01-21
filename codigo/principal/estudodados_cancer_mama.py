from sklearn.datasets import load_breast_cancer
import pandas as pd


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    print("Tamanho")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas (com target): {df.shape[1]}")
    print(f"Quantidade de variaveis: {len(data.feature_names)}")
    print()

    print("classes")
    print("Classes:", list(data.target_names))
    print("Quantidade por classe:")
    print(df["target"].value_counts().sort_index())
    print()
    print("Entendimento:")
    print("0 = malignant (maligno)")
    print("1 = benign (benigno)")
    print()

    print("Amostra)")
    print(df.head(5))
    print()

    print("Descricao")
    print(data.DESCR)


if __name__ == "__main__":
    main()
