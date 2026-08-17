import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def load_data():
    df = pd.read_csv("data.csv")
    return df


def data_cleaning(df):

    print("Missing Values")
    print(df.isnull().sum())

    print("\nDuplicates:", df.duplicated().sum())

    df.drop_duplicates(inplace=True)
    df.drop("id", axis=1, inplace=True)

    return df


def outlier_visu(df):

    sns.boxplot(x=df["radius_mean"])
    plt.title("Radius Mean")
    plt.show()

    return df


def encode_target(df):

    encoder = LabelEncoder()

    df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

    print("\nTarget mapping:")
    print(dict(zip(
        encoder.classes_,
        encoder.transform(encoder.classes_)
    )))

    return df


def prepare_data(df):

    X = df.drop("diagnosis", axis=1)

    y = df["diagnosis"]

    return X, y

def main():

    df = load_data()

    df = data_cleaning(df)

    # Encode
    df = encode_target(df)

    # Visualize outliers
    df = outlier_visu(df)

    # Prepare X and y
    X, y = prepare_data(df)

    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)


if __name__ == "__main__":
    main()