from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle



def train_model():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = dataset.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = svm.SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'dataset': dataset,
        'feature_names': feature_names
    }

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)

    return model, scaler, data