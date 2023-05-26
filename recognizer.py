import my_pca
import lbph
import eigen_face


def predict(option):
    if option == 0:
        my_pca.predict()
    elif option == 1:
        lbph.predict()
    elif option == 2:
        eigen_face.predict()
