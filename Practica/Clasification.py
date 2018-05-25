# -*- coding: utf-8 -*-
"""
@author: Ismael Marín Molina
"""
#Science kit learn files
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.externals import 	joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#Others files
import matplotlib.pyplot as plt
import numpy as np
import itertools #necesaria para el dibujo de la matrix y la curva ROC
from scipy import interp #necesario para la curva ROC
import os #para buscar archivos
import pandas as pd

#Catcher Warnings
import warnings
from sklearn.exceptions import DataConversionWarning

def plot_security(cm, Xtest, Ytest):

    value = cm.predict_proba(Xtest)
    value_8 = value[:,0]

    points = value_8[Ytest == 0]
    plt.scatter(range(len(points)), points)
    plt.title("Numeros y seguridad")
    plt.xlabel("Indice")
    plt.ylabel("Seguridad")

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Valor')
    plt.xlabel('Prediccion')
    pass

def plot_ROC_multiclass(XTest, YTest, clf ):
    YTest = preprocessing.label_binarize(YTest, np.unique(YTest))
    n_classes = YTest.shape[1]
    YScor = clf.decision_function(XTest)
    lw = 2

    fpr = dict()
    tpr = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(YTest[:,i], YScor[:,i])

    fpr["Micro"], tpr["Micro"], _ = roc_curve(YTest.ravel(), YScor.ravel())

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    #Media de los datos
    mean_tpr /= n_classes
    fpr["Macro"] = all_fpr
    tpr["Macro"] = mean_tpr

    plt.figure()
    plt.plot(fpr["Micro"],tpr["Micro"],
             label='Media a la baja',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["Macro"],tpr["Macro"],
             label='Media a la alta',
             color='navy', linestyle=':', linewidth=4)

    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue', 'lime', 'crimson', 'lightpink', 'darkgreen', 'salmon', 'sienna', 'bisque'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i],tpr[i], color=color, lw=lw,
                 label='ROC class {}'.format(i))

    plt.plot([0,1],[0,1],'k--',lw=lw)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')
    plt.title('ROC multiclase')
    plt.legend(loc="lower right")
    pass

#######################################Utility#####################################

def Find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return True
    return False

def Save(cls,name):
    joblib.dump(cls,name)
    pass

def Create(names):

    data = pd.read_csv('data/bank-additional-full.csv', sep=';')
    Label = np.array(data['y'])
    Features = data.values
    Features = Features[:,0:len(list(data))-1]

    np.save('data/Label.npy', Label)
    np.save('data/Feature.npy', Features)

    pass

#################################################################################
#if __name__ == 'main':
print(__doc__)
seed = 50627728
splits = 5
siz_Test = 0.3
saveName = 'Bank.pkl'
nameFeature = "Feature.npy"
nameLabel = "Label.npy"
path = "data/"

if not Find(nameFeature, path=path):
    print("############CREANDO ARCHIVOS NPY##################3")
    Create(names = [nameFeature, nameLabel])

#Para ignorar los errores provocados por la no convergencia de la función
#así como la conversión de los datos y los de desactualizado
#Valores de salida del print acotados a 4 decimales
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

#######Cargado /  Particionado#######
Feature = np.load("data/Feature.npy")
Label   = np.load("data/Label.npy")

Validation_Feature, Test_Feature , Validation_Label, Test_Labelt = train_test_split(
    Feature, Label, stratify=Label, test_size = siz_Test, random_state = seed)

Validation_Label = preprocessing.LabelEncoder().fit_transform(Validation_Label)

"""
######Seleccion de parametros a usar#####
parameters = [{'Model__penalty': ['l1'],'Model__C':[0.9,0.5,0.1]},
				{'Model__penalty': ['l2'],'Model__C':[0.9,0.5,0.1]},
              {'Model__penalty': ['l1'],'Model__C':[0.01,0.02,0.014142]},
              {'Model__penalty': ['l2'],'Model__C':[0.01,0.02,0.014142]}]

######Pipe donde incluimos Escalado y Modelo##########
pipe = Pipeline([('Scale',preprocessing.StandardScaler()), ('Model',LogisticRegression(random_state=seed,max_iter=1000))])
grid = GridSearchCV(pipe, param_grid=parameters, cv=splits)

#####Ajustado de los datos####
grid.fit(Validation_Feature, Validation_Label)
Save(grid,saveName) #Guardado del modelo para un uso más rapido en futuros momentos

####Impresion de los datos####
print("Mejor valor de la cross validation: {:.4f}".format(grid.best_score_))
#print("Valoracion Hiperparametros: {}".format(grid.decision_function))
print("Mejores parametros: {}".format(grid.best_params_))
print("Valor en el test:")
print(classification_report(Test_Label, grid.predict(Test_Feature)))


cls_nam = np.unique(Validation_Label)
plot_confusion_matrix(confusion_matrix(Test_Label,grid.predict(Test_Feature)),
                      cls_nam,
                      normalize=True)
plt.show()

plot_confusion_matrix(confusion_matrix(Validation_Label,grid.predict(Validation_Feature)),
                      cls_nam,
                      normalize=True)
plt.show()

plot_ROC_multiclass(Test_Feature, Test_Label, grid)
plt.show()

plot_security(grid, Validation_Feature, Validation_Label)
plt.show()
"""
