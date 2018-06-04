# -*- coding: utf-8 -*-
"""
@author: Ismael Marín Molina
"""
#Science kit learn files
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals import 	joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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

def Imputation(X):
    missings = np.where(X == 'unknown')
    row = missings[0]
    Ypos = missings[1]
    columns = np.unique(Ypos)
    CategoricalAtribute = [1,2,3,4,5,6,7,8,9,14]
    Rest = np.setdiff1d(CategoricalAtribute, columns)

    numberCha = []
    classesNumber = np.zeros(X.shape[0])
    for column in columns:
        numberCha.append(len(row[Ypos == column])/X.shape[0])
        le = preprocessing.LabelEncoder()
        X[:,column] = le.fit_transform(X[:,column])
        index, = np.where(le.classes_ == 'unknown')
        classesNumber[column] = index[0]

    for rest in Rest:
        le = preprocessing.LabelEncoder()
        X[:,rest] = le.fit_transform(X[:,rest])

    for i in range(len(columns)):
        print("La columna {} tiene un {}% de valores perdidos".format(columns[i],numberCha[i]*100))

    KnnImputer = np.array([columns[i] for i in range(len(numberCha)) if numberCha[i] > 0.01])
    SimpleImputer = np.setdiff1d(columns, KnnImputer)
    print()
    print("Las columnas seleccionadas para la imputacion con KNN: {}".format(KnnImputer))
    print("Las columnas seleccionadas para la imputacion simple : {}".format(SimpleImputer))

    for i in SimpleImputer:
        Simple = preprocessing.Imputer(missing_values=classesNumber[i], strategy='most_frequent')
        X[:,SimpleImputer] = Simple.fit_transform(X[:,SimpleImputer])

    for i in KnnImputer:
        Knn = KNeighborsClassifier(n_neighbors = 3, p=2, algorithm='kd_tree', leaf_size=12, n_jobs=-1)
        Label = X[:,i]
        row_miss = row[Ypos == i]
        normal_row = np.setdiff1d(range(len(Label)), row_miss)
        Caracteristicas = np.delete(X, i, axis=1)
        Test_Label  = Label[row_miss]
        Train_Label = Label[normal_row].astype(np.int64)
        Knn.fit(Caracteristicas[normal_row,:],Train_Label)
        X[row_miss,i] = Knn.predict(Caracteristicas[row_miss,:])
        print("{}:Tamaño conjunto de Test: {}".format(i,Test_Label.shape))
        print("X: {}: :Caracteristicas: {}".format(X.shape,Caracteristicas.shape))

    return X

def plot_Importance(rfc, Xres):
    importance = rfc.feature_importances_
    indices = np.argsort(importance)

    indices = indices[np.where(importance > 0.05)]

    plt.figure()
    plt.title("Feature importances (> 0.05)")
    plt.bar(range(len(indices)), importance[indices],
           color="r", align="center")
    plt.xticks(range(len(indices)), indices, rotation='vertical')
    plt.xlim([-1, len(indices)])
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
maxiter = 10000
splits = 5
siz_Test = 0.3
saveName = 'Bank.pkl'
nameFeature = "Feature.npy"
nameLabel = "Label.npy"
path = "data/"
CategoricalAtribute = [1,2,3,4,5,6,7,8,9,14]
RealAtribute = np.setdiff1d(range(20), CategoricalAtribute)

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

for i in np.unique(Validation_Label):
    print("Clase {} numero de instancias: {}".format(i,len(np.where(Validation_Label == i)[0])))

print()


# Eliminar las instancias donde la duración vale 0

durationIndex = 10

removeIndices = np.where( Validation_Feature[:,durationIndex] == 0 )[0]
Validation_Feature = np.delete( Validation_Feature, removeIndices, axis = 0 )
Validation_Label = np.delete( Validation_Label, removeIndices, axis = 0 )

removeIndices = np.where( Test_Feature[:,durationIndex] == 0 )[0]
Test_Feature = np.delete( Test_Feature, removeIndices, axis = 0 )
Test_Labelt = np.delete( Test_Labelt, removeIndices, axis = 0 )


#############################Valores Perdidos###############################

######Seleccion de parametros a usar#####
parameters = [{'Model__C':[1.0,1e-6], 'Model__kernel':['rbf','poly','sigmoid'], 'Model__decision_function_shape':['ovo','ovr']}]

#######Preprocesado de los datos, Scalado y Categorizado################
Validation_Feature[:,RealAtribute] = preprocessing.StandardScaler().fit(Validation_Feature[:,RealAtribute]).transform(Validation_Feature[:,RealAtribute])

missings = np.where(Validation_Feature == 'unknown')
print("Tenemos una catidad de: {} valores perdidos".format(len(missings[1])))
print("Distribucion:")
Validation_Feature = Imputation(Validation_Feature)
missings = np.where(Validation_Feature == 'unknown')
print("Tras el procesamiento: {} valores perdidos".format(len(missings[1])))

Validation_Feature = preprocessing.OneHotEncoder(categorical_features=CategoricalAtribute, handle_unknown='ignore').fit_transform(Validation_Feature).todense()
#Datos guardados en formato COOmatrix si se quieren ver en tamaño normal usar XXX.todense()

sm = SMOTE(ratio='minority',random_state=seed, k_neighbors=3)
Xres, Yres = sm.fit_sample(Validation_Feature, Validation_Label)

print()
print("Tras el Balanceado")
print("Dimension de la nueva matriz:{},     Dimension:{}".format(Xres.shape,Validation_Feature.shape))
for i in np.unique(Yres):
    print("Clase {} numero de instancias: {}".format(i,len(np.where(Yres == i)[0])))

print()

rfc = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=30, min_samples_leaf=10, max_features="sqrt").fit(Xres,Yres)

print("Scorer en la particion train: {}".format(rfc.score(Validation_Feature, Validation_Label)))

importance = rfc.feature_importances_
indices = np.argsort(importance)

indices = indices[np.where(importance > 0.05)]

print( "Índices: {}".format( indices ) )

prediction = rfc.predict(Xres)
prediction_index = np.where(Yres == prediction)[0]
prediction_index = prediction_index.reshape(-1)
indices = indices.reshape(-1)

print( indices )

Xres = Xres[prediction_index,:]
Xres = Xres[:,indices]
Yres = Yres[prediction_index]

##########################################################################

print( Xres[:,2].mean() )

######Pipe donde incluimos Escalado y Modelo##########
pipe = Pipeline([('Model',SVC(max_iter=maxiter))])
grid = GridSearchCV(pipe, param_grid=parameters, cv=splits)


"""
#####Ajustado de los datos####
grid.fit(Xres, Yres)
Save(grid,saveName) #Guardado del modelo para un uso más rapido en futuros momentos

"""

####Impresion de los datos####
print("Mejor valor de la cross validation: {:.4f}".format(grid.best_score_))
print("Mejores parametros: {}".format(grid.best_params_))

"""
"""

"""
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
