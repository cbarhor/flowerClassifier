#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 20:30:05 2021

@author: cbarhor
"""
import cv2
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance as dist



class FeatureExtractor:
    def __init__(self, hist_mode = 'hsv'):
        """
        La clase FeatureExtractor se emplea para extraer las caracteristicas de
        las imagenes del conjunto de datos. Tanto aquellas que estan etiquetadas
        por las que no

        Parameters
        ----------
        hist_mode : str, optional
            Opcion empleada para realizar la extraccion del 
            histograma de cada imagen en funcion de sus canales RBG o HSV.
            Se ha comprobado que se obtienen mejores resultados con los histogramas
            en formato hsv. Esto puede ser porque se representan mejor las caracteristicas
            relevantes, especialmente en el canal H (matiz). 
            RGB = red, green, blue
            HSV = hue, saturation, value: matiz, saturacion, luminosidad
            The default is 'hsv'.

        Returns
        -------
        None.

        """
        self.hist_mode = hist_mode
        self.getLabelledSamples()
        self.readAndPrepare()
        
        
    def extractFlatten(self,image, size=(32, 32)):
        """
        Metodo creado para convertir las imagenes en vectores de caracteristicas
        directamente. En general, esto es mala idea ya que se pierden las 
        caracteristicas espaciales

        Parameters
        ----------
        image : imagen cv2
            Imagen a convertir en feature.
        size : tuple, optional
            Tamaño de la imagen antes de ser convertida a feature. The default is (32, 32).

        Returns
        -------
        list
            Vector de caracteristicas de la imagen.

        """
        return cv2.resize(image, size).flatten()
    
    def extractHist(self,image, bins=(8, 8, 8)):
        """
        Metodo empleado para extraer el vector de caracteristicas
        de una imagen a partir de su histograma de color

        Parameters
        ----------
        image : imagen cv2
            imagen cuyo vector se quiere extraer.
        bins : tuple, optional
            numero de bins que se emplean por cada canal en la creacion del 
            histograma. The default is (8, 8, 8).

        Returns
        -------
        features : list
            vector de caracteristicas del color.

        """
        if self.hist_mode == 'rgb':
            #Histograma RGB
            hist = cv2.calcHist([image],[0, 1, 2], None, bins,[0, 256, 0, 256, 0, 256])
        else:
            #Histograma HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])

        # Se normaliza el histograma
        cv2.normalize(hist, hist)
        #Se convierte en un vector (1,)
        features = hist.flatten()

        
        return features
    
    def showHist(self, image, name):
        """
        Metodo empleado para realizar una inspeccion visual de los histogramas
        del conjunto etiquetado

        Parameters
        ----------
        image : imagen cv2
            imagen cuyo histograma se quiere graficar.
        name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.hist_mode == 'hsv':
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            chans = cv2.split(hsv)
            colors = ("h", "s", "v")
        else: 
            chans = cv2.split(image)
            colors = ("r", "g", "b")            
        plt.figure()
        plt.title("Histograma por canales")
        plt.xlabel("Bins")
        plt.ylabel("# de Pixeles")
        features = []
        for (chan, color) in zip(chans, colors):
            if color == "h":
                hist = cv2.calcHist([chan], [0], None, [180], [0, 180])
            else:
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.extend(hist)
            plt.plot(hist)
            plt.legend(colors)
            plt.xlim([0, 256])
            plt.title(name)
        plt.show()

    
    def readAndPrepare(self, datasetPath="./Data/technical_test_images/"):
        """
        Metodo empleado para leer las imagenes que se quieren clasificar
        y extraer sus vectores de cacteristicas.
        En images se guardan las imagenes de la siguiente manera:
            {nombre_imagen:imagen}
        En index se guardan los vectores de caracteristicas de la siguiente 
        manera: 
            {nombre_imagen:vector}

        Parameters
        ----------
        datasetPath : str, optional
            ruta relativa del conjunto de datos. The default is "./Data/technical_test_images/".

        Returns
        -------
        None.

        """
        index = {}
        images = {}
        print("Extrayendo histogramas del dataset...")
        for imagePath in glob.glob(datasetPath + "*.jpg"):
            filename = str(imagePath).split('/')[-1]
            filename = filename.split('.')[0]
            image = cv2.imread(imagePath)
            images[filename] = image
            index[filename] = self.extractHist(image)  
        self.index = index
        self.images = images
        print("[Hecho]")
        
    def getLabelledSamples(self, path='./Data/technical_test_images/Manual/'):  
        """
        Metodo empleado para extraer los vectores de caracteristicas de las 
        imagenes etiquetadas

        Parameters
        ----------
        path : str, optional
            ruta relativa hasta las imagenes etiquetadas. The default is './Data/technical_test_images/Manual/'.

        Returns
        -------
        None.

        """
        #Extraccion de los vectores para el conjunto de rosas:
        rose_images = {}
        rose_index = {}
        print("Extrayendo histogramas de las rosas etiquetadas...")
        for imagePath in glob.glob(path + "Rosas/*.jpg"):
            filename = str(imagePath).split('/')[-1]
            filename = filename.split('.')[0]
            image = cv2.imread(imagePath)
            rose_images[filename] = image
            rose_index[filename] = self.extractHist(image)  
            self.showHist(image, "Rosa: "+filename)
        self.rose_index = rose_index
        self.rose_images = rose_images       
        print("[Hecho]")

        #Extraccion de los vectores para el conjunto de girasoles:
        sunflower_images = {}
        sunflower_index = {}
        print("Extrayendo histogramas de los girasoles etiquetados...")
        for imagePath in glob.glob(path + "Girasoles/*.jpg"):
            filename = str(imagePath).split('/')[-1]
            filename = filename.split('.')[0]
            image = cv2.imread(imagePath)
            sunflower_images[filename] = image
            sunflower_index[filename] = self.extractHist(image) 
            self.showHist(image, "Girasol: "+filename)
        self.sunflower_index = sunflower_index
        self.sunflower_images = sunflower_images 
        print("[Hecho]")
    
    
class HistogramMethods:
    def __init__(self):
        """
        Esta clase se emplea para realizar la clasificacion de las imagenes
        en funcion de la distancia entre sus histogramas y los histogramas
        de las imagenes etiquetadas.
        Se pueden emplear varios metodos diferentes

        Returns
        -------
        None.

        """
        self.DIST_METHODS = {
            "Euclidean": dist.euclidean,
            "Manhattan": dist.cityblock,
            "Chebysev": dist.chebyshev
            }
        self.HIST_METHODS = {
            "Correlation": cv2.HISTCMP_CORREL,
            "Chi-Squared": cv2.HISTCMP_CHISQR,
            "Intersection": cv2.HISTCMP_INTERSECT,
            "Hellinger": cv2.HISTCMP_BHATTACHARYYA
            }
        

        
      
        
    def useHistComparations(self, index, sunflower_index, rose_index, methodUsed = "Intersection"):
        """
        Metodo empleado para calcular la distancia media entre los histogramas
        empleando las metricas de distancia implementadas por OpenCV

        Parameters
        ----------
        index : dict
            diccionario que contiene los vectores de caracteristicas de las 
            imagenes a clasificar.
        sunflower_index : dict
            diccionario que contiene los vectores de caracteristicas de las 
            imagenes etiquetadas de girasoles.
        rose_index : dict
            diccionario que contiene los vectores de caracteristicas de las 
            imagenes etiquetadas de girasoles.
        methodUsed : str, optional
            Metodo que se desea emplear. The default is "Euclidean".

        Returns
        -------
        results : dict
            diccionario con los resultados. Formato {nombre_imagen:clase}

        """
        print("Empleando la comparacion de histogramas para clasificacion. Metodo: {}".format(methodUsed))
        method = self.HIST_METHODS[methodUsed]
        results = {}
        reverse = False
        #Las metricas Correlation e Intersection se comportanjusto al reves que
        #las demas: cuanto mayor es la distancia, mas probabilidad hay de que 
        #pertenezca al conjunto
        if self.HIST_METHODS.get(method) in ("Correlation", "Intersection"):
            reverse = True
        # Se itera sobre cada vector de caracteristicas a clasificar
        for (k, hist) in index.items():

            rose_dists = []
            sunflower_dists = []
            #Se obtienen 5 metricas de distancia entre la imagen a clasificar y
            #las 5 imagenes de rosas etiquetadas
            for (i, rose_hist) in rose_index.items():
                d = cv2.compareHist(rose_hist, hist, method)
                rose_dists.append(d)
            #Se obtienen 5 metricas de distancia entre la imagen a clasificar y
            #las 5 imagenes de girasoles etiquetados                
            for (i, sunflower_hist) in sunflower_index.items():
                d = cv2.compareHist(sunflower_hist, hist, method)
                sunflower_dists.append(d)  

            #Se decide a que clase pertenece en funcion de las distancias medias
            if not reverse:
                if np.mean(rose_dists) < np.mean(sunflower_dists):
                    results[k] = 'rosa'
                else:
                    results[k] = 'girasol'
            else:
                if np.mean(rose_dists) > np.mean(sunflower_dists):
                    results[k] = 'rosa'
                else:
                    results[k] = 'girasol'     

            
        print("[Hecho]")
        return results
            
    def useHistDistances(self, index, sunflower_index, rose_index, methodUsed = 'Euclidean'):
        """
        Metodo empleado para calcular la distancia media entre los histogramas
        empleando las metricas de distancia implementadas en Scipy

        Parameters
        ----------
        index : dict
            diccionario que contiene los vectores de caracteristicas de las 
            imagenes a clasificar.
        sunflower_index : dict
            diccionario que contiene los vectores de caracteristicas de las 
            imagenes etiquetadas de girasoles.
        rose_index : dict
            diccionario que contiene los vectores de caracteristicas de las 
            imagenes etiquetadas de girasoles.
        methodUsed : str, optional
            Metodo que se desea emplear. The default is "Euclidean".

        Returns
        -------
        results : dict
            diccionario con los resultados. Formato {nombre_imagen:clase}

        """        
        print("Empleando la distancia entre histogramas para clasificacion. Metodo: {}".format(methodUsed))

        method = self.DIST_METHODS[methodUsed]
        results = {}
        #Se itera sobre cada vector de caracteristicas de las imagenes a clasificar
        for (k, hist) in index.items():
            rose_dists = []
            sunflower_dists = []
            #Se obtienen 5 metricas de distancia entre la imagen a clasificar y
            #las 5 imagenes de rosas etiquetadas            
            for (i, rose_hist) in rose_index.items():
                d = method(rose_hist, hist)
                rose_dists.append(d)
            #Se obtienen 5 metricas de distancia entre la imagen a clasificar y
            #las 5 imagenes de girasoles etiquetados                
            for (i, sunflower_hist) in sunflower_index.items():
                d = method(sunflower_hist, hist)
                sunflower_dists.append(d)      
            #Se decide a que clase pertenece en funcion de las distancias medias
            if np.mean(rose_dists) < np.mean(sunflower_dists):
                results[k] = 'rosa'
            else:
                results[k] = 'girasol'                

        print("[Hecho]")
        return results
            
class UnsupervisedMethods:
    def __init__(self, n_clusters = 2):
        """
        Esta clase se emplea para realizar la clasificacion de las imagenes
        haciendo uso de metodos de aprendizaje no supervisado.
    

        Parameters
        ----------
        n_clusters : int, optional
            numero de clasters. The default is 2.

        Returns
        -------
        None.

        """
        self.clusters = n_clusters 
    
    def prepareData(self, index):
        """
        Metodo empleado para preparar los datos de entrenamiento del modelo.
        Notese que no existen labels (no supervisado)

        Parameters
        ----------
        index : dict
            diccionario con los vectores de caracteristicas de las imagenes
            a clasificar.

        Returns
        -------
        None.

        """
        features = [] 
        for (name, hist) in index.items():
            features.append(hist)              
        self.features = features

           
    def useKMeans(self,index):   
        """
        Metodo empleado para realizar la clasificacion de las imagenes
        empleando el algortimo K-Means.
        Se entrena el modelo y con el se realizan las predicciones

        Parameters
        ----------
        index :  dict
            diccionario con los vectores de caracteristicas de las imagenes
            a clasificar.

        Returns
        -------
        results : dict
            diccionario con los resultados obtenidos.

        """
        results = {}
        model = KMeans(n_clusters=self.clusters)      
        fitted = model.fit(self.features)
        predictions = fitted.labels_
        for ((name, hist), prediction) in zip(index.items(), predictions):
            if prediction == 0:              
                results[name] = 'rosa'
            elif prediction == 1:
                results[name] = 'girasol'  
        return results
  
        
    def useGMM(self, index):
        """
        Metodo empleado para realizar la clasificacion de las imagenes
        empleando el algortimo GMM.
        Se entrena el modelo y con el se realizan las predicciones

        Parameters
        ----------
        index :  dict
            diccionario con los vectores de caracteristicas de las imagenes
            a clasificar.

        Returns
        -------
        results : dict
            diccionario con los resultados obtenidos.

        """        
        results = {}
        gm = GaussianMixture(n_components=self.clusters, random_state=0)
        gm.fit(self.features)
        predictions = gm.predict(self.features)        
        for ((name, hist), prediction) in zip(index.items(), predictions):
            if prediction == 0:              
                results[name] = 'rosa'
            elif prediction == 1:
                results[name] = 'girasol' 
        return results        
    
class SupervisedMethods:
    def __init__(self, n_clusters = 3):
        """
        Esta clase se emplea para realizar la clasificacion de las imagenes
        haciendo uso de metodos de aprendizaje supervisado.
    

        Parameters
        ----------
        n_clusters : int, optional
            numero de clusters. The default is 3.

        Returns
        -------
        None.

        """        
        self.clusters = n_clusters
        
  
    def useKNN(self, index):
        """
        Metodo empleado pararealizar la clasificacion de las imagenes
        empleando el algortimo K-NN.
        Se entrena el modelo y con el se realizan las predicciones

        Parameters
        ----------
        index :  dict
            diccionario con los vectores de caracteristicas de las imagenes
            a clasificar.

        Returns
        -------
        results : dict
            diccionario con los resultados obtenidos.

        """
        results = {}
 
        model = KNeighborsClassifier(n_neighbors=self.clusters, n_jobs=-1)
        model.fit(self.train_features, self.labels)  

        predictions = model.predict(self.features)
        for ((name, hist), prediction) in zip(index.items(), predictions):
            results[name] = prediction
        return results
    
    def useSVM(self, index):
        """
        Metodo empleado pararealizar la clasificacion de las imagenes
        empleando el algortimo SVM.
        Se entrena el modelo y con el se realizan las predicciones

        Parameters
        ----------
        index :  dict
            diccionario con los vectores de caracteristicas de las imagenes
            a clasificar.

        Returns
        -------
        results : dict
            diccionario con los resultados obtenidos.

        """        
        results = {}
        modelo = SVC(C = 100, kernel = 'linear', random_state=123)
        modelo.fit(self.train_features, self.labels)
        predictions = modelo.predict(self.features)
        for ((name, hist), prediction) in zip(index.items(), predictions):
            results[name] = prediction
        return results
    
    def prepareData(self, index, sunflower_index, rose_index):
        """
        Metodo empleado para preparar los vectores de caracteristicas
        tanto de las imagenes a clasificar como de las imagenes etiquetadas

        Parameters
        ----------
        index : dict
            diccionario con los vectores de caracteristicas de las imagenes a
            clasificar.
        sunflower_index : dict
            diccionario con los vectores de caracteristicas de las imagenes etiquetadas
            de girasoles.
        rose_index : dict
            diccionario con los vectores de caracteristicas de las imagenes de
            rosas etiquetadas

        Returns
        -------
        None.

        """
        features = []
        train_features = []
        labels = []
        
        for (name, rose_hist) in rose_index.items():
            train_features.append(rose_hist)
            labels.append('rosa')
        for (name, sunflower_hist) in sunflower_index.items():
            train_features.append(sunflower_hist)
            labels.append('girasol')  
        for (name, hist) in index.items():
            features.append(hist)  
            
        self.features = features
        self.labels = labels
        self.train_features = train_features
        
    
class Pipeline:
    def __init__(self):
        """
        Esta clase se emplea para gestionar el uso del resto de algoritmos 
        y metodos, marcando el flujjo del programa

        Returns
        -------
        None.

        """
        self.prepareFolders()
        self.featureManager = FeatureExtractor(hist_mode = 'hsv')
        
    def prepareFolders(self):
        """
        Metodo empleado para crear las carpetas necesarias para poder ejecutar
        el resto del script

        Returns
        -------
        None.

        """
        print("Creando carpetas necesarias...")
        if not os.path.isdir('./Resultados/'):  
            os.mkdir('./Resultados/')
    
    def writeToTxtFile(self, method_name, results):
        """
        Metodo empleado para generar un fichero de texto con los resultados
        obtenidos. 
        Su formato es el siguiente:
            nombre_imagen -> clase
        Se ha optado por este metodo en lugar de copiar todas las imagenes
        en diferentes carpetas para evitar el crecimiento exponencial
        del numero de imagenes con cada ejecucion

        Parameters
        ----------
        method_name : str
            nombre del txt. Idealmente, el metodo empleado para poder compararlos.
        results : dict
            diccionario con los resultados.

        Returns
        -------
        None.

        """
        print("Escribiendo resultados en {}".format('./Resultados/'+method_name+'.txt'))
        file = open('./Resultados/'+method_name+'.txt', 'w')
        for (image_name, classif) in results.items():
            file.write(str(image_name)+" -> "+str(classif)+"\n")
        file.close()
        
    def moveImagesToFolders(self, file_path):
        """
        Metodo empleado para copiar las imagenes en dos carpetas segun sus clases
        en base asu clasificacion. Usar solo cuando se este seguro de que la clasificacion
        es buena. 

        Parameters
        ----------
        file_path : str
            ruta relativa donde se encuentra el fichero de resultados.

        Returns
        -------
        None.

        """
        print("Copiando imagenes a carpetas segun su clasificacion...")
        roses_folder = './Resultados/Rosas/'
        sunflowers_folder = './Resultados/Girasoles/'
        #Crea las carpetas necesarias, si procede
        if not os.path.isdir(roses_folder):  
            os.mkdir(roses_folder)    
        if not os.path.isdir(sunflowers_folder):  
            os.mkdir(sunflowers_folder)   
        #Lee el fichero de resultados y obtiene el nombre de la
        #imagen y su clasificacion
        file = open(file_path, 'r')
        lines = file.readlines()
        for line in lines:
            image, classif = line.split(' -> ')
            image_path = "./Data/technical_test_images/"+image+".jpg"
            classif = re.sub(r"[\n\t\s]*", "", classif)
            #Se copia la imagen en la carpeta de girasoles
            if classif == 'girasol':
                shutil.copyfile(image_path, sunflowers_folder+image+".jpg")
            #Se copia la imagen en la carpeta de rosas    
            elif classif == 'rosa':
                shutil.copyfile(image_path, roses_folder+image+".jpg")
        print("[Hecho]")
 
        
        
    def checkNumbers(self, results):
        """
        Metodo empleado para obtener resultados cuantitativos sobre 
        cuantas imagenes se han clasificado en girasoles y cuantas como rosas

        Parameters
        ----------
        results : dict
            diccionario con los resultados.

        Returns
        -------
        None.

        """
        sunflowers = 0
        roses = 0
        for (_, classif) in results.items():
            if classif == 'girasol':
                sunflowers += 1
            elif classif == 'rosa':
                roses += 1
        print("Imagenes clasificadas como rosa: {}".format(roses))
        print("Imagenes clasificadas como girasol: {}".format(sunflowers))
            
    def useHistMethods(self):
        """
        Metodo diseñado para emplear de una forma facil los metodos de 
        distancia de histograma.
        Primero se clasifican las imagenes
        Despues se escribe el txt de resultados
        Y finalmente se imprimen los resultados por pantalla
        
        Se prueba un metodo de OpenCV y otro de Scipy

        Returns
        -------
        None.

        """
        histMeth = HistogramMethods()
        # self.DIST_METHODS = {
        #     "Euclidean": dist.euclidean,
        #     "Manhattan": dist.cityblock,
        #     "Chebysev": dist.chebyshev
        #     }
        # self.HIST_METHODS = {
        #     "Correlation": cv2.HISTCMP_CORREL,
        #     "Chi-Squared": cv2.HISTCMP_CHISQR,
        #     "Intersection": cv2.HISTCMP_INTERSECT,
        #     "Hellinger": cv2.HISTCMP_BHATTACHARYYA
        #     }        
        
        histCompMethod = "Correlation"
        histDistMethod = 'Euclidean'
        results1 = histMeth.useHistComparations(self.featureManager.index, self.featureManager.sunflower_index, self.featureManager.rose_index, methodUsed = histCompMethod)
        self.writeToTxtFile(histCompMethod, results1)
        self.checkNumbers(results1)
        results2 = histMeth.useHistDistances(self.featureManager.index, self.featureManager.sunflower_index, self.featureManager.rose_index, methodUsed = histDistMethod )
        self.writeToTxtFile(histDistMethod, results2)
        self.checkNumbers(results2)
        
    def useSupervisedMethods(self):
        """
        Metodo diseñado para emplear de una forma facil los metodos de 
        aprendizaje supervisado.
        Primero se clasifican las imagenes
        Despues se escribe el txt de resultados
        Y finalmente se imprimen los resultados por pantalla
        
        Se prueba el metodo KNN y el SVM despues

        Returns
        -------
        None.

        """        
        supervised = SupervisedMethods()
        supervised.prepareData( self.featureManager.index, self.featureManager.sunflower_index, self.featureManager.rose_index)
        results1 = supervised.useKNN(self.featureManager.index)
        self.writeToTxtFile("KNN", results1)
        self.checkNumbers(results1)      
        results2 = supervised.useSVM(self.featureManager.index)
        self.writeToTxtFile("SVM", results2)
        self.checkNumbers(results2)     
        
    def useUnsupervisedMethods(self):
        """
        Metodo diseñado para emplear de una forma facil los metodos de 
        aprendizaje no supervisado.
        Primero se clasifican las imagenes
        Despues se escribe el txt de resultados
        Y finalmente se imprimen los resultados por pantalla
        
        Se prueba el metodo K-Means y el GMM despues

        Returns
        -------
        None.

        """         
        unsupervised = UnsupervisedMethods()
        unsupervised.prepareData( self.featureManager.index)
        results1 = unsupervised.useKMeans(self.featureManager.index)
        self.writeToTxtFile("KMeans", results1)
        self.checkNumbers(results1)          
        results2 = unsupervised.useGMM(self.featureManager.index)
        self.writeToTxtFile("GMM", results2)
        self.checkNumbers(results2)  


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.useHistMethods() 
    pipeline.useSupervisedMethods()  
    pipeline.useUnsupervisedMethods()  
    pipeline.moveImagesToFolders('./Resultados/SVM.txt') 
        
        
        
        
        
