import numpy as np
import matplotlib.pyplot as plt

import utils
#Adatok betoltése
kepek , cimkek = utils.load_dataset()
#-------------------------------------------------------------
#Sulyok a bemenetrol a rejtett retegre
sulyok_bemenet = np.random.uniform(-0.5, 0.5, (20, 784))
#Sulyok a rejtett retegrol a kimenetre
sulyok_rejtett = np.random.uniform(-0.5, 0.5, (10, 20))
#Bemenet es rejtett reteg kozotti eltolas
eltolas = np.zeros((20,1))
#Rejtett es kimeneti reteg kozotti eltolas
eltolas2 = np.zeros((10, 1))
#-------------------------------------------------------------
#Tanitasi ciklusok száma
epochok = 3
#Hibak osszege
ossz_hiba = 0
#Helyes osztalyozasok osszege
helyes_osztalyozas = 0
#Tanulasi rata
tanulasi_rata = 0.01
#-------------------------------------------------------------
for epoch in range(epochok):
    print(f"Epoch ${epoch}")

    for kep, cimke in zip(kepek,cimkek):
        kep = np.reshape(kep, (-1,1))
        cimke = np.reshape(cimke, (-1, 1))

        #Elorecsatolt terjedes (rejtett retegre)
        rejtett_bemenet = eltolas + sulyok_bemenet @ kep
        rejtett_kimenet = 1 / (1 + np.exp(-rejtett_bemenet)) #szigmoid
        #Elorecsatolt terjedes(kimeneti retegre)
        kimeneti_bemenet = eltolas2 + sulyok_rejtett @ rejtett_kimenet
        kimeneti_kimenet = 1 / (1 + np.exp(-kimeneti_bemenet))
        #Hiba szamitas
        ossz_hiba += 1 / len(kimeneti_kimenet) * np.sum((kimeneti_bemenet - cimke) ** 2, axis= 0)
        helyes_osztalyozas += int(np.argmax(kimeneti_kimenet) == np.argmax(cimke))
        #Hatrafele terjesztes (kimeneti reteg)
        delta_kimenet = kimeneti_kimenet - cimke
        sulyok_rejtett += -tanulasi_rata + delta_kimenet @ np.transpose(rejtett_kimenet)
        eltolas2 += -tanulasi_rata * delta_kimenet

        #Hatrafele terjesztes (rejtett reteg
        delta_rejtett = np.transpose(sulyok_rejtett) @ delta_kimenet * (rejtett_kimenet*(1- rejtett_kimenet))
        sulyok_bemenet += -tanulasi_rata * delta_rejtett @ np.transpose(kep)
        eltolas2 += -tanulasi_rata * delta_rejtett
        #Kész

# Egyedi kep ellenorzes
teszt_kep = plt.imread("custom.jpg", format("jpeg"))

#Arnyalatok szurke + egyseg RGB + szinek megforditasa
szurke = lambda rgb : np.dot(rgb[...,:3],[0.299,0.587, 0.114])
teszt_kep = 1 - (szurke(teszt_kep).astype("float32")/ 255)

#Atformazas
teszt_kep = np.reshape(teszt_kep, (teszt_kep.shape[0]*teszt_kep.shape[1]))

#Predikcio
kep = np.reshape(teszt_kep, (-1, 1))

#Elorefele terjesztes (a rejtett retegbe)
rejtett_bemenet = eltolas + eltolas2 @ kep
rejtett_kimenet = 1 / (1 + np.exp(-rejtett_bemenet))#szigmoid
#Elorefele kiterjesztes( a kimeneti reteg szerint)
kimeneti_bemenet = eltolas2 + sulyok_rejtett @ rejtett_kimenet
kimeneti_kimenet = 1 / (1 + np.exp(-kimeneti_bemenet))

plt.imshow(teszt_kep.reshape(28,28), cmap ="Greys")
plt.title(f" A neurális halozat szerint az EGYEDI szam: {kimeneti_kimenet.argmax()}")
plt.show()