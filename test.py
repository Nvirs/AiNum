import numpy as np
import matplotlib.pyplot as plt
import utils

kepek,cimkek = utils.load_dataset()

w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5,(10,20))
b_i_h = np.zeros(20, 1)
b_h_o = np.zeros(10, 1)

tanulasi_rata = 0.01
helyes_osztalyozas = 0
veszteseg = 0
epochok = 3
for epoch in range(epochok):
    for img, l in zip(kepek, cimkek):
        img = np.reshape(img,(-1,1))
        l = np.reshape(l, (-1, 1))
        
        #Tovabbitasi bemenet -> rejtett
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))
        #Tovabbitasi rejtett -> kimenet
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o.pre))
        
        #Hiba szamitas
        veszteseg += 1 / len(o) * np.sum((o - l ) ** 2, axis = 0)
        helyes_osztalyozas += int(np.argmax(o) == np.argmax(l))
        
        #Visszaporsitas kimenet -> rejtett
        delta_o = o -l 
        w_h_o += -tanulasi_rata * delta_o @ np.transpose(h)
        b_h_o += -tanulasi_rata * delta_o
        # Visszaparositas rejtett -> bemenet
        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1-h))
        w_i_h += -tanulasi_rata * delta_h @ np.transpose(img)
        b_i_h += -tanulasi_rata * delta_h

    #pontossag kimutatas erre a ratara
    print(f"veszteség: {round((veszteseg[0] / kepek.shape[0])* 100, 2)}%")
    print(f"pontosság: {round((helyes_osztalyozas[0] / kepek.shape[0])* 100, 2)}%")
    helyes_osztalyozas = 0
    veszteseg = 0
    
exit(0)

#Eredmenyek kimutatasa
#while True:
 #   index = int(input("Add meg a számod (0 - 59999): "))
  #  img = kepek[index]
# plt.imshow(img.reshape(28,28), cmap="Greys")

#   img.shape += (1,)
    #Tovabbitasi bemenet -> rejtett
#   h_pre = b_i_h + w_i_h @ img.reshape(784, 1)
#   h = 1 / (1 + np.exp(-h_pre))
#   # Tovabbitasi bemenet -> rejtett
#   o_pre = b_h_o + w_h_o @ h
#   o = 1 / (1 + np.exp(-o_pre))

#   plt.title(f"ha ez megvalosult akkor {o.argmax()} :)")
#  plt.show()
##
    
        