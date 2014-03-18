#strategija krizic kruzica
from pobjeda import pobjeda
from copy import copy
from random import randint
def slobodna_polja (ploca):
    slobodna=[]
    for i in range (3):
        for j in range (3):
            if ploca [i][j]==-1:
                slobodna.append([i,j])
    return slobodna

def x_polja (ploca):
    x=[]
    for i in range (3):
        for j in range (3):
            if ploca [i][j]==1:
                x.append ([i,j])
    return x

def o_polja(ploca):
    o=[]
    for i in range (3):
        for j in range (3):
            if ploca [i][j]==0:
                o.append ([i,j])
    return o
#izbaci sredine
def izbaci_sredine (polja):
    bez_sredina=[]
    for element in polja:
        if element not in [[0,1],[1,2],[2,1],[1,0]]:
            bez_sredina.append (element)
    return bez_sredina
    
#strategija x pobjeda
def strategija_x (ploca):
    kopija=copy(ploca)
    slobodna=slobodna_polja(kopija)
    broj_slobodnih=len(slobodna)
    polje=slobodna[randint(0,broj_slobodnih-1)]
    if broj_slobodnih==9:
        #prvi potez
        bez_sredina=izbaci_sredine(slobodna)
        polje=bez_sredina [randint(0,len(bez_sredina)-1)]
    elif broj_slobodnih==7:
        #drugi potez
        if [1,1] in slobodna:
            polje=[1,1]
    elif broj_slobodnih==5:
        #treci potez
        pass
    elif broj_slobodnih==3:
        #cetvrti potez
        pass    
    elif broj_slobodnih==1:
        #peti potez
        polje=slobodna[0]
   #Blokada
    for i in range (3):
        for j in range (3):
            if kopija [i][j]==-1:
                kopija [i][j]=0
                if pobjeda (kopija)=="pobjeda o":
                     polje=[i,j]
                kopija[i][j] = -1


#Napad za pobjedu
    for i in range (3):
        for j in range (3):
            if kopija [i][j]==-1:
                kopija [i][j]=1
                if pobjeda (kopija)=="pobjeda x":
                    polje=[i,j]
                kopija[i][j] = -1
    return polje
	
#kruzic strategija
def strategija_o (ploca):
    kopija=copy (ploca)
    slobodna= slobodna_polja (kopija)
    broj_slobodnih=len(slobodna)
    polje=slobodna [randint(0,broj_slobodnih-1)]
    
    if broj_slobodnih==8:
        #prvi potez
        if [1,1] in slobodna:
             polje=[1,1]
    elif broj_slobodnih==6:
        if ploca[1][0]==1 and ploca[0][1]==1:
            polje=[0,0]
        if ploca[0][1]==1 and ploca[1][2]==1:
            polje=[0,2]
        if ploca[1][2]==1 and ploca[2][1]==1:
            polje=[2][2]
        if ploca[2][1]==1 and ploca[1][0]==1:
            polje=[2][0]
                    
    elif broj_slobodnih==4:
    #treci potez
        pass
    elif broj_slobodnih==2:
    #cetvrti potez
        polje==slobodna[1]

    # Blokada
    for i in range (3):
        for j in range (3):
            if kopija [i][j]==-1:
                kopija[i][j]=1
                if pobjeda (kopija)=="pobjeda x":
                    polje=[i,j]
                kopija[i][j] = -1
                
	#Napad za pobjedu
    for i in range (3):
        for j in range (3):
            if kopija [i][j]==-1:
                kopija [i][j]=0
                if pobjeda (kopija)=="pobjeda o":
                    polje=[i,j]
                kopija[i][j] = -1
    
    return polje         

if __name__=="__main__":
    ploca=[[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
    print(strategija_x (ploca))
    #print(x_polja(ploca))
    #print(o_polja(ploca))

