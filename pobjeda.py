#Odredjivanje pobjednika u Krizic-kruzicu
def pobjeda (ploca):
    rezultat="nerjeseno"
    #STUPCI
    for j in range(3):
        brojx=0
        brojo=0
        for i in range(3):
            if ploca[i][j]==1:
                brojx=brojx+1
            elif ploca[i][j]==0:
                brojo=brojo+1
        brojp=3-brojx-brojo
        if (brojp>0) and (rezultat=="nerjeseno"):
            rezultat="igra traje"
        elif brojx==3:
            rezultat="pobjeda x"
        elif brojo==3:
            rezultat="pobjeda o"
       # print("brojx u stupcima je "+str(brojx))
       # print("brojo u stupcima je "+str(brojo))
       # print("brojp u stupcima je "+str(brojp))
#REDCI
    for i in range(3):
        brojx=0
        brojo=0
        for j in range(3):
            if ploca[i][j]==1:
                brojx=brojx+1
            elif ploca[i][j]==0:
                brojo=brojo+1
        brojp=3-brojx-brojo
        if (brojp>0) and (rezultat=="nerjeseno"):
            rezultat="igra traje"
        elif brojx==3:
            rezultat="pobjeda x"
        elif brojo==3:
            rezultat="pobjeda o"
       # print("brojx u redcima je "+str(brojx))
       # print("brojo u redcima je "+str(brojo))
       # print("brojp u redcima je "+str(brojp))
#DIJAGONALE
#1
    brojx=0
    brojo=0
    for i in range (3):
        if ploca[i][i]==1:
            brojx=brojx+1
        elif ploca[i][i]==0:
            brojo=brojo+1
    brojp=3-brojx-brojo
    if (brojp>0) and (rezultat=="nerjeseno"):
            rezultat="igra traje"
    elif brojx==3:
            rezultat="pobjeda x"
    elif brojo==3:
            rezultat="pobjeda o"
   # print("brojx u dijagonali 1 je "+str(brojx))
   # print("brojo u dijagonali 1 je "+str(brojo))
   # print("brojp u dijagonali 1 je "+str(brojp))
#2
    brojx=0
    brojo=0
    for i in range (3):
        if ploca[2-i][i]==1:
            brojx=brojx+1
        elif ploca[2-i][i]==0:
            brojo=brojo+1
    brojp=3-brojx-brojo
    if (brojp>0) and (rezultat=="nerjeseno"):
            rezultat="igra traje"
    elif brojx==3:
            rezultat="pobjeda x"
    elif brojo==3:
            rezultat="pobjeda o"
    # print("brojx u dijagonali 2 je "+str(brojx))
    # print("brojo u dijagonali 2 je "+str(brojo))
    #print("brojp u dijagonali 2 je "+str(brojp))
    return rezultat
