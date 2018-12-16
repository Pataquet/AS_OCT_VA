class Positions(object):
    colum = -1
    inicio = -1
    fin = -1
    distanciaSuav =  -1
    distancia = -1

    # The class "constructor" - It's actually an initializer
    def __init__(self, position, inicio, fin):
        self.colum = position
        self.inicio = inicio
        self.fin = fin
        self.distancia = fin - inicio