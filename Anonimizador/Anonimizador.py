#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Samuel Ropert
Dlab

"""
import pandas as pd
import hashlib
import argparse
import tkinter as tk  

# ------------------------- #
#     Anonimizador Data     #
# ------------------------- #
# Metodo Hashing and salting
# Se asume que el archivo de datos está en excel
# Usar rut y 'salt' para crear un hash unico mas dificil de decifrar
# 



def anonimizadorrut(rut,salt):    
    anonimo = hashlib.sha3_256(bytes(rut.encode('utf-8')))    
    anonimo.update(salt)
    return(anonimo.hexdigest())


def replacedata(data,campo,salt):
    id = []
    for i in range(data.shape[0]):
        aux = str(data[campo][i])
        id.append(anonimizadorrut(aux,salt))
    del data[campo]
    data['id'] = id
    return(data)



if __name__ == "__main__":
    # ------------- #
    #      GUI      # 
    # ------------- #

    window = tk.Tk() 
    window.title("Anonimizador Rut - COVID19")
    greeting = tk.Label(text="Hello, Tkinter")

     
    print("Anonimizador")
    #parser = argparse.ArgumentParser(description='Anonimizador COVID19')
    #parser.add_argument('-i', "--input", dest="input", required = True, help="Archivo origen de datos")
    #parser.add_argument('-s', "--salt", dest="salt", required = False, help="Codigo para mejorar seguridad")

    #args=parser.parse_args()

    datafile = input("Ingresar ubicación y nombre del archi")


    if args.input:
        path = args.input
    else:
        raise IOError("Agregar el path del archivo de datos")

    if args.salt:  
        salt = bytes(args.salt.encode('utf-8'))        
    else:
        salt = b"_covid19"

    # ------------------------- #
    #         Read data         #
    # ------------------------- #
    # Reconocer extension (pendiente)     
    data = pd.read_excel(path)
    
    campo = 'rut' 
    keys = data.keys()


    # ------------------------- #
    #       Replace data        #
    # ------------------------- #

    data = replacedata(data,campo,salt)

    # ------------------------- #
    #        Save  data         #
    # ------------------------- #
    data.to_excel('anonimizado_'+path)


    