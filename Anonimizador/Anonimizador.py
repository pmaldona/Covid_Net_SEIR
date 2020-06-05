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
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
 

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
    greeting.pack()

    window.mainloop()

     
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



 
 
class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Anonimizador de Ruts")
        self.minsize(640, 400)
        #self.wm_iconbitmap('icon.ico')
 
        self.labelFrame = ttk.LabelFrame(self, text = "Abrir archivo")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
 
        self.button()
 
 
 
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "Browse A File",command = self.fileDialog)
        self.button.grid(column = 1, row = 1)
 
 
    def fileDialog(self):
 
        self.filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =(("csv files","*.csv"),("all files","*.*")) )
        self.label = ttk.Label(self.labelFrame, text = "")
        self.label.grid(column = 1, row = 2)
        self.label.configure(text = self.filename)