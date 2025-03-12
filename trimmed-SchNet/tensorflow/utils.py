import numpy as np
import sys

def Loadgen(path,conversion):
    try:
        file = open(path, "r+")
    except OSError:
        print ("Could not open gen file "+ path)
        sys.exit()
    lines = file.readlines()

    aux = lines[0].split(' ')
    aux = list(filter(lambda x: x != '', aux))
    Nat = int(aux[0])

    types = lines[1].split(' ')
    types = list(filter(lambda x: x != '', types))
    types[-1] = types[-1].strip("\n")

    periodic = False
    if len(aux) > 0:
        if aux[1] == 'S\n':
            periodic = True

    lines = lines[2:]

    aux_types = []

    geometry = []
    for i in range(Nat):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        aux_types.append(a[1])
        a = list(map(float, a[2:]))
        geometry.append(a)
    geometry = np.array(geometry)/conversion

    lines = lines[Nat:]

    for i in range(len(aux_types)):
        if aux_types[i] == '1':
            aux_types[i] = 0
        if aux_types[i] == '2':
            aux_types[i] = 1
        if aux_types[i] == '3':
            aux_types[i] = 2

    unit_cell = []
    for i in range(len(lines)):
        a = lines[i].split(' ')
        a = list(filter(lambda x: x != '', a))
        a = list(map(float, a))
        unit_cell.append(a)
    arr_t = np.array(unit_cell)/conversion
    unit_cell = arr_t.tolist()

    return Nat, geometry, periodic, aux_types, np.array(unit_cell)
