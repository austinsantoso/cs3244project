def fakeZeros(a, b, c, d):
    output = []
    for i in range(a):
        ilayer = []
        for j in range(b):
            jlayer = []
            for k in range(c):
                klayer = []
                for l in range(d):
                    klayer.append(0.)
                jlayer.append(klayer)
            ilayer.append(jlayer)
        output.append(ilayer)
    return output

print(fakeZeros(2,1,3,4))