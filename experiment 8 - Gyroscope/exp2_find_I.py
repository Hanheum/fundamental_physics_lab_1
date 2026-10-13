m = 0.05 #kg
g = 9.808
d = 0.2 #m

def I(omega, w_pulley):
    return m*g*d/(omega * w_pulley*0.8)

print(I(0.134, 68.97))
