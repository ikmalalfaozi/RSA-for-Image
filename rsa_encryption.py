import numpy as np
from PIL import Image
import random
from sympy import isprime, gcd

def image_to_rgb(filePath):
    with Image.open(filePath) as img:
        return np.array(img.convert("RGB"))

def inspectColor(img_array):
    return np.int64(np.array(img_array[:,:,0])), np.int64(np.array(img_array[:,:,1])), np.int64(np.array(img_array[:,:,2])) 

def save(img_chip,filepath):
    pil_img = Image.fromarray(img_chip.astype(np.uint8), mode = "RGB")
    pil_img.save(filepath)
    return

def generatePrime():
    p = random.randrange(128,1024)
    while (not(isprime(p))):
        p = random.randrange(128,1024)

    q = random.randrange(128,1024)
    while (p == q or not(isprime(q))):
        q = random.randrange(128,1024)

    return p,q

def mod(m,k,n):
    if (k == 1):
        return m % n 
    elif (k % 2) == 1:
        m1 = mod(m**2 % n,k//2,n)
        return (m1*m % n)
    else:
        return mod(m**2 % n, k//2, n)

def encryption(m,e,n):
    m_chip = np.copy(m)
    m_key = np.empty_like(m)
    for i in range(len(m)):
        for j in range(len(m[0])):
            m_chip[i][j] = mod(m_chip[i][j],e,n)
            m_chip[i][j] = m_chip[i][j] // 256
            m_chip[i][j] %= 256
    return m_chip, m_key

def decription(m_chip,m_key,d,n):
    m_plain = np.copy(m_chip)
    for i in range(len(m_chip)):
        for j in range(len(m_chip[0])):
            m_plain[i][j] = mod(m_plain[i][j] + 256*m_key[i][j], d, n)
    return m_plain
    
def RSA(file_path):
# Proses enkripsi
    # Langkah 1
    img = image_to_rgb(file_path)
    # Langkah 2
    r, g, b= inspectColor(img)
    # Langkah 3
    p, q = generatePrime()
    # Langkah 4 dan 5
    n = p*q
    m = (p-1)*(q-1)
    # Langkah 6
    e = random.randrange(1,m)
    while (gcd(e,m) != 1):
        e = random.randrange(1,m)
    # Langkah 7 
    k = 1
    while ((1+m*k) % e != 0):
        k += 1
    d = int((1+m*k)/e)
    # Langkah 8 dan 9
    r, MKey_r = encryption(r,e,n)
    g, MKey_g = encryption(g,e,n)
    b, MKey_b = encryption(b,e,n)
    # Langkah 10
    img_chip = np.dstack([r,g,b])

# Proses dekripsi
    r = decription(r,MKey_r,d,n)
    g = decription(g,MKey_g,d,n)
    b = decription(b,MKey_b,d,n)
    img_plain = np.dstack([r,g,b])

    return img_chip, img_plain

# ALGORITMA UTAMA
filepath = input("Masukkan path relatif gambar: ")
img_chip, img_plain = RSA(filepath)
save(img_chip,"chiper_image.jpg")
save(img_plain,"plain_image.jpg")