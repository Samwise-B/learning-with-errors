import numpy as np
import random
import galois
import itertools
import math
import time

def encrypt(pt, key, q):
    A = key[0]
    b = key[1]

    out = []
    for bit in pt:
        # create a random vector
        r = np.random.randint(2, size=A.shape[0])
        a_prime = r.transpose().dot(A) % q
        b_prime = (r.transpose().dot(b) + bit * q/2) % q
        out.append((a_prime, int(b_prime)))

    return np.array(out, dtype=object)

def decrypt(ct, secret_key, q):
    plaintext = []
    for (a_prime, b_prime) in ct:
        v = a_prime.transpose().dot(secret_key)
        m_prime = (b_prime - v) % q
        if m_prime < q/4 or m_prime > q * 0.75:
            pt_bit = 0 
        else:
            pt_bit = 1
        plaintext.append(pt_bit)
    return np.array(plaintext)

def crack1(ct, pk, q):
    GF = galois.GF(q)

    A = GF(pk[0])
    b = GF(pk[1])
    A_T = A.T

    A = A_T.dot(A)
    b = A_T.dot(b)

    # calculate the inverse
    A_inv = np.linalg.inv(A)

    s = (A_inv @ b)

    plaintext = decrypt(ct, s, q)

    return plaintext

def vector_length(vec, q):
    midpoint = math.floor(q/2)
    origin = []
    for i in vec:
        if i >= midpoint:
            origin.append(q)
        else:
            origin.append(0)

    new_vec = np.array(vec - origin)

    return new_vec, np.linalg.norm(new_vec)

#https://stackoverflow.com/questions/6284396/permutations-with-unique-values
class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1

def crack2(ct, pk, q):
    GF = galois.GF(q)

    # get A and B
    A = pk[0]
    b = pk[1]
    
    # create basis of the lattice
    new_row = np.zeros(A.shape[1]+1, dtype=int)
    new_row[-1] = 1
    B = np.c_[A, b]
    B = GF(np.r_[B, [new_row]])

    # get inverse of AT*A
    A = GF(pk[0])
    b = GF(pk[1])
    m = A.shape[0]

    A_inv = np.linalg.inv(A.T @ (A))

    minSecret = None
    minNorm = None
    maxIters = 5

    # generate all possible error vectors
    for i in range(1, maxIters):
        e_gen = [1 for j in range(i)] + [0 for j in range(A.shape[0] - i)]
        e_gens = [e_gen]
        for j in range(i):
            new_gen = e_gens[-1].copy()
            new_gen[j] = -1
            e_gens.append(new_gen)

        for e_gen in e_gens:
            for error in list(perm_unique(e_gen)):
                secret = A_inv @ (A.T @ (b - GF(np.array(error) % q)))
                secret = np.r_[secret, q-1]

                As = np.array(B @ secret)
                latticePoint, norm = vector_length(As, q)

                if (minNorm == None or norm < minNorm):
                    minSecret = secret
                    minNorm = norm


                if np.array_equal(np.abs(latticePoint[:-1]), np.abs(error)):
                    plaintext = decrypt(ct, secret[:-1], q)
                    return plaintext

    plaintext = decrypt(ct, minSecret[:-1], q)

    return plaintext

def crack3(ct, pk, q):
    # create a field (ints mod q)
    GF = galois.GF(q)

    # get A and B
    A = pk[0]
    b = pk[1]
    m = A.shape[0]
    
    # create basis of the lattice
    new_row = np.zeros(A.shape[1]+1, dtype=int)
    new_row[-1] = 1
    B = np.c_[A, b]
    B = np.r_[B, [new_row]]

    points = []
    norms = []
    minNorm = math.inf
    minVec = None
    iteration = 0
    while minNorm > m / 5:        
        # generate pointA, set it to the temp min for the iteration
        pointA = np.random.randint(0, q, size=B.shape[1])
        tmpMinVec = pointA

        # transform pointA into lattice and get its length
        latticePoint = (B @ pointA) % q
        latticePoint, normA = vector_length(latticePoint, q)

        # assign tmpMinNorm
        tmpMinNorm = normA

        # assign min norm and min vec if not already chosen
        # check if generated vector is new min
        if (minNorm == math.inf or tmpMinNorm < minNorm) and tmpMinNorm != 0:
            minNorm = tmpMinNorm
            minVec = tmpMinVec

        # loop over every point so far, try and reduce pointA as much as possible
        for i in range(len(points)):
            pointB = points[i]
            normB = norms[i]

            diff = (pointA - pointB) % q
            diff[-1] = q-1

            latticeDiff = (B @ diff) % q
            latticeDiff, norm = vector_length(latticeDiff, q)

            # check if (new point - some old point) is more reduced
            if norm < tmpMinNorm and norm != 0:
                tmpMinVec = diff
                tmpMinNorm = norm

            # check if new min norm found
            if norm < minNorm and norm != 0:
                minNorm = norm
                minVec = diff

        points.append(tmpMinVec)
        norms.append(tmpMinNorm)
        iteration += 1
    
    plaintext = decrypt(ct, minVec[:-1], q)
    return plaintext


