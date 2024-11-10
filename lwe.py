import numpy as np
from sympy import Matrix, solve, ZZ
import random
import galois
import itertools
import math
import time

q = 101
m = 32
n = 24

error = np.random.normal(size=m, scale=0.5).astype(int)

def key_gen():
    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, q, size=n)
    # sample e from gaussian distribution?
    #e = np.random.normal(size=m, scale=0.5).astype(int)
    e = error
    #e = 0
    print("ERROR:", e)
    print(e.shape)

    print("b with no errors:", A.dot(s) % q)
    b = (A.dot(s) + e) % q
    print("b shape:", b.shape)

    return {
        "secret": s,
        "public": (A, b)
    }

def encrypt(pt, key, q):
    A = key[0]
    b = key[1]
    #print("public key:", key)
    #print("A shape:", A.shape)
    #print("b shape:", b.shape)
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
    #solution = np.linalg.lstsq(pk[0], pk[1])[0]
    A = GF(pk[0])
    b = GF(pk[1])
    A_T = A.T
    print(A.shape, b.shape)
    A = A_T.dot(A)
    b = A_T.dot(b)
    print(A.shape, b.shape)
    
    # calculate the inverse
    A_inv = np.linalg.inv(A)
    print(A_inv)

    s = (A_inv @ b)
    print(s)

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

#def crack2(ct, pk, q):
    numPoints = 10000

    # create a field (ints mod q)
    GF = galois.GF(q)

    # get A and B
    A = pk[0]
    b = pk[1]
    print("old shape", A.shape)
    
    # create basis of the lattice
    new_row = np.zeros(A.shape[1]+1, dtype=int)
    new_row[-1] = 1
    B = np.c_[A, b]
    B = np.r_[B, [new_row]]
    print("new shape", B.shape)

    point = np.zeros(B.shape[1], dtype=int)
    point[-1] = 1

    point2 = point
    point2[0] = 1

    print(B.dot(point) % q)
    print(B.dot(point2) % q)

    points = []
    latticePoints = []
    norms = []
    minNorm = None
    minVec = None
    secret = None
    # generate initial shapes
    for i in range(numPoints):
        # generate a point and add to the list of points, ending in q-1
        point = np.random.randint(0, q, size=B.shape[1])
        point[-1] = q-1
        points.append(point)

        # transform it into a lattice point
        latticePoint = B.dot(point) % q
        latticePoints.append(latticePoint)

        # calculate the norm relative to the closest origin and get the respective vector
        latticePoint, norm = vector_length(latticePoint, q)
        norms.append(norm)

        # update min norm and minVector
        if (minNorm == None or norm < minNorm) and norm != 0:
            minNorm = norm
            minVec = latticePoint
            secret = point
            print(minVec, norm, secret)

    print("taking differences")
    # iteratively calculate some new set of points by taking differences
    for j in range(1000):
        print("iteration:", j)
        new_points = []
        for (vecA, vecB) in itertools.combinations(random.sample(points, math.ceil(numPoints / q)), 2):
            latticePointA = B.dot(vecA) % q
            latticePointB = B.dot(vecB) % q

            _, normA = vector_length(latticePointA, q)
            _, normB = vector_length(latticePointB, q)

            diff = (vecA - vecB) % q
            diff[-1] = q-1

            # transform it into a lattice point
            latticePoint = B.dot(diff) % q

            # calculate the norm relative to the closest origin and get the respective vector
            latticePoint, norm = vector_length(latticePoint, q)

            if norm < normA and norm < normB:
                new_points.append(diff)

            # update norm
            if norm < minNorm and norm != 0:
                minNorm = norm
                minVec = latticePoint
                secret = point
                print(minVec, norm, secret)
        points = points + new_points

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
    # print("old shape", A.shape)
    
    # create basis of the lattice
    new_row = np.zeros(A.shape[1]+1, dtype=int)
    new_row[-1] = 1
    B = np.c_[A, b]
    B = GF(np.r_[B, [new_row]])
    # print("new shape", B.shape)

    # get inverse of AT*A
    A = GF(pk[0])
    b = GF(pk[1])
    m = A.shape[0]
    # print(A.shape, b.shape, np.linalg.matrix_rank(A))
    # print("A transpose shape: ", np.linalg.matrix_rank(A.T), A.T @ (A))
    A_inv = np.linalg.inv(A.T @ (A))

    options = list(itertools.combinations([1, -1], 2))
    minSecret = None
    minNorm = None
    maxIters = 5
    #print(maxIters)
    # generate all possible error vectors
    for i in range(1, maxIters):
        print("iteration:", i)
        start = time.time()
        e_gen = [1 for j in range(i)] + [0 for j in range(A.shape[0] - i)]
        e_gens = [e_gen]
        for j in range(i):
            new_gen = e_gens[-1].copy()
            new_gen[j] = -1
            e_gens.append(new_gen)

        print(e_gens)
        for e_gen in e_gens:
            for error in list(perm_unique(e_gen)):
                #print(error)
                secret = A_inv @ (A.T @ (b - GF(np.array(error) % q)))
                secret = np.r_[secret, q-1]

                As = np.array(B @ secret)
                latticePoint, norm = vector_length(As, q)

                if (minNorm == None or norm < minNorm):
                    minSecret = secret
                    minNorm = norm
                    #print("new min:", minNorm, minSecret, latticePoint)

                if np.array_equal(np.abs(latticePoint[:-1]), np.abs(error)):
                    print(latticePoint, error, secret[:-1])
                    #plaintext = decrypt(ct, secret[:-1], q)
                    #print(plaintext)
                    #return plaintext
                    
            #print("iteration time:", time.time() - start)

    print(minNorm, np.array(minSecret))

    plaintext = decrypt(ct, minSecret[:-1], q)
    print(plaintext)

    return plaintext

def crack3(ct, pk, q):
    # create a field (ints mod q)
    GF = galois.GF(q)

    # get A and B
    A = pk[0]
    b = pk[1]
    #print("old shape", A.shape)

    # indexes = random.sample([i for i in range(m)], n)
    # newA = []
    # new_b = []
    # for index in indexes:
    #     newA.append(A[index])
    #     new_b.append(b[index])

    # A = np.array(newA)
    # b = np.array(new_b) + 1
    
    # create basis of the lattice
    new_row = np.zeros(A.shape[1]+1, dtype=int)
    new_row[-1] = 1
    B = np.c_[A, b]
    B = np.r_[B, [new_row]]
    #print("new shape", B.shape)

    start = time.time()

    points = []
    norms = []
    minNorm = math.inf
    minVec = None
    iteration = 0
    while minNorm > m / 5:
        # if iteration % 100 == 0:
        #     print("iteration:", iteration)
        # if iteration % 1000 == 0:
        #     print("time taken:", time.time() - start)
        # if iteration % 1500 == 0 and iteration > 0:
        #     slice = int(1500 * iteration / 1500) - 2
        #     points = points[slice:]
        #     norms = norms[slice:]

        
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
            #print("new point, new min:", minNorm, minVec)

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
                #print("new min:", minNorm, minVec, latticeDiff)

        points.append(tmpMinVec)
        norms.append(tmpMinNorm)
        iteration += 1
    
    plaintext = decrypt(ct, minVec[:-1], q)
    print("total time: ",time.time() - start)
    print("plaintext:", plaintext)
    return plaintext

if __name__ == "__main__":
    keys = key_gen()
    secret = keys['secret']
    public = keys['public']
    print("SECRET KEY:", secret, secret.shape)
    #print(public)

    pt_0 = np.array([0,1,1,0,1,0,0,1,1,0,1,1,0,0,1,0,1,1,1,0])
    ct = encrypt(pt_0, public, q)
    #print("ciphertext", ct)

    pt_1 = decrypt(ct, secret, q)
    #print("plaintext: ", pt_1)

    count = 0
    isEqual = True
    for i in range(len(pt_0)):
        if (pt_0[i] != pt_1[i]):
            count += 1
    
    #print(pt_0)
    #print(pt_1)
    #print("number of incorrect bits: ", count)

    #print("#### crack 1 ##########")
    #crack1(ct, public, q)

    print("#### crack 2 ####")
    crack2(ct, public, q)

    # print("#### crack 3 ####")
    # crack3(ct, public, q)

