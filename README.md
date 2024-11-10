# Learning With Errors (LWE) Cryptosystem Project

This project involved the implementation and analysis of the **Learning With Errors (LWE)** cryptosystem, including the development of encryption and decryption functions, followed by the design and implementation of various attacks on the system. The goal was to explore both the construction and potential vulnerabilities of cryptographic schemes based on the LWE problem, a key component of modern cryptography known for its resistance to quantum computing attacks.

## Project Overview

The project was divided into four main tasks:
1. **Implementing the Learning With Errors Cryptosystem**: Design and implement the core encryption and decryption functions.
2. **Attacking Learning Without Errors**: Develop an attack against the system when the error distribution is always 0.
3. **Attacking Learning With A Few Errors**: Create an attack exploiting the scenario where errors are rare but non-zero.
4. **Attacking Learning With Errors**: Design an attack for the general LWE system, where the error distribution is more complex.

Each task required a deep understanding of the cryptographic scheme and hands-on experience with **NumPy** for efficient computation, while also exploring cryptanalysis techniques to break the system's security.

## Key Tasks and Technical Skills Developed

### 1. Implementing the LWE Cryptosystem

#### Key Skills:
- **Cryptography**: Implemented both **encryption** and **decryption** for the LWE cryptosystem, focusing on operations like matrix-vector multiplication, modular arithmetic, and random number generation.
- **Modular Arithmetic**: Applied modular arithmetic techniques (mod q) to ensure correct encryption and decryption, an essential skill for working with cryptographic algorithms.
- **NumPy for Efficient Computation**: Utilized **NumPy** to handle large matrix operations and vector arithmetic, optimizing performance for cryptographic calculations.

#### Deliverables:
- **`encrypt(plaintext, public_key, q)`**: A function that encrypts the plaintext using the public key and returns the ciphertext, using randomization to ensure security.
- **`decrypt(ciphertext, private_key, q)`**: A function that decrypts the ciphertext using the private key and returns the original plaintext.

### 2. Attacking ‘Learning Without Errors’ (LWE with χ = 0)

#### Key Skills:
- **Cryptanalysis**: Developed an attack on the system when there are no errors (χ = 0), which exploits the structure of the encryption mechanism to retrieve the plaintext directly from the ciphertext.
- **Mathematical Reasoning**: Applied mathematical reasoning to identify weaknesses in the LWE scheme when the error distribution is set to zero, learning how even small changes to the error distribution can drastically affect security.
  
#### Deliverables:
- **`crack1(ciphertext, public_key, q)`**: A function that decodes the ciphertext using only the public key (no private key), effectively "breaking" the system when errors are absent.

### 3. Attacking ‘Learning With A Few Errors’ (LWE with rare errors)

#### Key Skills:
- **Probabilistic Cryptanalysis**: Designed an attack that takes advantage of the low-probability errors in the system (χ is almost always 0), requiring knowledge of probabilistic modeling and error correction.
- **Error Distribution Understanding**: Gained an understanding of how small but non-zero errors can be exploited in cryptographic systems to launch successful attacks, despite the error distribution being designed to be sparse.

#### Deliverables:
- **`crack2(ciphertext, public_key, q)`**: A function that exploits the sparse error distribution to recover the plaintext, leveraging the rare errors in the system.

### 4. Attacking ‘Learning With Errors’ (Full LWE Attack)

#### Key Skills:
- **Advanced Cryptanalysis**: Applied advanced techniques to launch an attack on the full LWE system, where the error distribution is arbitrary, exploring how LWE can be broken even with a complex error structure.
- **Security Analysis**: Evaluated the security of the cryptosystem by designing and implementing a robust attack capable of overcoming the cryptographic protections afforded by the LWE problem.

#### Deliverables:
- **`crack3(ciphertext, public_key, q)`**: A function that attacks the standard LWE system, recovering the plaintext despite the presence of errors.

## Technical Skills Acquired

### Cryptographic Algorithms
- **Public Key Encryption**: Implemented core cryptographic operations such as key generation, encryption, and decryption in a cryptosystem based on the **Learning With Errors** problem.
- **Quantum-Secure Cryptography**: Gained an understanding of how LWE is resistant to quantum attacks, a fundamental concept for modern cryptographic systems.
- **Error Distribution Handling**: Worked with different error distributions (χ = 0, rare errors, and standard LWE errors) and learned how these distributions impact the security and effectiveness of the cryptosystem.

### Cryptanalysis and Security
- **Attacking Public-Key Cryptosystems**: Gained practical experience in breaking cryptographic systems, a key skill for security researchers. Designed and implemented cryptanalytic attacks on systems with various error distributions.
- **Mathematical and Computational Cryptanalysis**: Developed the ability to apply mathematical reasoning, error correction, and probabilistic models to break cryptosystems, essential for a career in cryptography and cybersecurity.

### Software Engineering and Optimization
- **NumPy for Cryptographic Computation**: Used **NumPy** for efficient vector and matrix manipulation, crucial for implementing large-scale cryptographic operations and speeding up computational tasks in cryptography.
- **Code Optimization**: Optimized the cryptographic operations and cryptanalysis attacks to handle large matrices and efficiently decode ciphertexts.

### Problem-Solving and Algorithm Design
- **Complex Algorithm Design**: Developed algorithms to solve both encryption and cryptanalysis problems, testing and refining the solutions through multiple stages of the project.
- **Security Protocol Design**: Learned to identify and exploit vulnerabilities in cryptographic protocols, improving my ability to design secure systems and assess the security of existing protocols.

## Summary

This project provided a comprehensive exploration of the **Learning With Errors** cryptosystem, offering valuable insights into both the theoretical foundations and practical implementation of cryptographic algorithms. The experience developed my skills in:
- **Cryptography and Public Key Systems**
- **Cryptanalysis Techniques**
- **Error Distribution in Cryptosystems**
- **NumPy for High-Performance Cryptographic Computation**
- **Security Evaluation and Vulnerability Identification**

These skills are critical for roles in cybersecurity, cryptographic research, and data security, and have equipped me with the practical expertise needed to tackle modern cryptographic challenges, including those that are resistant to quantum attacks.
