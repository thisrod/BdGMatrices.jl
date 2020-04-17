using BdGMatrices, LinearAlgebra
using Test

C = 2748.85
Ω = 0.34403008
h = 0.2523
N = 36

A = BdGMatrix(C, Ω, N, h)
BdGMatrices.relax!(A)

# The old school way

eye = Matrix(I,N,N)
H = -kron(eye, A.D2)/2 - kron(A.D2, eye)/2 + diagm(0=>A.V[:])
J = 1im*(repeat(y_axis(A),1,N)[:].*kron(A.D,eye)-repeat(x_axis(A),N,1)[:].*kron(eye,A.D))
Q = diagm(0=>A.ψ[:])
BdGmat = [
    H+2C/h*abs2.(Q)-Ω*J    -C/h*Q.^2;
    C/h*conj.(Q).^2    -H-2C/h*abs2.(Q)-Ω*J
]

uv = Complex.(randn(2*N^2))
uvout = similar(uv)
LinearAlgebra.mul!(uvout, A, uv)

@test uvout ≈ BdGmat*uv