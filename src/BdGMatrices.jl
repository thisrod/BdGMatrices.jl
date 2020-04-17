module BdGMatrices

using BandedMatrices, Optim, LinearAlgebra

export BdGMatrix, x_axis, y_axis, z_plane, zplot, argplot

struct BdGMatrix <: AbstractMatrix{Complex{Float64}}
    C::Float64
    Ω::Float64
    N::Int
    h::Float64
    V::Matrix{Float64}
    ψ::Matrix{Complex{Float64}}
    nnc::Matrix{Float64}
    D::BandedMatrix{Float64}
    D2::BandedMatrix{Float64}
end

# Array interface

Base.size(A::BdGMatrix) = (2*A.N^2, 2*A.N^2)

# coordinates

y_axis(N,h) = h/2*(1-N:2:N-1)
x_axis(N,h) = y_axis(N,h)'
z_plane(N,h) = Complex.(x_axis(N,h), y_axis(N,h))

for f in (:y_axis, :x_axis, :z_plane)
    @eval $f(A::BdGMatrix) = $f(A.N, A.h)
end

BdGMatrix(C, Ω, N, h) =
    BdGMatrix(C, Ω, N, h,
        abs2.(z_plane(N,h)),					# V
        Matrix{Complex{Float64}}(undef,N,N),	# ψ
        Matrix{Float64}(undef,N,N),			# nnc
        (1/h).*op(N, st1),					# D
        (1/h^2).*op(N, st2)					# D2
    )
    
# Finite difference matrices for 1st and 2nd derivative

st1 = [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60]
st2 = [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]

function op(N,stencil)
    mid = (length(stencil)+1)÷2
    diags = [i-mid=>fill(stencil[i],N-abs(i-mid)) for i = keys(stencil)]
    BandedMatrix(Tuple(diags), (N,N))
end

# GPE energy and nonlinear Hamiltonian

Lop(A::BdGMatrix) = ψ ->
    -(A.D2*ψ+ψ*A.D2')/2 + A.V.*ψ+A.C/A.h*abs2.(ψ).*ψ - 
        1im*A.Ω*(y_axis(A).*(ψ*A.D')-x_axis(A).*(A.D*ψ))
Ham(A::BdGMatrix) = ψ ->
    -(A.D2*ψ+ψ*A.D2')/2+A.V.*ψ + A.C/2A.h*abs2.(ψ).*ψ -
        1im*A.Ω*(y_axis(A).*(ψ*A.D')-x_axis(A).*(A.D*ψ))

"""
Find the ground state order parameter and adjust V to chemical potential

TODO two argument relax! with initial condition
"""
function relax!(A::BdGMatrix)
    L = Lop(A)
    H = Ham(A)
    gs = size(z_plane(A))
    togrid(xy) = reshape(xy, gs)
    E(xy) = sum(conj.(togrid(xy)).*H(togrid(xy))) |> real
    grdt!(buf,xy) = copyto!(buf, 2*L(togrid(xy))[:])
    init = Complex.(exp.(-A.V/2)/√π)
    result = optimize(E, grdt!, init[:],
        GradientDescent(manifold=Sphere()),
        Optim.Options(iterations = 400_000, allow_f_increases=true)
    )
    # TODO check convergence
    A.ψ .= togrid(result.minimizer)
    A.V .-= sum(conj.(A.ψ).*L(A.ψ)) |> real
    nothing
end

# Allocation free application
function LinearAlgebra.mul!(uvout::AbstractVector, A::BdGMatrix, uv::AbstractVector)
    C, Ω, V, N, h, D, D2 = A.C, A.Ω, A.V, A.N, A.h, A.D, A.D2
    x, y = x_axis(A), y_axis(A)
    u = reshape(view(uv,1:N^2), N, N) 
    v = reshape(view(uv,A.N^2+1:2*N^2), N, N)
    Au = reshape(view(uvout,1:N^2), N, N)
    Au .= -(D2*u+u*D2')/2 + V.*u + 2C/h*abs2.(A.ψ).*u - 
        1im*Ω*(y.*(u*D')-x.*(D*u)) -
        C/h*A.ψ.^2 .* v
    Av = reshape(view(uvout,A.N^2+1:2*N^2), N, N)
    Av .= (D2*v+v*D2')/2 - V.*v - 2C/h*abs2.(A.ψ).*v - 
        1im*Ω*(y.*(v*D')-x.*(D*v)) +
        C/h*conj.(A.ψ).^2 .* u
    uvout
end

# Axes for plotting

paxes(A::BdGMatrix) = (x_axis(A)[:], y_axis(A))

end # module
