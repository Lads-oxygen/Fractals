using CUDA
# using BenchmarkTools
# using Test
using Adapt
using Serialization
using Plots
using Polynomials
# using Roots

N = 8192*2
const MAXITER = 50
const TOL = 1e-20
const origin = ComplexF32(-1, 1)
width = abs2(origin)
const Δ = width / (N - 1)

poly = Vector{ComplexF32}([-1, 0, 0, 1])
p = Polynomial(poly)


function eval(f, x)
    f_x = last(f)
    for i = length(f)-1:-1:1
        @inbounds f_x = f_x * x + f[i]
    end
    return f_x
end


function secant!(f, cfx)

    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    zm = complex(0)
    zn = origin + complex(x * Δ, -y * Δ)
    f_zm = f[1]
    f_zn = eval(f, zn)
    # @cuprint("grid ", gridDim().z)
    # @cuprintln(x, " ", y, " fzn ", zn.re, " ", zn.im)

    for i = 1:MAXITER
    
        if abs(f_zn - f_zm) < TOL
            # @cuprintln("division by 0")
            cfx[x, y] = -999
            return
        end
    
        temp = zn - f_zn * (zn - zm) / (f_zn - f_zm)
        zm = zn
        zn = temp
        f_zm = f_zn
        f_zn = eval(f, zn)
    
        if abs(zn - zm) < TOL && abs(f_zn) < TOL
            # @cuprintln("root is", zn.re, " ", zn.im)
            cfx[x, y] = zn
            return
        end
    end
    cfx[x, y] = -999
    # @cuprintln("did not converge")
    nothing
end

f_x = Matrix{ComplexF32}(undef, N, N)

cpoly = cu(poly)
cfx = cu(f_x)


function run!(cpoly, cfx)
    # kernel = @cuda launch = false secant!(cpoly, cfx)
    threads = min(N, 8)
    blocks = cld(N, threads)
    CUDA.@sync begin
        @cuda threads = (threads, threads) blocks = (blocks, blocks) secant!(cpoly, cfx)
    end
    println(threads, " ", blocks)
    copyto!(f_x, cfx)
    return nothing
end


function main()
    run!(cpoly, cfx)
    # serialize("data", arr)
    # fractal = gen_frac(f_x)
    # make_fig(fractal)
    return nothing
end


function gen_frac(f_x)
    roots = Polynomials.roots(p)
    fractal = Matrix{Int8}(undef, N, N)
    # display(f_x)
    # println("\n")
    # display(roots)
    # println("\n")
    for i in 1:N, j in 1:N
        # a = filter(x -> abs(x-f_x[i, j]) < TOL, roots)
        index = findall(x -> abs(f_x[i, j] - roots[x]) < 1, eachindex(roots))
        try
            fractal[i, j] = index[1]
            # println("No error, index is: ", index[1], " ", fractal[i, j])
        catch e
            # println("Error here: ", e)
            fractal[i, j] = 0
        end
        # print(" ",fractal[i, j])
    end
    fractal
end



function make_fig(fractal)
    # fractal = deserialize("data")
    gr()
    # display(fractal)
    img = heatmap(1:N, 1:N, fractal, legend=:none, axis=nothing)
    savefig(img, "./fractal.png")

    # histogram2d(f_x, nbins = (40, 40), show_empty_bins = true, normed = true, aspect_ratio = 1)
end

main()
# make_fig()


# run!(a, f)
# @time run!(a, f)  

# print(a, b)






# bench_secant!(b)  # run it once to force compilation
# CUDA.@time bench_secant!(b)
# println(b)



