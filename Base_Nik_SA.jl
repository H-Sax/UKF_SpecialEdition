#### Using a CallBack  
using DifferentialEquations,  LinearAlgebra, Distributions, Random, GlobalSensitivity, QuasiMonteCarlo


function Valve(R, deltaP)
    q = 0.0
    if (-deltaP) < 0.0 
        q =  deltaP/R
    else
        q = 0.0
    end
    return q

end

function ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    global tr
    #tᵢ = rem(t + (1 - Eshift) * τ, τ)
    tᵢ = t - tr
    Eₚ = (tᵢ <= τₑₛ) * (1 - cos(tᵢ / τₑₛ * pi)) / 2 +
         (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * (1 + cos((tᵢ - τₑₛ) / (τₑₚ - τₑₛ) * pi)) / 2 +
         (tᵢ <= τₑₚ) * 0

    E = Eₘᵢₙ + (Eₘₐₓ - Eₘᵢₙ) * Eₚ

    return E
end



function DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    global tr
    #tᵢ = rem(t + (1 - Eshift) * τ, τ)
    tᵢ = t - tr
    DEₚ = (tᵢ <= τₑₛ) * pi / τₑₛ * sin(tᵢ / τₑₛ * pi) / 2 +
          (tᵢ > τₑₛ) * (tᵢ <= τₑₚ) * pi / (τₑₚ - τₑₛ) * sin((τₑₛ - tᵢ) / (τₑₚ - τₑₛ) * pi) / 2
    (tᵢ <= τₑₚ) * 0
    DE = (Eₘₐₓ - Eₘᵢₙ) * DEₚ

    return DE
end

#Shi timing parameters
Eshift = 0.0
Eₘᵢₙ = 0.03
τₑₛ = 0.3
τₑₚ = 0.45 
Eₘₐₓ = 1.5
Rmv = 0.06



function NIK!(du, u, p, t)
    pLV, psa, psv, Vlv, Qav, Qmv, Qs = u 
    τₑₛ, τₑₚ, Rmv, Zao, Rs, Csa, Csv, Eₘₐₓ, Eₘᵢₙ = p
    # pressures (more readable names)
# the differential equations
    du[1] = (Qmv - Qav) * ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) + pLV / ShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift) * DShiElastance(t, Eₘᵢₙ, Eₘₐₓ, τ, τₑₛ, τₑₚ, Eshift)
    # 1 Left Ventricle
    du[2] = (Qav - Qs ) / Csa #Systemic arteries     
    du[3] = (Qs - Qmv) / Csv # Venous
    du[4] = Qmv - Qav # volume
    du[5]    = Valve(Zao, (pLV - psa)) - Qav # AV 
    du[6]   = Valve(Rmv, (psv - pLV)) - Qmv # MV
    du[7]     = (du[2] - du[3]) / Rs # Systemic flow
    nothing 
end
##
M = [1.  0  0  0  0  0  0
     0  1.  0  0  0  0  0
     0  0  1.  0  0  0  0
     0  0  0  1.  0  0  0
     0  0  0  0  0  0  0
     0  0  0  0  0  0  0 
     0  0  0  0  0  0  1. ]

Nik_ODE = ODEFunction(NIK!,mass_matrix=M)

u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0]

p = [0.3, 0.45, 0.06, 0.033, 1.11, 1.13, 11.0, 1.5, 0.03]

c = 16 # number of cycles needed plus 1 
function HRV(c)
    t_τL = zeros(c) 
    t_τL[1] = rand(Uniform(0.8,1.1),1)[1]
    for i in 1:c-1
        t_τL[i+1] = t_τL[i] + rand(Uniform(0.8,1.1),1)[1]
    end 
    return t_τL
end 

t_τL = HRV(c)


τ :: Float64 = t_τL[1]
tr = 0.0
tspan = (0, 15)
prob = ODEProblem(Nik_ODE, u0, tspan, p)


function condition(u,t,integrator)
    global τ
    integrator.t - tr > τ
end

n :: Int = 0
# need counter 
function affect!(integrator)
    global n
    global tr
    n = n + 1
    τ_new = t_τL[n+1] - t_τL[n]
    #print(τ_new)
    global τ = τ_new
    tr = t_τL[n]
end


save_positions = (false,false)

cb = DiscreteCallback(condition, affect!, save_positions=save_positions)

@time sol = solve(prob, Rodas5P(autodiff = false), adaptive = false, dt = 0.002, reltol = 1e-8, abstol = 1e-8, callback = cb)

plot(sol, label = ["P_LV" "P_SA" "P_SV" "V_LV" "Q_av" "Q_mv" "Q_s"], tspan = (10*τ, 13*τ))
plot(sol)
plot(sol, idxs = [1,2,3], tspan = (10*τ, 13*τ))
plot(sol, idxs = 4, tspan = (10*τ, 13*τ))

## Begin SA ##
x = range(start = 10 ,stop = 13,step = 0.002)
@time sol = solve(prob, Rodas5P(autodiff = false), adaptive = false, dt = 0.002, reltol = 1e-8, abstol = 1e-8, callback = cb, saveat = x)
τ :: Float64 = t_τL[1]
tr = 0.0
n :: Int = 0

# LV pressure, SA pressure, LV volume 

f1 = function (p)
    prob_func(prob,i,repeat) = remake(prob; u0 = [8.0, 8.0, 8.0, 265.0, 0.0, 0.0, 0.0], p=p[:,i])
    ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
    sol = solve(ensemble_prob,Rodas5P(autodiff = false), adaptive = false, dt = 0.002, reltol = 1e-8, abstol = 1e-8,EnsembleThreads();saveat=x,trajectories=size(p,2),callback= cb)
    out = zeros(4503,size(p,2))
    for i in 1:size(p,2)
      out[1:1501,i] = Array(sol[i][1,:]')
      out[1502:3002,i] = Array(sol[i][2,:]')
      out[3003:4503,i] = Array(sol[i][4,:]')
    end
    out
end
### Sobol GSA on continous outputs ####
N = 12000
lb = [0.21, 0.36, 0.042, 0.0231, 0.777, 0.791, 7.7, 1.05, 0.021]
ub = [0.34, 0.585, 0.078, 0.0429, 1.443, 1.469, 14.3, 1.95, 0.039]

bounds = tuple.(lb,ub)
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(N, lb, ub, sampler)
@time sobol_result_time = gsa(f1,Sobol(),A,B,batch=true)

## Need to multiply the output by the variance of the corresponding measurement
τ :: Float64 = t_τL[1]
tr = 0.0
n :: Int = 0
sol = solve(prob,Rodas5P(autodiff = false), adaptive = false, dt = 0.002, reltol = 1e-8, abstol = 1e-8, saveat=x, callback = cb )

function z(S)
    for i in 1:9
        for j in 1:1501
            if S[j,i] ≤ 0
                S[j,i] = 1e-6
            else 
                S[j,i] = S[j,i]
            end 
        end 
    end 
    return s11
end 

# LV pressure 
s11 = sobol_result_time.S1[1:1501,:]
s1T = sobol_result_time.ST[1:1501,:]
s11 = z(s11)
s1T = z(s1T)
# Systemic Artery pressure
s21 = sobol_result_time.S1[1502:3002,:]
s2T = sobol_result_time.ST[1502:3002,:]
s21 = z(s21)
s2T = z(s2T)
# LV Volume 
s31 = sobol_result_time.S1[3003:4503,:]
s3T = sobol_result_time.ST[3003:4503,:]
s31 = z(s31)
s3T = z(s3T)

using HDF5

h5open("/home/harry/Desktop/PhD/Year 2/Special Edition - Kalman/Nik - Base Case/s3T.h5", "w") do file
    write(file, "A", s3T)  # alternatively, say "@write file A"
end

s21 = h5open("/home/harry/Desktop/PhD/Year 2/Special Edition - Kalman/Nik - Base Case/s21.h5", "r") do file
    read(file, "A")
end



using LaTeXStrings
plot(x,s11,title="First Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,s1T,title="Total Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,log.(s11),title="First Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence", legend = :outertopleft, legendfontsize=15)
plot(x,log.(s1T),title="Total Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")

plot(x,s21,title="First Order Indices - SA.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,s2T,title="Total Order Indices - SA.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,log.(s21),title="First Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,log.(s2T),title="Total Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")

plot(x,s31,title="First Order Indices - LV.V",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,s3T,title="Total Order Indices - LV.V",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,log.(s31),title="First Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")
plot(x,log.(s3T),title="Total Order Indices - LV.p",label = [L"τ_{es}" L"τ_{ep}" L"Rmv" L"Zao" L"Rs" L"Csa" L"Csv" L"E_{max}" L"E_{min}"], xlabel = "Cardiac Cycles", ylabel = "Influence")


plot(x, log.(s11[:,4]), label = L"Zao", xlabel = "Cardiovascular Cycle", yaxis  = "Influence")


using CairoMakie
CairoMakie.activate!(type = "svg")
# Arterial Pressure 
begin
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98))
    #f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98));
    gabc = f[1:3,1]
    ga = gabc[1, 1] = GridLayout()
    gb = gabc[2, 1] = GridLayout()
    gc = gabc[3, 1] = GridLayout()

    gdef = f[1:3, 2] = GridLayout()
    gd = gdef[1, 1] = GridLayout()
    ge = gdef[2, 1] = GridLayout()
    gf = gdef[3, 1] = GridLayout()

    gghi = f[1:3, 3] = GridLayout()
    gg = gghi[1, 1] = GridLayout()
    gh = gghi[2, 1] = GridLayout()
    gi = gghi[3, 1] = GridLayout()

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(ga[1,1], title = L"τ_{es}", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s11[:,1])
    hidexdecorations!(ax, grid = false)

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gb[1,1], title = L"τ_{ep}",xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s11[:,2])
    hidexdecorations!(ax, grid = false)

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gc[1,1], title = L"Rmv", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)", yscale = log10)
    lines!(x, s11[:,3])

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gd[1,1],title = L"Zao", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s11[:,4])
    hidexdecorations!(ax, grid = false)

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(ge[1,1],title = L"Rs", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s11[:,5])
    hidexdecorations!(ax, grid = false)

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gf[1,1], title = L"Csa", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)", yscale = log10)
    lines!(x, s11[:,6])

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gg[1,1], title = L"Csv", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s11[:,7])
    hidexdecorations!(ax, grid = false)

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gh[1,1],title = L"E_{max}", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s11[:,8])
    hidexdecorations!(ax, grid = false)

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gi[1,1], title = L"E_{min}", xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, xlabel = L"t(s)", yscale = log10)
    lines!(x, s11[:,9])
    for (label, layout) in zip(["A", "B", "C", "D", "E", "F", "G", "H", "I"], [ga, gb, gc, gd, ge, gf, gg, gh, gi])
        Label(layout[1, 1, TopRight()], label,fontsize = 18,font = :bold,halign = :right)
    end

    #resize_to_layout!(f)

    f
end   


# All measurements
begin
    f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98))
    #f = Figure(backgroundcolor = RGBf(0.98, 0.98, 0.98));
    gab = f[1:2,1]
    ga = gab[1, 1] = GridLayout()
    gb = gab[2, 1] = GridLayout()

    gcd_ = f[1:2, 2] = GridLayout()
    gc = gcd_[1, 1] = GridLayout()
    gd = gcd_[2, 1] = GridLayout()

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(ga[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10, title = "Ventricular Pressure")
    lines!(x, s11[:,1], label = L"τ_{es}")
    lines!(x, s11[:,2], label = L"τ_{ep}")
    lines!(x, s11[:,3], label = L"Rmv")
    lines!(x, s11[:,4], label = L"Zao")
    lines!(x, s11[:,5], label = L"Rs")
    lines!(x, s11[:,6], label = L"Csa")
    lines!(x, s11[:,7], label = L"Csv")
    lines!(x, s11[:,8], label = L"E_{max}")
    lines!(x, s11[:,9], label = L"E_{min}")
    f[3, 1:2] = Legend(f, ax, orientation = :horizontal)
    #axislegend(f[1:2,0],position = :lb, bgcolor = (:grey90, 0.25));


    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gb[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10, title = "Ventricular Volume")
    lines!(x, s31[:,1])
    lines!(x, s31[:,2])
    lines!(x, s31[:,3])
    lines!(x, s31[:,4])
    lines!(x, s31[:,5])
    lines!(x, s31[:,6])
    lines!(x, s31[:,7])
    lines!(x, s31[:,8])
    lines!(x, s31[:,9])

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gc[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10, title = "Arterial Pressure")
    lines!(x, s21[:,1])
    lines!(x, s21[:,2])
    lines!(x, s21[:,3])
    lines!(x, s21[:,4])
    lines!(x, s21[:,5])
    lines!(x, s21[:,6])
    lines!(x, s21[:,7])
    lines!(x, s21[:,8])
    lines!(x, s21[:,9])

    x = range(start = 10 ,stop = 13,step = 0.002)
    ax = Axis(gd[1,1], xgridstyle = :dash, ygridstyle = :dash,xticksize = 0.5,  yticksize = 5, yscale = log10)
    lines!(x, s21[:,4]; label = L"Zao")
    lines!(x, s21[:,6]; label = L"Csa")
    axislegend(position = :lb, bgcolor = (:grey90, 0.25),  orientation = :horizontal);


    for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
        Label(layout[1, 1, TopRight()], label,fontsize = 18,font = :bold,halign = :right)
    end

    #resize_to_layout!(f)

    f
end    


## Time Averaged 
τ :: Float64 = t_τL[1]
tr = 0.0
n :: Int = 0
sol = solve(prob,Rodas5P(autodiff = false), adaptive = false, dt = 0.002, reltol = 1e-8, abstol = 1e-8, saveat=x, callback = cb )


### LV.P #####
S1_LVP = sobol_result_time.S1[1:1501,:]
ST_LVP = sobol_result_time.ST[1:1501,:]
S11_LVP = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:1501
    S11_LVP[i] = (sum(S1_LVP[1:j,i]*var(sol[1,1:j])))/(sum(var(sol[1,1:j])))/length(x)
    end 
end 

STT_LVP = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:1501
    STT_LVP[i] = (sum(ST_LVP[1:j,i]*var(sol[1,1:j])))/(sum(var(sol[1,1:j])))/length(x)
    end 
end 
sum(S11_LVP)
bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],S11_LVP,title="First Order Indices - LV.P",legend=false)
bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],STT_LVP,title="Total Order Indices - LV.P",legend=false)
bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],log.(S11_LVP .+ 0.006),title="First Order Indices - LV.P",legend=false)
bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],log.(STT_LVP),title="Total Order Indices - LV.P",legend=false)


### SA.p #####
S1_SAP = sobol_result_time.S1[1502:3002,:]
ST_SAP = sobol_result_time.ST[1502:3002,:]
S11_SAP = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:1501
    S11_SAP[i] = (sum(S1_SAP[1:j,i]*var(sol[2,1:j])))/(sum(var(sol[2,1:j])))/length(x)
    end 
end 

STT_SAP = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:1501
    STT_SAP[i] = (sum(ST_SAP[1:j,i]*var(sol[2,1:j])))/(sum(var(sol[2,1:j])))/length(x)
    end 
end 

bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],S11_SAP,title="First Order Indices - SA.p",legend=false)
bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],STT_SAP,title="Total Order Indices - SA.p",legend=false)


### LV.V #####
S1_LVV = sobol_result_time.S1[3003:4503,:]
ST_LVV = sobol_result_time.ST[3003:4503,:]
S11_LVV = Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:length(x)
    S11_LVV[i] = (sum(S1_LVV[1:j,i]*var(sol[4,1:j])))/(sum(var(sol[4,1:j])))/length(x)
    end 
end 

STT_LVV= Vector{Float64}(undef,9)
for i in 1:9
    for j in 1:length(x)
    STT_LVV[i] = (sum(ST_LVV[1:j,i]*var(sol[4,1:j])))/(sum(var(sol[4,1:j])))/length(x)
    end 
end 

bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],S11_LVV,title="First Order Indices - LV.V",legend=false)
bar(["τₑₛ", "τₑₚ", "Rmv", "Zao", "Rs", "Csa", "Csv", "Eₘₐₓ", "Eₘᵢₙ"],STT_LVV,title="Total Order Indices - LV.V",legend=false)

S1 = transpose([S11_LVP S11_SAP S11_LVV])
ST = transpose([STT_LVP STT_SAP STT_LVV])
using CairoMakie
CairoMakie.activate!(type = "svg")
begin
    
f = Figure(resolution = (900, 600),backgroundcolor = RGBf(0.98, 0.98, 0.98));

ax = Axis(f[1,1], xticklabelrotation = π / 3, xticklabelalign = (:right, :center), xticks = (1:3, [L"LV.P", L"SA.P", L"LV.V"]), yticks = (1:9, [L"τ_{es}", L"τ_{ep}", L"Rmv", L"Zao", L"Rs", L"Csa", L"Csv", L"E_{max}", L"E_{min}"]), title = L"Sobol - First~Order", xlabel = L"Measurements", ylabel = L"Parameters")
hm = CairoMakie.heatmap!(ax,S1, colormap=:plasma)
for i in 1:3, j in 1:9
    txtcolor = S1[i, j] < -1000.0 ? :white : :black
    text!(ax, "$(round(S1[i,j], digits = 2))", position = (i, j),
        color = txtcolor, align = (:center, :center), fontsize = 15)
end
CairoMakie.Colorbar(f[1,2],hm);

ax = Axis(f[1,3], xticklabelrotation = π / 3, xticklabelalign = (:right, :center), xticks = (1:3, [L"LV.P", L"SA.P", L"LV.V"]), yticks = (1:9, [L"τ_{es}", L"τ_{ep}", L"Rmv", L"Zao", L"Rs", L"Csa", L"Csv", L"E_{max}", L"E_{min}"]), title = L"Sobol - Total~Order", xlabel = L"Measurements", ylabel = L"Parameters")
hm = CairoMakie.heatmap!(ax,ST, colormap=:plasma)
for i in 1:3, j in 1:9
    txtcolor = ST[i, j] < -1000.0 ? :white : :black
    text!(ax, "$(round(ST[i,j], digits = 2))", position = (i, j),
        color = txtcolor, align = (:center, :center), fontsize = 15)
end
CairoMakie.Colorbar(f[1,4],hm);

f
end