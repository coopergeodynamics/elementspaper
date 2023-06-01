#!/usr/bin/env python
# coding: utf-8

# Craton with Lateral Variations
# ----------
# 
# Let's start simple and put a craton with a constant viscosity into the temperature field of a well-developed isoviscous convecting mantle.  Note, be sure that the temperature & mesh files that you load correspond to the same Ra, mesh size, and resolution as what you want to use for the craton models.

# In[36]:


import underworld as uw
from underworld import function as fn
import underworld.visualisation as vis
import math
import time as timekeeper
import numpy

import matplotlib.pyplot as plt
plt.ion()
from IPython import display


rank = uw.mpi.rank 


# Set up parameters of model space
# ------
# 

# In[37]:


# Set simulation box size.
boxHeight = 1.0
boxLength = 3.0
# Set the resolution.
res = 128   # make sure this resolution matches what you eventually will use for the other model.  
            # Otherwise you'll have to play some extrapolation tricks 
# Set min/max temperatures.
tempMin = 0.0
tempMax = 1.0


# Set up mesh
# -----------

# In[38]:


mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"), 
                                 elementRes  = (int(boxLength*res), res), 
                                 minCoord    = (0., 0.), 
                                 maxCoord    = (boxLength, boxHeight),
                                 periodic     = [True, False])

velocityField       = mesh.add_variable(         nodeDofCount=2 )
pressureField       = mesh.subMesh.add_variable( nodeDofCount=1 )
temperatureField    = mesh.add_variable(         nodeDofCount=1 )
temperatureDotField = mesh.add_variable(         nodeDofCount=1 )

# Initialise values
velocityField.data[:]       = [0.,0.]
pressureField.data[:]       = 0.


# Let's Load Data from other model
# -------
# 

# In[39]:


# Read temperature data
readTemperature = True
# Read swarm data
loadData = True

#determining the last step ran

dir_output = '/home/cmcooper/InitialConditions128_1e7PBC/'
if not loadData:
    step = 0
    time = 0.0
    rStep = -1.0
else:
    dataload = numpy.loadtxt(dir_output + 'FrequentOutput.dat', skiprows=4)
    nL = dataload[-1,0]
    nL = int(-1-(nL % 1000))
    step = int(dataload[nL,0])
    time = dataload[nL,1] 
    rStep = step

initstep=step    
    
#print('Starting at step %i and time %.2E' %(step,time))


if readTemperature:
    temperatureField.load(dir_output + 'temperature_%i.h5' %step, interpolate=True)

else:  #this will set up a sinusoidal temp field
    pertStrength = 0.2
    deltaTemp = tempMax - tempMin
    for index, coord in enumerate(mesh.data):
        pertCoeff = math.cos( math.pi * coord[0] ) * math.sin( math.pi * coord[1] )
        temperatureField.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
        temperatureField.data[index] = max(tempMin, min(tempMax, temperatureField.data[index]))


# Boundary & Initial Conditions
# ------

# Set top and bottom wall temperature boundary values.

# In[40]:


for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = tempMax
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = tempMin


# In[41]:


iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]

BottomWall = mesh.specialSets["MinJ_VertexSet"] 
TopWall = mesh.specialSets["MaxJ_VertexSet"] 
LeftWall = mesh.specialSets["MinI_VertexSet"] 
RightWall = mesh.specialSets["MaxI_VertexSet"] 


# Construct sets for ``I`` (vertical) and ``J`` (horizontal) walls.

# Create Direchlet, or fixed value, boundary conditions. More information on setting boundary conditions can be found in the **Systems** section of the user guide.

# In[42]:


# 2D velocity vector can have two Dirichlet conditions on each vertex, 
# v_x is fixed on the iWalls (vertical), v_y is fixed on the jWalls (horizontal) - freeslip on all sides

# make sure these match the boundary conditions you'll eventually use for the full model

velBC  = uw.conditions.DirichletCondition( variable        = velocityField, 
                                           indexSetsPerDof = (None, jWalls) )

# Temperature is held constant on the jWalls
tempBC = uw.conditions.DirichletCondition( variable        = temperatureField, 
                                           indexSetsPerDof = (jWalls,) )


# In[43]:


figtemp = vis.Figure( figsize=(800,400) )
figtemp.append( vis.objects.Surface(mesh, temperatureField, colours="gray") )
#figtemp.append( vis.objects.Mesh(mesh) )
#figtemp.show()

temperatureFieldIntegral = uw.utils.Integral(fn = temperatureField,mesh= mesh,integrationType="volume")
volume_integral = uw.utils.Integral( mesh=mesh, fn=1., integrationType="volume" )
volume = volume_integral.evaluate()
avTemperature = temperatureFieldIntegral.evaluate()[0]/volume[0]
#print (avTemperature)


# Set up material parameters and functions
# ----------
# 
# Set functions for viscosity, density and buoyancy force. These functions and variables only need to be defined at the beginning of the simulation, not each timestep.

# Starting with Materials and Swarm
# ------

# In[66]:


swarm         = uw.swarm.Swarm( mesh=mesh )
swarmLayout   = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
swarm.populate_using_layout( layout=swarmLayout )

nParticles = 20

# particle population control object (has to be called)
population_control = uw.swarm.PopulationControl(swarm,
                                                aggressive=False,splitThreshold=0.15, maxDeletions=2,maxSplits=5,
                                                particlesPerCell=nParticles)


# Set up material swarms
materialIndex  = swarm.add_variable( dataType="int",    count=1 )

# all the potential materials.  for the reference case, let's just have average crustal structure
continentalMaterial = 0
weakMaterial = 1
weakMaterial2 = 2
refmantleMaterial = 3

continentalDepth = 0.7
contX1 =  0.00 
contX2 =  0.00

weakzoneDepth = 0.7
weakzoneX1 = 1.65
weakzoneX2 = 1.75

#give shapes a material

diag = True #toggle between trapezoidal vs straight root shape
weak = True 
    
materialIndex.data[:] = refmantleMaterial
for index,coord in enumerate(swarm.particleCoordinates.data):
    if diag:
        
        if weak: 
            if coord[1] >= -2.8*coord[0]+3.8 and coord[1] >= 2.8* coord[0]-4.6 and coord[1]>=continentalDepth: 
                materialIndex.data[index] = weakMaterial
            if coord[0] > 1.1 and coord[0] < 1.9 and coord[1] > continentalDepth:
                materialIndex.data[index] = continentalMaterial
            if coord[0] > 1.65 and coord[0] < 1.75 and coord[1] > continentalDepth:
                materialIndex.data[index] = weakMaterial2
                        
        else:
            if coord[1] >= -2.8*coord[0]+3.8 and coord[1] >= 2.8* coord[0]-4.6 and coord[1]>=continentalDepth: 
                materialIndex.data[index] = continentalMaterial
            
            
    else:
        if coord[1] > continentalDepth and coord[0] > contX1 and coord[0] < contX2:
            materialIndex.data[index] = continentalMaterial
        
        if weak:    
            if coord[1] > weakzoneDepth and coord[0] > weakzoneX1 and coord[0] < weakzoneX2:
                materialIndex.data[index] = weakMaterial  

materialPoints = vis.objects.Points(swarm, materialIndex, pointSize=3.,  colours='#009E73 #E69F00 #F0E442 grey')
materialPoints.colourBar.properties = {"ticks" : 2, "margin" : 40, "align" : "center"}

figMaterialMesh = vis.Figure(figsize=(800,400),title="Materials and Mesh", quality=3)
#figMaterialMesh.append( vis.objects.Mesh(mesh) )
figMaterialMesh.append( materialPoints )
#figMaterialMesh.show() 




# Setting Values to Materials
# ---------

# Viscosity Functions
# -

# In[48]:


#Arrhenius viscosity

arr = False

if arr:
    eta0 = 1.0e-6
    activationEnergy = 27.63102112
    fn_viscosity = eta0 * fn.math.exp( activationEnergy / (temperatureField+1.) )

#F-K approximation

FK = False #toggle to use temp depend viscosity

if FK :
    surfEtaCont = 1.0e4    #highest viscosity for continents
    surfEtaWeak = 1.0e3
    surfEtaWeak2 = 1.0e2
    surfEtaMantle = 1.0e3  #highest viscosity for mantle
    cEtaCont = numpy.log(surfEtaCont) / tempMax
    cEtaWeak = numpy.log(surfEtaWeak) /tempMax
    cEtaWeak2 = numpy.log(surfEtaWeak2) /tempMax
    cEtaMantle = numpy.log(surfEtaMantle) / tempMax

else :
    cEtaCont = 1e4
    cEtaWeak = 1e3
    cEtaWeak2 = 1e0
    cEtaMantle = 1.0

refcEtaMap  = {      continentalMaterial : cEtaCont, 
                     weakMaterial : cEtaWeak, weakMaterial2 : cEtaWeak2,
                     refmantleMaterial : cEtaMantle }

refcEtaFn    = fn.branching.map( fn_key = materialIndex, mapping = refcEtaMap )

if FK :
    background_viscosity = uw.function.math.exp(refcEtaFn *(tempMax-temperatureField))

else :   
    background_viscosity = refcEtaFn 



# Density & Buoyancy Functions
# --

# 
# $$
#     Ra = \frac{\alpha\rho g \Delta T h^3}{\kappa \eta_{ref}}   ;   Rb = \frac{ \Delta\rho g h^3}{\kappa\eta_{ref}}
# $$
# 

# In[52]:


#density


densCont = -0.95
densWeak = -0.98
densWeak2 = 0.0
densMantle = 0.0

refDensMap = {       continentalMaterial: densCont,
                     weakMaterial: densWeak, weakMaterial2: densWeak2,
                     refmantleMaterial: densMantle}

# Rayleigh number.
Ra = 1.0e7  # make sure this matches what you used in your start-up models.  also, watch your resolution if you set this higher

Rb = 1.0e7  #sets up buoyancy scheme  

# Define our vertical unit vector using a python tuple (this will be automatically converted to a function).
z_hat = ( 0.0, 1.0 )

contbuoy = False # set this to true if you plan on using different densities for the continental material

if contbuoy:
    # construct the density function using material properties outlined above
    densityFn = fn.branching.map( fn_key = materialIndex, mapping = refDensMap )
    # creating a buoyancy force vector
    buoyancyFn = (Ra * temperatureField - Rb * densityFn)  * z_hat
    
else:
    # Construct our density function.
    densityFn = Ra * temperatureField
    # Now create a buoyancy force vector using the density and the vertical unit vector. 
    buoyancyFn = densityFn * z_hat


# Yielding Functions
# --

# In[53]:


yielding = False

if yielding:
    
    stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = velBC,
                            fn_viscosity  = background_viscosity, 
                            fn_bodyforce  = buoyancyFn )

    # get the default stokes equation solver
    solver = uw.systems.Solver( stokes )
    solver.solve(nonLinearIterate=yielding)
    
      # first define strain rate tensor
    strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
    strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)
    
    figStrainRate=vis.Figure()
    figStrainRate.append(vis.objects.Surface(mesh,strainRate_2ndInvariantFn))
    #figStrainRate.show()
    
    
    frictionInf     = fn.misc.constant(0.1)
    frictionFn      = frictionInf 
    cohesion       = fn.misc.constant(4e2)
    rho = fn.misc.constant(1.)
    g   = fn.misc.constant(10.)
    coord  = fn.coord()
    fn_depth         = (coord[1] - 1.)*-1.*1e6
    figDepth=vis.Figure()
    figDepth.append(vis.objects.Surface(mesh, fn_depth))
    figDepth.show()
    fn_pressure_lith = (rho * g * fn_depth)
    figLithPress=vis.Figure()
    figLithPress.append(vis.objects.Surface(mesh,fn_pressure_lith))
    figLithPress.show()
    yieldStressFn   = cohesion + (frictionFn * fn_pressure_lith)
    figYieldStress = vis.Figure(title="Yield Stress", quality=3)
    #figYieldStress.append( vis.objects.Points(swarm, fn.misc.max(0.0,fn.misc.min(yieldStressFn, 1.0)) , pointSize=3.0, fn_mask=materialIndex,colours="#00BBFF:0.5 #FF5500:0.5") )
    figYieldStress.append( vis.objects.Surface(mesh, yieldStressFn, pointSize=3.0,colours="#00BBFF:0.5 #FF5500:0.5") )
    #figYieldStress.show()


    # now compute a viscosity assuming yielding
    min_viscosity = 1.0 
    yieldingViscosityFn =  0.5 * yieldStressFn / (strainRate_2ndInvariantFn+1.0e-18)
    fn_viscosity = fn.exception.SafeMaths( fn.misc.max(fn.misc.min(yieldingViscosityFn, 
                                                              background_viscosity), 
                                                  min_viscosity))

else:
    fn_viscosity = background_viscosity


figEta = vis.Figure(title=" Viscosity", quality=3)
figEta.append ( vis.objects.Points(swarm,fn_colour = uw.function.math.log10(fn_viscosity), fn_size=7 ))
#figEta.show() 




# Bookkeeping
# -----
# Where should we keep our results?

# In[55]:


outputPath = '/home/cmcooper/marginsandextraWZcratonRa1e7_BOff/'
# Make output directory if necessary

if rank==0:
   import os
   if not os.path.exists(outputPath):
      os.makedirs(outputPath)

writefigures = True  #toggle to set whether to write figures to output directory

# Output model timestep info

if  rank==0:
    start = timekeeper.time()

    fw = open(outputPath + "FrequentOutput.dat","w")
    fw.write("%s \n" %(timekeeper.ctime()))
    fw.close()
    


# System Setup
# -------
# **Setup a Stokes system**
# 
# Underworld uses the Stokes system to solve the incompressible Stokes equations.  

# In[56]:


stokes = uw.systems.Stokes( velocityField = velocityField, 
                            pressureField = pressureField,
                            conditions    = velBC,
                            fn_viscosity  = fn_viscosity, 
                            fn_bodyforce  = buoyancyFn )

# get the default stokes equation solver
solver = uw.systems.Solver( stokes )


# **Set up the advective diffusive system**
# 
# Underworld uses the AdvectionDiffusion system to solve the temperature field given heat transport through the velocity field.

# In[57]:


advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField, 
                                         fn_diffusivity = 1.0, 
                                         conditions     = tempBC )

# Create a system to advect the swarm YOU MUST USE THIS IF YOU USE SWARMS TO PUT IN MATERIALS
advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )


# **Analysis Tools**

# In[58]:


nuTop    = uw.utils.Integral( fn=temperatureField.fn_gradient[1], 
                              mesh=mesh, integrationType='Surface', 
                              surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"])

nuBottom = uw.utils.Integral( fn=temperatureField,               
                              mesh=mesh, integrationType='Surface', 
                              surfaceIndexSet=mesh.specialSets["MinJ_VertexSet"])


# In[59]:


nu = - nuTop.evaluate()[0]/nuBottom.evaluate()[0]
#print('Nusselt number = {0:.6f}'.format(nu))


# In[60]:


intVdotV = uw.utils.Integral( fn.math.dot( velocityField, velocityField ), mesh )

vrms = math.sqrt( intVdotV.evaluate()[0]/ volume [0] )
#print('Initial vrms = {0:.3f}'.format(vrms))


# **Heat Fluxes**

# In[61]:


surfaceHF = uw.utils.Integral( fn = temperatureField.fn_gradient[1], mesh = mesh, integrationType = "surface", surfaceIndexSet = TopWall)
bottomHF = uw.utils.Integral( fn = temperatureField.fn_gradient[1], mesh = mesh, integrationType = "surface", surfaceIndexSet = BottomWall)
leftHF = uw.utils.Integral( fn = temperatureField.fn_gradient[0], mesh = mesh, integrationType = "surface", surfaceIndexSet = LeftWall)
rightHF = uw.utils.Integral( fn = temperatureField.fn_gradient[0], mesh = mesh, integrationType = "surface", surfaceIndexSet = RightWall)


# Main time stepping loop
# -----

# In[62]:


# init these guys

if  writefigures:
    figtemp.save_image(outputPath +"TemperatureField_%i" %initstep)
    figEta.save_image(outputPath +"Viscosity_%i" %initstep)
    figMaterialMesh.save_image(outputPath +"Materials_%i" %initstep)

steps_end = 10000 + initstep
checkpointstep = 250

start = timekeeper.time()

simtime = start

dt = min(advector.get_max_dt(), advDiff.get_max_dt())

if rank ==0:
        fw = open( outputPath + "FrequentOutput.dat","a")
        fw.write("Setup time: %.2f seconds\n" %(timekeeper.time() - start))
        fw.write("--------------------- \n")
        fw.write("Step \t Time \t Stopwatch \t Average Temperature \t Nusselt Number \t Vrms \t Surface Heat Flux \t Bottom Heat Flux \t Other Walls Heat Flux \n")
        start = timekeeper.time()
        fw.close()


if rank == 0:
      start = timekeeper.time() # Setup clock to calculate simulation CPU time.

trackHF = True

if trackHF:    
    arrMeanTemp = numpy.zeros((steps_end-initstep)+1)
    arrSurfHF = numpy.zeros((steps_end-initstep)+1) 
    arrOtherWallsHF = numpy.zeros((steps_end-initstep)+1) 
    arrMaxTemp = numpy.zeros((steps_end-initstep)+1) 
    arrNu = numpy.zeros((steps_end-initstep)+1)
    arrVrms = numpy.zeros ((steps_end-initstep)+1)
    

# perform timestepping

while step < steps_end:
    # Solve for the velocity field given the current temperature field.
    solver.solve(nonLinearIterate=yielding)
    dt = min(advector.get_max_dt(), advDiff.get_max_dt())
    advector.integrate(dt)
    advDiff.integrate(dt)
    simtime += dt
    time += dt
    step += 1
    avTemperature = temperatureFieldIntegral.evaluate()[0]/volume[0]
    vrms = math.sqrt( intVdotV.evaluate()[0] / volume[0])
    nu = - nuTop.evaluate()[0]/nuBottom.evaluate()[0]
            
    if trackHF:
        
        surfHF = -1. * surfaceHF.evaluate()[0]
        bottHF = -1. * bottomHF.evaluate()[0]
        wallsHF = abs(bottomHF.evaluate()[0])+abs(leftHF.evaluate()[0])+abs(rightHF.evaluate()[0])
        
        if rank == 0:
            arrMeanTemp[step-initstep] = avTemperature
            arrSurfHF[step-initstep] = surfHF
            arrOtherWallsHF[step-initstep] = wallsHF
            arrNu[step-initstep] = nu
            arrVrms[step-initstep] = vrms

    if rank==0:
        fw = open( outputPath  + "FrequentOutput.dat","a")
        fw.write("%i \t %.2f \t %.2f \t  %.5f \t %.5f \t %.5f \t %.5f \t %.5f \t %.5f \t \n" %(step, time, timekeeper.time() - start, avTemperature, nu, vrms, surfHF, bottHF, wallsHF ))
        start = timekeeper.time()
        fw.close()
        
    if step % checkpointstep == 0.:
        MeshHand=mesh.save(outputPath + "mesh_%i.h5" %step)
        TempInfo=temperatureField.save(outputPath +"temperature_%i.h5" %step, MeshHand )
        SwarmInfo=swarm.save(outputPath +"swarm_%i.h5" %step) 
        SwarmVarInfo=materialIndex.save(outputPath + "materialIndex_%i.h5" %step)
        VelInfo=velocityField.save(outputPath + "velocityField_%i.h5" %step, MeshHand)
        PressInfo=pressureField.save(outputPath + "pressureField_%i.h5" %step, MeshHand)
        velocityField.xdmf (outputPath + "velocityField_%i.xdmf" %step, VelInfo,"Velocity", MeshHand, "Mesh")
        temperatureField.xdmf(outputPath + 'temperature_%i.xdmf' %step, TempInfo, "Temperature", MeshHand, "Mesh")
        pressureField.xdmf(outputPath + 'pressure_%i.xdmf' %step, PressInfo, "Pressure", MeshHand, "Mesh")
        materialIndex.xdmf(outputPath + 'materials_%i.xdmf' %step, SwarmVarInfo, "Materials", SwarmInfo, "Swarm")
        
    if step % checkpointstep == 0.:
        figtemp.save_image(outputPath +"TemperatureField_%i" %step)
        figEta.save_image(outputPath +"Viscosity_%i" %step)
        figMaterialMesh.save_image(outputPath +"Materials_%i" %step)
    
    if step % 10.0 == 0.0:
        population_control.repopulate()
        
