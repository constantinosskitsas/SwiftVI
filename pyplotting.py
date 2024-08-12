### Adapted from Online and Reinforcement Learning at Copenhagen University ##
import numpy as np
import pylab as pl


# Plotting function.
def plot(data, names, y_label = "#(Bad Episodes)", exp_name = "error"):
	timeHorizon = len(data[0][0])
	colors= ['black', 'blue', 'purple','cyan','yellow', 'orange', 'red']
	nbFigure = pl.gcf().number+1

	# Average the results and plot them.
	avg_data = []
	pl.figure(nbFigure)
	for i in range(len(data)):
		avg_data.append(np.mean(data[i], axis=0))
		pl.plot(avg_data[i], label=names[i], color=colors[i%len(colors)])

	# Compute standard deviantion and plot the associated error bars.
	step=(timeHorizon//10)
	for i in range(len(data)):
		std_data = 1.96 * np.std(data[i], axis=0) / np.sqrt(len(data[i]))
		pl.errorbar(np.arange(0,timeHorizon,step), avg_data[i][0:timeHorizon:step], std_data[0:timeHorizon:step], color=colors[i%len(colors)], linestyle='None', capsize=10)
	
	# Label and format the plot.
	pl.legend()
	pl.xlabel("Time steps", fontsize=13)
	pl.ylabel(y_label, fontsize=13)
	pl.ticklabel_format(axis='both', useMathText = True, useOffset = True, style='sci', scilimits=(0, 0))

	# Uncomment below to get log scale y-axis.
	#pl.yscale('log')
	#pl.ylim(1)

	# Save the plot.
	name = ""
	for n  in names:
		name += n + "_"
	pl.savefig("pyplotfiles/Figure_" + name + exp_name + '.pdf')
	

startS = 10*10
nA = 4
endS = 20*20
incS = 2
l_ = [7*7-5-1*4]#[4*4,5*5,6*6,7*7,8*8]#[10*10,12*12,14*14] #,16*16,18*18,20*20]
for ite in l_:#range(startS,endS,incS):	
    #mbie_v_opt = np.loadtxt(f"pyplotfiles/mbie_v_opt_{ite}_{nA}.txt", delimiter=" ")
    #mbie_v_pol = np.loadtxt(f"pyplotfiles/mbie_v_pol_{ite}_{nA}.txt", delimiter=" ")
    #mbie_v_pol_e = np.loadtxt(f"pyplotfiles/mbie_v_pol_e{ite}_{nA}.txt", delimiter=" ")
    #mbie_v_opt_e = np.loadtxt(f"pyplotfiles/mbie_v_opt_e{ite}_{nA}.txt", delimiter=" ")
    #print(mbie_v_opt.shape)
    #plot([mbie_v_opt, mbie_v_pol], ["V_star", "V"], y_label="Learned Value over time", exp_name=f"mbie Value over time{ite}_{nA}")
    #plot([mbie_v_opt_e, mbie_v_pol_e], ["Expected_V_star", "Expected_V"], y_label="Learned Value over time", exp_name=f"mbie Expected Value over time{ite}_{nA}")

    swiftmbie_v_opt = np.loadtxt(f"pyplotfiles/swiftmbie_v_opt_{ite}_{nA}.txt", delimiter=" ")
    swiftmbie_v_pol = np.loadtxt(f"pyplotfiles/swiftmbie_v_pol_{ite}_{nA}.txt", delimiter=" ")
    swiftmbie_v_pol_e = np.loadtxt(f"pyplotfiles/swiftmbie_v_pol_e{ite}_{nA}.txt", delimiter=" ")
    swiftmbie_v_opt_e = np.loadtxt(f"pyplotfiles/swiftmbie_v_opt_e{ite}_{nA}.txt", delimiter=" ")
    print(swiftmbie_v_opt.shape)
    plot([swiftmbie_v_opt, swiftmbie_v_pol], ["V_star", "V"], y_label="Learned Value over time", exp_name=f"swiftmbie Value over time{ite}_{nA}")
    plot([swiftmbie_v_opt_e, swiftmbie_v_pol_e], ["Expected_V_star", "Expected_V"], y_label="Learned Value over time", exp_name=f"swiftmbie Expected Value over time{ite}_{nA}")
    
    #baombie_v_opt = np.loadtxt(f"pyplotfiles/baombie_v_opt_{ite}_{nA}.txt", delimiter=" ")
    #baombie_v_pol = np.loadtxt(f"pyplotfiles/baombie_v_pol_{ite}_{nA}.txt", delimiter=" ")
    #baombie_v_pol_e = np.loadtxt(f"pyplotfiles/baombie_v_pol_e{ite}_{nA}.txt", delimiter=" ")
    #baombie_v_opt_e = np.loadtxt(f"pyplotfiles/baombie_v_opt_e{ite}_{nA}.txt", delimiter=" ")
    #print(baombie_v_opt.shape)
    #plot([baombie_v_opt, baombie_v_pol], ["V_star", "V"], y_label="Learned Value over time", exp_name=f"baombie Value over time{ite}_{nA}")
    #plot([baombie_v_opt_e, baombie_v_pol_e], ["Expected_V_star", "Expected_V"], y_label="Learned Value over time", exp_name=f"baombie Expected Value over time{ite}_{nA}")

    uclr_g_v_opt = np.loadtxt(f"pyplotfiles/ucrlg_v_opt_{ite}_{nA}.txt", delimiter=" ")
    uclr_g_v_pol = np.loadtxt(f"pyplotfiles/ucrlg_v_pol_{ite}_{nA}.txt", delimiter=" ")
    uclr_g_v_pol_e = np.loadtxt(f"pyplotfiles/ucrlg_v_pol_e{ite}_{nA}.txt", delimiter=" ")
    uclr_g_v_opt_e = np.loadtxt(f"pyplotfiles/ucrlg_v_opt_e{ite}_{nA}.txt", delimiter=" ")
    print(uclr_g_v_opt.shape)
    plot([uclr_g_v_opt, uclr_g_v_pol], ["V_star", "V"], y_label="Learned Value over time", exp_name=f"UCLR_Gamma Value over time{ite}_{nA}")
    plot([uclr_g_v_opt_e, uclr_g_v_pol_e], ["Expected_V_star", "Expected_V"], y_label="Learned Value over time", exp_name=f"UCLR_Gamma Expected Value over time{ite}_{nA}")
        
        