'''
Some ploting utils, such as Bland-Altman.
Joshua Stough, with Sush Raghunath

I may add additional plotting as needed.
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


# plotting functions. scatter takes argument c a 2D array of RGBs. 
# By the way, since argument c is named, it's part of kwargs:
# see http://geekodour.blogspot.com/2015/04/args-and-kwargs-in-python-explained.html
# See how it's called in vis_linear_and_BA_plots below.

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, alpha=.5, *args, **kwargs)
    plt.axhline(md,           color='red', linestyle='--')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--')
    

# BA plot with separate bias and confidence for
# the two principle colors found in the **kwargs 'c' variable.
def bland_altman_plot_TwoPop(data1, data2, *args, **kwargs):
    
    if 'c' in kwargs:
        # Pandas, amazing!
        colorSeries = pd.Series([tuple(row) for row in kwargs['c']])
        colorCounts = colorSeries.value_counts()
        # returns the color counts in descending order.
        pop1c, pop2c = colorCounts.index[0], colorCounts.index[1]
        print('TwoPop: found colors {}({}) and {}({})'.\
             format(pop1c, colorCounts[0], pop2c, colorCounts[1]))

        pop1i = colorSeries[colorSeries == pop1c].index.values
        pop2i = colorSeries[colorSeries == pop2c].index.values
        
    
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    print('Dims data1 {}, diff {}'.format(data1.shape, diff.shape))

    plt.scatter(mean, diff, alpha=.5, *args, **kwargs)
#     plt.axhline(md,           color='k', linestyle='-')
#     plt.axhline(md + 1.96*sd, color='k', linestyle='--')
#     plt.axhline(md - 1.96*sd, color='k', linestyle='--')
    plt.axhline(md,           linestyle='-')
    plt.axhline(md + 1.96*sd, linestyle='--')
    plt.axhline(md - 1.96*sd, linestyle='--')
    print('Bias {:.4f}, med bias {:.4f}, BA limits +- {:.4f} (:.4f)'.format(md, np.median(diff), 1.96*sd, sd))
    print('50, 75, 95 abs error percentile:\n {:.4f}  {:.4f}  {:.4f}'.format(np.percentile(abs(diff), 50), np.percentile(abs(diff), 75),
                                                                             np.percentile(abs(diff), 95)))

    if 'c' in kwargs:
        # Now get the md and sd for the two populations.
        md1 = np.mean(diff[pop1i])
        sd1 = np.std(diff[pop1i])
        md2 = np.mean(diff[pop2i])
        sd2 = np.std(diff[pop2i])
        # And plot
        for md, sd, c in [(md1, sd1, pop1c), (md2, sd2, pop2c)]:
            plt.axhline(md,           color=c, linestyle='--')
            plt.axhline(md + 1.96*sd, color=c, linestyle='--')
            plt.axhline(md - 1.96*sd, color=c, linestyle='--')
    

# Call with scatterarg a dictionary containing {'c': color_array} where
# color_array is N x 3 RGB for the N data samples. 
def vis_linear_and_BA_plots(dataclinical, dataauto, strtitle, 
                            datalabels = ['Clinical', 'Auto'], 
                            scatterarg={}, 
                            base_filename = None):

    x = np.asarray([dataclinical, dataauto]).transpose()

    dnum = len(x[:,0])
    print('# of data : ', len(x[:,0]))
    lr = linregress(x[:,0],x[:,1])

    #ax.plot(x, fit[0] * x + fit[1], color='red')
    #ax.scatter(x, y)

    #plt.figure(figsize=(6,6))
    plt.figure()
    #plt.subplot(211)
    plt.scatter(x[:,0], x[:,1], alpha=.5, **scatterarg)
    plt.plot(x[:,0], lr[0] * x[:,0] + lr[1], color='red')
    plt.xlabel(datalabels[0])
    plt.ylabel(datalabels[1])
    plt.title('r = '+str(lr[2]) + ' #'+str(dnum))
    plt.axis('equal')
    plt.axis('square')
    plt.show()
    
    if base_filename:
        # User's responsibility to set the style before getting here.
        # Save the plot, use the provided filename.
        plt.savefig(base_filename+'_LN.pdf', bbox_inches='tight')
        
        plt.savefig(base_filename+'_LN.png', bbox_inches='tight',
                    dpi=200, transparent=False);

        plt.savefig(base_filename+'_LN_transp.png', bbox_inches='tight',
                    dpi=200, transparent=True);

#     plt.figure(figsize=(6,4))
#     #plt.subplot(212)
#     bland_altman_plot(x[:,0], x[:,1], **scatterarg)
#     plt.title(strtitle+' - Bland-Altman Plot (#{})'.format(dnum))
#     plt.ylabel('Clinical - Auto')
#     plt.show()


    #plt.figure(figsize=(6,4))
    plt.figure()
    #plt.subplot(212)
    bland_altman_plot_TwoPop(x[:,0], x[:,1], **scatterarg)
    plt.title(strtitle) # +' - Bland-Altman Plot (#{})'.format(dnum))
    plt.ylabel(datalabels[0] + ' - ' + datalabels[1])
    plt.xlabel('Mean of ('+datalabels[0] + ', ' + datalabels[1] + ')')
    plt.show()
    
    if base_filename:
        # User's responsibility to set the style before getting here.
        # Save the plot, use the provided filename.
        plt.savefig(base_filename+'_BA.pdf', bbox_inches='tight')
        
        plt.savefig(base_filename+'_BA.png', bbox_inches='tight',
                    dpi=200, transparent=False);

        plt.savefig(base_filename+'_BA_transp.png', bbox_inches='tight',
                    dpi=200, transparent=True);
        
    
    