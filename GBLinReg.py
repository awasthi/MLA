import numpy as np
import matplotlib.pyplot as plt

def estimate_coeff(x,y):
    #size of data set
    n=np.size(x)
    mean_x,mean_y = np.mean(x),np.mean(y)

    # cross deviation and deviation about x
    SS_xy = np.sum(y*x - n*mean_y*mean_x)
    SS_xx = np.sum(x*x - n*mean_x*mean_x)
    
    b_1 = SS_xy / SS_xx
    b_0 = mean_y - b_1*mean_x
    return(b_0,b_1)
    
def plot_reg_line(x,y,b):
    plt.scatter(x,y,color="m",marker="o",s=30)
    y_pred = b[0]+b[1]*x #response vector
    
    plt.plot(x,y_pred,color="g")
    
    plt.xlabel('Size')
    plt.ylabel('Cost')
    
    plt.show()
    
def main():
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([300,350,500,700,800,850,900,900,1000,1200])
    
    b = estimate_coeff(x,y)
    print("Estimated coeff: \nb_o={} \nb_1={}",(b[0],b[1]))
    
    plot_reg_line(x,y,b)
    
if __name__== "__main__":
    main()
    