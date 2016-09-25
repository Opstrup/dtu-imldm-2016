# exercise 2.1.2
from ex2_1_1 import *
# (requires data structures from ex. 2.1.1)

from pylab import *

# Data attributes to be plotted
i = 0
j = 1

firei = 0
firej = 1

##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type and need to be cast to array. 
figure()
X = array(X)
fireX = array(fireX)
#plot(X[:,i], X[:,j], 'o');

plot(fireX[:,i], fireX[:,j], 'o');
# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
f.hold()
#title('NanoNose data')
title('ForestFire data')

for c in range(fireC):
    # select indices belonging to class c:
    class_mask = y.A.ravel()==c
#    plot(X[class_mask,i], X[class_mask,j], 'o')
    plot(fireX[class_mask,firei], fireX[class_mask,firej], 'o')

#legend(classNames)
#xlabel(attributeNames[i])
#ylabel(attributeNames[j])
legend(fireClassNames)
xlabel(fireAttributeNames[i])
ylabel(fireAttributeNames[j])

# Output result to screen
show()