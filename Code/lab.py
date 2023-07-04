from sklearn.cluster import KMeans
from __lib__ import *
import timeit

points = [[1,2],[2,3],[3,4]]
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]
# Plotting the points
plt.scatter(x_coords,y_coords)

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plotting Points')

# Display the plot
plt.show()