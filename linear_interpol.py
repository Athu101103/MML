#Ws2 Q1
import matplotlib.pyplot as plt

def linear_regression(x_vals, y_vals):
    n = len(x_vals)
    sum_x = sum(x_vals)
    sum_y = sum(y_vals)
    sum_xy = sum([x_vals[i] * y_vals[i] for i in range(n)])
    sum_xx = sum([x_vals[i] * x_vals[i] for i in range(n)])

    slope = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_xx) - (sum_x)**2)
    y_intercept = (sum_y/n) - slope*(sum_x/n)
    return (slope, y_intercept)

speed_vals = [i for i in range(1, 17)]
braking_distances = [3, 6, 11, 21, 32, 47, 65, 87, 112, 110, 171, 204, 241, 282, 325, 376]

speed_change_vals = [braking_distances[i]-braking_distances[i-1] for i in range(1, len(braking_distances))]
print(speed_change_vals) 

plt.plot(speed_change_vals)
plt.savefig("fig")
plt.show()

# Linear regression on speed vs. change
m, c = linear_regression(speed_vals[:-1], speed_change_vals)
print(f"\n\tDifference eqn: Δa = {m:.2f}*speed + {c:.4f}")

# Predicted values
pred_speed_change_vals = [m*x + c for x in speed_vals[:-1]]

# Error
errors = [speed_change_vals[i] - pred_speed_change_vals[i] for i in range(len(speed_change_vals))]

plt.plot(errors, color='r')
plt.show()

