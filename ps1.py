import random
import matplotlib.pyplot as plt

#number of steps
n = 100

def analyse(a_ans, r, num_steps):
  ans = [a_ans]
  for i in range(1, num_steps):
    a_ans = r**i * ans[0]
    ans.append(a_ans)
  return ans

#number of a,r values
num_a_values = 5
num_r_values = 100

a_values = []
for i in range(num_a_values):
  a = random.uniform(0.5, 1)
  a_values.append(a)

print("Case 1 : r=0\n")
for i in range(num_a_values):
  #case 1:
  plt.figure()
  plt.title(f'Analysis for a = {a_values[i]:.4f}')
  plt.xlabel('Step')
  plt.ylabel('Value')

  r = 0
  ans = analyse(a_values[i], r, n)
  plt.plot(ans, label=r)

  plt.grid(True)
  plt.show()

print("Case 2 : (0<r<1)\n")
for i in range(num_a_values):
  #case 2:
  plt.figure()
  plt.title(f'Analysis for a = {a_values[i]:.4f}')
  plt.xlabel('Step')
  plt.ylabel('Value')

  for j in range(min(num_r_values, 10)):
    r = random.uniform(0, 1)
    ans = analyse(a_values[i], r, n)
    plt.plot(ans, label=f'{r:.2f}')

  plt.grid(True)
  plt.show()
  
print("Case 3 : (-1<r<0)\n")
for i in range(num_a_values):
  #case 3:
  plt.figure()
  plt.title(f'Analysis for a = {a_values[i]:.4f}')
  plt.xlabel('Step')
  plt.ylabel('Value')

  for j in range(min(num_r_values, 10)):
    r = random.uniform(-1, 0)
    ans = analyse(a_values[i], r, n)
    plt.plot(ans, label=f'{r:.2f}')

  plt.grid(True)
  plt.show()

print("Case 4 : (r>1)\n")
for i in range(num_a_values):
  #case 4:
  plt.figure()
  plt.title(f'Analysis for a = {a_values[i]:.4f}')
  plt.xlabel('Step')
  plt.ylabel('Value')

  j=0
  while j < num_r_values:
    r = random.uniform(-2, 2)
    if abs(r)>1:
      ans = analyse(a_values[i], r, n)
      plt.plot(ans, label=f'{r:.2f}')
      j+=1


  plt.grid(True)
  plt.show()
  

#q2
import matplotlib.pyplot as plt

dosages=[0.1,0.2,0.3]

for dose in dosages:
    vals=[dose]
    for _ in range(100):
        vals.append((vals[-1]/2)+dose)
    plt.title(f"Decay of digoxin")
    plt.plot(vals, label=f"{dose} mg")
    plt.legend(title = "Initial dose")
    plt.xlabel('Dosage Period')
    plt.ylabel('Digoxin Concentration (mg)')


plt.show()

#q3
import random 

n_vals = [500, 1000, 10000, 100000]

# Uniform
for n in n_vals:
  data = random.uniform(size=n)
  counts, bin_edges, _ = plt.hist(data, bins=100, color='gray', edgecolor='black', label=f'Histogram with n={n}')
  bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
  plt.plot(bin_midpoints, counts, marker='o', linestyle='-', color='b', label='Frequency Polygon')

  plt.title(f'Uniform Distribution with n={n}')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.show()

# Exponential
for n in n_vals:
  data = random.exponential(scale=2, size=n)
  counts, bin_edges, _ = plt.hist(data, bins=100, color='gray', edgecolor='black', label=f'Histogram with n={n}')
  bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
  plt.plot(bin_midpoints, counts, marker='o', linestyle='-', color='b', label='Frequency Polygon')

  plt.title(f'Exponential Distribution with n={n}')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.show()

# Weibull
for n in n_vals:
  data = random.weibull(a=0.5, size=n)
  counts, bin_edges, _ = plt.hist(data, bins=100, color='gray', edgecolor='black', label=f'Histogram with n={n}')
  bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
  plt.plot(bin_midpoints, counts, marker='o', linestyle='-', color='b', label='Frequency Polygon')

  plt.title(f'Weibull Distribution with n={n} & a=0.5')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.show()

# Triangular
for n in n_vals:
  data = random.triangular(-100, 0, 100, n)
  counts, bin_edges, _ = plt.hist(data, bins=100, color='gray', edgecolor='black', label=f'Histogram with n={n}')
  bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
  plt.plot(bin_midpoints, counts, marker='o', linestyle='-', color='b', label='Frequency Polygon')

  plt.title(f'Triangular Distribution with n={n}')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.show()
  
  
#q4
'''
import random

n_vals=[500,1000,10000,100000]
  
for n in n_vals:
    data = random.uniform(size=n)

    counts,bin_count,_ = plt.hist(data,bins=100,colour='gray',edgecolor='black')
    bin_mp=(bin_edges[1:]+bin_edges[:-1])/2
    plt.plot(bin_mp,counts,marker='o',linestyle='-')
    
    plt.show()'''

import random
    
n_vals = [500, 1000, 10000, 100000]

# Uniform
for n in n_vals:
  data = [random.uniform(0, 1) for _ in range(n)]
  counts, bin_edges, _ = plt.hist(data, bins=100, color='gray', edgecolor='black', label=f'Histogram with n={n}')
  bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
  plt.plot(bin_midpoints, counts, marker='o', linestyle='-', color='b', label='Frequency Polygon')

  plt.title(f'Uniform Distribution with n={n}')
  plt.xlabel('Value')
  plt.ylabel('Frequency')
  plt.show()


#q5
cost_price = 0.30
selling_price = 0.45
salvage = 0.05
stock = 60

news_types = ['Good', 'Normal', 'Poor']
news_types_limits = [(1, 35), (36, 80), (81, 100)]

demand_levels = [40, 50, 60, 70, 80, 90, 100]

bounds = {
    'Good':   [(1, 3), (4, 8), (9, 23), (24, 43), (44, 78), (79, 93), (94, 100)],
    'Normal': [(1, 10), (11, 28), (29, 68), (69, 88), (89, 96), (97, 100)],
    'Poor':   [(1, 44), (45, 66), (67, 82), (83, 94), (95, 100)]
}

def get_news_type(r):
    for i, (lo, hi) in enumerate(news_types_limits):
        if lo <= r <= hi:
            return news_types[i]

def map_demand(category, r):
    for i, (lo, hi) in enumerate(bounds[category]):
        if lo <= r <= hi:
            return demand_levels[i]

def calculate_metrics(demand):
    global salvage

    sold = min(demand, stock)
    unsold = stock - sold
    unmet = max(0, demand - stock)

    income = sold * selling_price
    tot_salvage = unsold * salvage
    cost = stock * cost_price
    missed_profit = unmet * (selling_price - cost_price)
    profit = income + tot_salvage - cost

    return income, missed_profit, tot_salvage, profit, sold

def simulate(days):
    results = {'news_type': [], 'demand_quantity': [], 'income': [], 'missed_profit': [], 'tot_salvage': [], 'profit': [], 'sold_quantity': []}

    for _ in range(days):
        r_cat = np.random.randint(1, 101)
        category = get_news_type(r_cat)
        results['news_type'].append(category)

        if category == 'Good':
            r_demand = int(np.random.exponential(scale=50)) % 100 + 1
        elif category == 'Normal':
            r_demand = int(np.clip(np.random.normal(50, 10), 0, 100)) + 1
        else:
            r_demand = int(np.random.poisson(50)) % 100 + 1

        demand = map_demand(category, r_demand)
        results['demand_quantity'].append(demand)

        income, missed, salvage, profit, sold = calculate_metrics(demand)
        results['income'].append(income)
        results['missed_profit'].append(missed)
        results['tot_salvage'].append(salvage)
        results['profit'].append(profit)
        results['sold_quantity'].append(sold)

    return results

for N in [200, 500, 1000, 10000]:
    print(f"\nSimulation for {N} Days")
    res = simulate(N)

    table = []
    for i in range(10):
        table.append([
            i+1, res['news_type'][i], res['demand_quantity'][i], stock, res['sold_quantity'][i],
            f"{res['income'][i]:.2f}", f"{res['missed_profit'][i]:.2f}", f"{res['tot_salvage'][i]:.2f}", f"{res['profit'][i]:.2f}"
        ])

    headers = ['Day', 'News Type', 'Demand (#)', 'Bought (#)', 'Sold (#)', 'Sales Income ($)', 'Missed Profit ($)', 'Salvage Income ($)', 'Profit ($)']
    print(tabulate(table, headers=headers, tablefmt='grid'))

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    axs[0].plot(range(1,N+1), res['income'], color='green')
    axs[0].set_title("Sales Income")

    axs[1].plot(range(1,N+1), res['missed_profit'], color='red')
    axs[1].set_title("Missed Profit")

    axs[2].plot(range(1,N+1), res['tot_salvage'], color='blue')
    axs[2].set_title("Salvage Income")

    axs[3].plot(range(1,N+1), res['profit'], color='orange')
    axs[3].set_title("Daily Profit")

    for ax in axs:
        ax.set_xlabel("Day")
        ax.set_ylabel("Amount ($)")

    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
#q6
simulation_time = 1000
interarrival_mean = 10
service_time_mean = 10  # mean service time for Poisson
possible_service_times = [8, 9, 10, 11, 12]

arrival_times = []
current_time = 0

while current_time < simulation_time:
    inter_arrival = np.random.exponential(scale=interarrival_mean)
    current_time += inter_arrival

    if current_time > simulation_time:
        break
    arrival_times.append(current_time)

n_customers = len(arrival_times)

def get_valid_poisson():
    while True:
        val = np.random.poisson(lam=service_time_mean)
        if val in possible_service_times:
            return val

service_times = [get_valid_poisson() for _ in range(n_customers)]

start_service_times = []
departure_times = []
wait_times = []
queue_lengths = []
time_points = []
server_busy_time = 0

server_available_time = 0
queue = []

for i in range(n_customers):
    arrival = arrival_times[i]
    service = service_times[i]

    start_service = max(arrival, server_available_time)
    departure = start_service + service
    wait_time = start_service - arrival

    server_available_time = departure

    start_service_times.append(start_service)
    departure_times.append(departure)
    wait_times.append(wait_time)
    server_busy_time += service

    time_points.append(arrival)
    queue_lengths.append(len([d for d in departure_times if d > arrival]) + 1)

average_wait_time = np.mean(wait_times)
average_queue_length = np.mean(queue_lengths)
utilization = server_busy_time / simulation_time

print("Total customers served: ", n_customers)
print(f"Average waiting time in queue: {average_wait_time:.2f} min")
print(f"Average number of customers waiting: {average_queue_length:.2f}")
print(f"Utilization of booking station: {utilization*100:.2f} %")

plt.figure(figsize=(10, 4))
plt.step(time_points, queue_lengths, where='post')
plt.xlabel("Time (minutes)")
plt.ylabel("Number of customers in system")
plt.title("Sample Path of the Queueing System")
plt.grid(True)
plt.tight_layout()
plt.show()

