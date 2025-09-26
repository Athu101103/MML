# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 19:20:05 2025

@author: shwet
"""
import matplotlib.pyplot as plt
import pandas as pd

#moving average
years=[1992,1993,1994,1995]
data = [293,246,231,282,301,252,227,291,304,259,239,296,306,265,240,300]
year=list(range(1,17))

#1 - year , 2-quarter , 3-data,  4-quarter moving total , 5- 4quarter moving total avg , 6 - 4quarter centered moving avg, 7 - ratio of centered to actual (6)/(3)

#linear regression
n=len(data)
sum_x=sum(year)
sum_y=sum(data)
sum_xy = sum([x*y for x,y in zip(year,data)])
sum_x2 = sum([x**2 for x in year])
#y=a+bx
b= ((n*sum_xy) - (sum_x*sum_y))/((n*sum_x2)-(sum_x**2))
a = (sum_y - (b*sum_x))/n


y_pred = [a + b*x for x in year]
plt.scatter(year, data)
plt.plot(year,y_pred, "-",color="red")
plt.show()

###############################################
#rel cyc residual
#trend percentage


###############################################
#moving avg

moving_total=[]
moving_total_avg=[]

for i in range(len(data)-3):
    moving_total.append(sum(data[i:i+4]))
    moving_total_avg.append(sum(data[i:i+4])/4)
    
centered_moving_avg = []
for i in range(len(moving_total_avg)-1):
    centered_moving_avg.append(sum(moving_total_avg[i:i+2])/2)
    

actual_to_moving=[]
for i in range(len(centered_moving_avg)):
    val=(data[i+2]/centered_moving_avg[i])*100
    actual_to_moving.append(val)
    
df=pd.DataFrame(
    {
         "year": [1992]+[""]*3 + [1993]+[""]*3 + [1994]+[""]*3 + [1995]+[""]*3,
         "quarters": ["Winter","Spting","Summer","Fall"] * 4,
         "Data": data,
         "4 Quarter moving total" : [None]*2 + moving_total + [None],
         "4 Quarter moving avg": [None]*2 + moving_total_avg + [None],
         "4 Quarter centered moving avg" : [None]*2 + centered_moving_avg + [None]*2,
         "Percentage of actual to moving" :[None]*2 + actual_to_moving + [None]*2
     
     })

#print(df["year"],df["quarters"],df["Data"],df["4 Quarter moving total"], df["4 Quarter moving avg"],df["4 Quarter centered moving avg"], df["Percentage of actual to moving"] )

print(df.to_string(float_format="{:.2f}".format))
#################################################
#seasonal index
quarters={0:[],1:[],2:[],3:[]}
for i in range(len(actual_to_moving)):
    quarters[(i+2)%4].append(actual_to_moving[i])
    
modified_mean=[]
for q in quarters:
    mini=min(quarters[q])
    maxi=max(quarters[q])
    mean=(sum(quarters[q])-mini-maxi)/2
    print(f"Quarter {q+1}:",quarters[q])
    print(f"Min: {mini} , Max : {maxi}")
    print("MEAN:", mean)
    modified_mean.append(mean)
    
    
total_indices=sum(modified_mean)
if total_indices<400:
    adj_constant=400/total_indices
else:
    adj_constant=total_indices/400
    
seasonal_indice=[x*adj_constant for x in modified_mean]


#deseasonalized data
deseasonalized_data=[]
for i in range(0,len(data),4):
    deseasonalized_data.append(data[i]/(seasonal_indice[0]/100))
    deseasonalized_data.append(data[i+1]/(seasonal_indice[1]/100))
    deseasonalized_data.append(data[i+2]/(seasonal_indice[2]/100))
    deseasonalized_data.append(data[i+3]/(seasonal_indice[3]/100))


df2 = pd.DataFrame({
    "Year": [1992]+[""]*3 + [1993]+[""]*3 + [1994]+[""]*3 + [1995]+[""]*3,
    "Quarter": ["Winter","Spting","Summer","Fall"] * 4,
    "Given data": data,
    "Deseasonalized Data":deseasonalized_data
})
print(df2)


#do regression for deseasonalised data
n=len(deseasonalized_data)
sum_x=sum(year)
sum_y=sum(deseasonalized_data)
sum_xy = sum([x*y for x,y in zip(year,deseasonalized_data)])
sum_x2 = sum([x**2 for x in year])
#y=a+bx
b= ((n*sum_xy) - (sum_x*sum_y))/((n*sum_x2)-(sum_x**2))
a = (sum_y - (b*sum_x))/n


y_pred = [a + b*x for x in year]
plt.scatter(year, data)
plt.plot(year,y_pred, "-",color="red")
plt.show()

#relative cyclic residual
rel_cyclic_resid = [((y_act - y_pre) / y_pre) * 100 for y_pre, y_act in zip(y_pred, data)]
print("Relative cyclic residual :")
for yr,rel in zip(year,rel_cyclic_resid ):
    print(f"{yr} : {rel:.2f}")

#plot original,deaseasonalized and the trend
plt.plot(year, data ,color="black",label="original")
plt.plot(year,deseasonalized_data,color="blue",label="De-seasonalized")
plt.plot(year, y_pred,color="red",label="trend(regression)")
plt.xlabel("Years")
plt.ylabel("Values")
plt.title("Comparison plot")
plt.legend()
plt.show()

########################################################################3
#4

import matplotlib.pyplot as plt
import pandas as pd

years = [1992, 1993, 1994, 1995]
year=list(range(1,17))
#data is yearly spring summer fall winter
data=[293, 246, 231, 282, 301, 252, 227, 291, 304, 259, 239, 296, 306, 265, 240, 300]
#print(year,len(year))
#print(data,len(data))

#fit regression line
sum_y = sum(data)
sum_x= sum(year)
sum_xy = sum([x*y for x,y in zip(year,data)])
sum_x2 = sum([x**2 for x in year])

# y = a + bx
a = ((sum_y * sum_x2) - (sum_x * sum_xy)) / ((len(year) * sum_x2) - sum_x**2)
b = ((len(year) * sum_xy )- (sum_x * sum_y)) / ((len(year) * sum_x2) - sum_x**2)

print(f"The equation that best fits the data is : y = {a:.2f} + {b:.2f}x")
y_pred=[a + b*x for x in year]

plt.scatter(year, data ,color="black")
plt.plot(year, y_pred,color="red")
plt.title(f"Linear Regression : y = {a:.2f} + {b:.2f}x ")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

#4 quarter moving total (4) & (5)
moving_total=[]
moving_total_average=[]
for i in range(len(data)-3):
    moving_total.append(sum(data[i:i+4]))
    moving_total_average.append(sum(data[i:i+4])/4)



#4 quartered centered moving avg (6)
centered_moving_average=[]
for i in range(len(moving_total_average)-1):
  centered_moving_average.append(sum(moving_total_average[i:i+2])/2)


#actual to moving percentage - (3)/(6) x 100
actual_to_moving=[]
for i in range(len(centered_moving_average)):
  actual_to_moving.append((data[i+2]/centered_moving_average[i])*100)

#print("Moving total:",moving_total)
#print("moving_total_average:",moving_total_average)
#print("centered_moving_average",centered_moving_average)
#print("actual_to_moving:",actual_to_moving)

"""display_year=["1991","","","","1992","","","" ,"1993","","","","1994","","","" ,"1995","","",""]
Quarter=["Spring","Summer","Fall","Winter"]*5
df=pd.DataFrame({
    "Year":display_year,
    "Quarter":Quarter,
    "Value (3)":data,
    "Moving Total (4)":moving_total,
    "Moving Total Average (5)":moving_total_average,
    "Centered Moving Average (6)":centered_moving_average,
    "Actual to Moving Percentage (3)/(6)*100":actual_to_moving
})"""

quarters= {0:[],1:[],2:[],3:[]}
for i in range(len(actual_to_moving)):
  quarters[(i+2)%4].append(actual_to_moving[i])
print(quarters)

#min and max val for each quarter
Modified_mean=[]
for q in quarters:
  min=quarters[q][0]
  max=quarters[q][0]
  for val in quarters[q]:
    if val<min:
      min=val
    if val>max:
      max=val
  print(f"Quarter {q+1}:")
  print("MIN : ", min)
  print("MAX : ", max)
  Modified_mean.append((sum(quarters[q])-(min+max))/2)
print(Modified_mean)

total_of_indices=sum(Modified_mean)
print(total_of_indices)

if total_of_indices<400:
  adjusting_constant=400/total_of_indices
else:
  adjusting_constant=total_of_indices/400
print("Adjusting constant:", adjusting_constant)


#seasonal indice
seasonal_indices=[x*adjusting_constant for x in Modified_mean]
print("Seasonal indices:",seasonal_indices)
print("Mean of seasonal indices:",sum(seasonal_indices)/4)

# Create DataFrame for display
display_year = ["1992","","","","1993","","","","1994","","","","1995","","",""]
Quarter = ["Spring","Summer","Fall","Winter"]*4
df = pd.DataFrame({
    "Year": display_year,
    "Quarter": Quarter,
    "Value (3)": data,
    "Moving Total (4)": [None]*3 + moving_total + [None]*0,  # Pad to length 16
    "Moving Total Average (5)": [None]*3 + moving_total_average + [None]*0,  # Pad to length 16
    "Centered Moving Average (6)": [None]*3 + centered_moving_average + [None]*1,  # Pad to length 16
    "Actual to Moving Percentage (3)/(6)*100": [None]*3 + actual_to_moving + [None]*1  # Pad to length 16
})

# Print DataFrame with formatted floating-point numbers
print(df.to_string(float_format="{:.2f}".format))

#plot centered moving average
plt.scatter(year, data ,color="black")
plt.plot(year[3:-1], centered_moving_average,color="red")
plt.title(f"Centered Moving average")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

deseasoned_data=[]
#Deseasonalising data
for i in range(0,len(data),4):
  deseasoned_data.append(data[i]/(seasonal_indices[0]/100))
  deseasoned_data.append(data[i+1]/(seasonal_indices[1]/100))
  deseasoned_data.append(data[i+2]/(seasonal_indices[2]/100))
  deseasoned_data.append(data[i+3]/(seasonal_indices[3]/100))

df2 = pd.DataFrame({
    "Year": display_year,
    "Quarter": Quarter,
    "Given data": data,
    "Deseasonalized Data":deseasoned_data
})

display(df2)

#fitting regression for deseasonalised data

sum_y = sum(deseasoned_data)
sum_x= sum(year)
sum_xy = sum([x*y for x,y in zip(year,deseasoned_data)])
sum_x2 = sum([x**2 for x in year])

# y = a + bx
a = ((sum_y * sum_x2) - (sum_x * sum_xy)) / ((len(year) * sum_x2) - sum_x**2)
b = ((len(year) * sum_xy )- (sum_x * sum_y)) / ((len(year) * sum_x2) - sum_x**2)

print(f"The equation that best fits the data is : y = {a:.2f} + {b:.2f}x")
y_pred=[a + b*x for x in year]

plt.scatter(year, data ,color="black")
plt.plot(year, y_pred,color="red")
plt.title(f"Linear Regression (Deseasonalized): y = {a:.2f} + {b:.2f}x ")
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

#relative cyclic residual
rel_cyclic_resid = [((y_act - y_pre) / y_pre) * 100 for y_pre, y_act in zip(y_pred, data)]
print("Relative cyclic residual :")
for yr,rel in zip(year,rel_cyclic_resid ):
    print(f"{yr} : {rel:.2f}")

#plot original,deaseasonalized and the trend
plt.plot(year, data ,color="black",label="original")
plt.plot(year,deseasoned_data,color="blue",label="De-seasonalized")
plt.plot(year, y_pred,color="red",label="trend(regression)")
plt.xlabel("Years")
plt.ylabel("Values")
plt.title("Comparison plot")
plt.legend()
plt.show()


