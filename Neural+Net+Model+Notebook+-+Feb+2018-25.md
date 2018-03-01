
# Neural Network For Composite Materials Design - FEM Solver Trained

The data for the effective property solutions from the FEM solver are written out to txt files. The inputs from the particles radii inform the outputs of the effective properties. This data can then be used to train a neural network as a proxy to solve the system of PDEs inside the Genetic Algorithm and thereby speed up the computation process.

## Data Pre-Processing

The first step is to pre-process the data to yield the required layer of inputs and outputs. 

Start by finding the relevant markers and their positions by appending them into relevant lists and then write the process into a function to be able to iterate over the required lists


```python
import numpy as np
import pandas as pd

def data_extract_txt(table_flat_list, etensor_flat_list, keff_flat_list, file_iter):
    #Open the data file
    f_test = open('mesoscale_ga_data_feb_17_datatest_n10_iter_%i.txt' % file_iter)

    #append all lines to a list, index them and 
    #find the lines that contain the 'Next Table' for a particle table
    # find the lines with 'Etensor_mat' and K_eff_mat to deliminate those matrices

    #lists and counters
    lines_list = []
    counter_next_table = 0
    counter_etensor = 0
    counter_keff = 0
    next_table_list = []
    etensor_list = []
    keff_list = []

    #find the relevant line numbers
    for line in f_test.readlines():
        lines_list.append(line)
        if 'Next Table' in line:
            next_table_list.append(counter_next_table)
        if 'Etensor_mat' in line:
            etensor_list.append(counter_etensor)
        if 'K_eff_mat' in line:
            keff_list.append(counter_keff)

        #increase the counters
        counter_next_table += 1
        counter_etensor += 1
        counter_keff += 1




    #Extract the particle table - from the start of the the table list to the beginning of the Etensor list
    num_structures = len(next_table_list)

    fiber_table1 = np.zeros((150,4))
    etensor1 = np.zeros((6,6))
    keff1 = np.zeros((3,3))

    for ii in range (0, num_structures):
        #Determine the relevant ranges
        #start and stops for the particle table
        table_start = next_table_list[ii]
        table_stop = etensor_list[ii]
        #the etensor
        etensor_start = etensor_list[ii] + 1
        etensor_stop = keff_list[ii] - 1
        #the k tensor
        keff_start = keff_list[ii] + 1
        if (ii == (num_structures -1)):
            keff_stop = len(lines_list) - 2
        else:
            keff_stop = next_table_list[ii+1] - 1

        #Contruction of the Particle Table

        #first element in the table
        table_1_split = lines_list[table_start].split()
        #first element
        parse1 = table_1_split[2].split("[")
        parse2 = parse1[1].split(",")
        float1 = float(parse2[0])
        float2 = float(table_1_split[3].split(",")[0])
        float3 = float(table_1_split[4].split(",")[0])
        float4 = float(table_1_split[5].split("]")[0])

        fiber_table1[0] = np.array([float1, float2, float3, float4])

        #all other elements in the table
        count_table = 1
        for jj in range(table_start+1, table_stop):
            table_2_split = lines_list[jj].split()
            parse1 = table_2_split[0].split("[")
            parse2 = parse1[1].split(",")
            float1 = float(parse2[0])
            float2 = float(table_2_split[1].split(",")[0])
            float3 = float(table_2_split[2].split(",")[0])
            float4 = float(table_2_split[3].split("]")[0])

            fiber_table1[count_table] = np.array([float1, float2, float3, float4])

            count_table += 1

        #Construction of the Etensor
        count_tensor = 0
        for jj in range(etensor_start, etensor_stop):
            tensor_1_split = lines_list[jj].split()
            parse1 = tensor_1_split[0].split("[")
            parse2 = parse1[1].split(",")
            float1 = float(parse2[0])
            float2 = float(tensor_1_split[1].split(",")[0])
            float3 = float(tensor_1_split[2].split(",")[0])
            float4 = float(tensor_1_split[3].split(",")[0])
            float5 = float(tensor_1_split[4].split(",")[0])
            float6 = float(tensor_1_split[5].split("]")[0])

            etensor1[count_tensor] = np.array([float1, float2, float3, float4, float5, float6])

            count_tensor += 1

        #Construction of the Keff
        count_keff = 0
        for jj in range(keff_start, keff_stop):
            keff_1_split = lines_list[jj].split()
            parse1 = keff_1_split[0].split("[")
            parse2 = parse1[1].split(",")
            float1 = float(parse2[0])
            float2 = float(keff_1_split[1].split(",")[0])
            float3 = float(keff_1_split[2].split("]")[0])


            keff1[count_keff] = np.array([float1, float2, float3])

            count_keff += 1


        #Flatten the table, etensor, and keff to prepare for input into the data frame
        table_flat_list.append(fiber_table1.flatten())
        etensor_flat_list.append(etensor1.flatten())
        keff_flat_list.append(keff1.flatten())
```


```python
#iterator for the txt file
table_flat_list = []
etensor_flat_list = []
keff_flat_list = []

for file_iter in range(1,5):

    data_extract_txt(table_flat_list, etensor_flat_list, keff_flat_list, file_iter)
```


```python
len(table_flat_list)
```




    36



Required input for the neural net on scikit is in a the form of a table where each row represents a set of input and output values; therefore there are 600 columns for the inputs and 45 columns for the output


```python
#flatten inputs and outputs into one big data list
pd_data_list = []

num_structures = len(table_flat_list)

for ii in range(0, num_structures):
    list_line = np.array([np.array(table_flat_list[ii]), np.array(etensor_flat_list[ii]), np.array(keff_flat_list[ii])])
    
    #append to list
    pd_data_list.append(list_line)

#create pandas data frame
data_frame_list = pd.DataFrame(pd_data_list, columns = ['Particle Table', 'Etensor', 'Keff'], dtype='float')
```


```python
data_frame_list.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Particle Table</th>
      <th>Etensor</th>
      <th>Keff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.529380427109, 0.0180876623966, 0.9397186344...</td>
      <td>[515323206234.0, 142410309173.0, 142295255807....</td>
      <td>[94.6492968247, 0.0, 0.0, 0.0, 94.649561605, -...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.58514756792, 0.330246801131, 0.230286513296...</td>
      <td>[515906164747.0, 142625196470.0, 142221215287....</td>
      <td>[94.6493096109, 0.0, 0.0, 0.0, 94.6495583475, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.449807278627, 0.554712899686, 0.61702651979...</td>
      <td>[520355550893.0, 143678521742.0, 142997571282....</td>
      <td>[94.6605856501, 0.0, 0.0, 0.0, 94.6607124519, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.489593852868, 0.286400281041, 0.77837257713...</td>
      <td>[529690596238.0, 146720247572.0, 146827158820....</td>
      <td>[94.7272209259, 0.0, 0.0, 0.0, 94.7275333902, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.557263997515, 0.174167231764, 0.58500257389...</td>
      <td>[539427682806.0, 148189965084.0, 148247309752....</td>
      <td>[94.7499684317, 0.0, 0.0, 0.0, 94.7495322182, ...</td>
    </tr>
  </tbody>
</table>
</div>



# Neural Network Setup

Set up the data for the neural net, split up into training data, etc


```python
from sklearn.model_selection import train_test_split

X = data_frame_list['Particle Table']
Y_mech = data_frame_list['Etensor']
Y_therm = data_frame_list['Keff']

#Mechanical
x_train_mech, x_test_mech, y_train_mech, y_test_mech = train_test_split(X, Y_mech)
#Thermal
x_train_therm, x_test_therm, y_train_therm, y_test_therm = train_test_split(X, Y_therm)



#convert the list of arrays back to a numpy array!!! BITHESSSSS
# np1 = np.vstack(x_train)

# np1.shape
```

Use the standard scalar to pre-process the data with scaling


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# #fit only to the training data
scaler.fit(np.vstack(x_train_mech))
scaler.fit(np.vstack(x_train_therm))


# Now apply the transformations to the data:
x_train_mech = scaler.transform(np.vstack(x_train_mech))
x_test_mech = scaler.transform(np.vstack(x_test_mech))
x_train_therm = scaler.transform(np.vstack(x_train_therm))
x_test_therm = scaler.transform(np.vstack(x_test_therm))
```

Train the model



```python
from sklearn.neural_network import MLPRegressor

mlp_etensor = MLPRegressor(hidden_layer_sizes=(500,400,300,200,100), max_iter = 5000000)
mlp_etensor.fit(np.vstack(x_train_mech),np.vstack(y_train_mech))

mlp_keff = MLPRegressor(hidden_layer_sizes=(500,400,300,200,100,50), max_iter = 5000000)
mlp_keff.fit(np.vstack(x_train_therm),np.vstack(y_train_therm))
```




    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(500, 400, 300, 200, 100, 50),
           learning_rate='constant', learning_rate_init=0.001,
           max_iter=5000000, momentum=0.9, nesterovs_momentum=True,
           power_t=0.5, random_state=None, shuffle=True, solver='adam',
           tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)



# Predictions and Evaluations

Use the predict feature in the scikit learn tool


```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error

predictions_mech = mlp_etensor.predict(np.vstack(x_test_mech))
predictions_therm = mlp_keff.predict(np.vstack(x_test_therm))
```


```python
mean_squared_error(predictions_mech, np.vstack(y_test_mech))
```




    2.9899604471368846e+21




```python
mean_squared_error(predictions_therm, np.vstack(y_test_therm))
```




    72.898500381947045




```python
def error_func(predictions, test_data):
    error = 0.0
    for ii in range(0, len(test_data)):
        error += np.linalg.norm((predictions[ii]-test_data[ii]))/np.linalg.norm(test_data[ii])
        
    return error
```


```python
predictions_mech[0]
```




    array([  5.18150505e+11,   1.43194368e+11,   1.43267778e+11,
            -8.29882272e+07,  -1.82065991e+07,   1.83635666e+08,
             1.43194351e+11,   5.20006377e+11,   1.43524168e+11,
            -2.37004519e+08,   3.93470506e+08,   4.07874126e+07,
             1.43267709e+11,   1.43524160e+11,   5.20547566e+11,
            -5.57958085e+07,   2.58925372e+08,   8.36956068e+07,
            -1.46050955e+08,  -2.33738872e+08,  -6.20594523e+07,
             1.90152137e+11,   3.13759696e+07,   8.50625828e+07,
            -3.44964006e+07,   3.81108790e+08,   2.60268144e+08,
             3.93800079e+07,   1.90298264e+11,  -2.42511590e+08,
             1.67730951e+08,   4.16834909e+07,   8.17005345e+07,
             8.48078245e+07,  -1.39451328e+08,   1.90115453e+11])




```python
np.vstack(y_test_mech)[0]
```




    array([  5.44508910e+11,   1.50398777e+11,   1.51305792e+11,
            -2.64901322e+08,   7.28642326e+07,   6.53537902e+08,
             1.50398777e+11,   5.45882169e+11,   1.51493918e+11,
            -4.74130250e+08,   1.22943502e+09,   3.87286176e+08,
             1.51305792e+11,   1.51493918e+11,   5.52473252e+11,
            -1.31106227e+08,   1.51237588e+09,   3.02839294e+08,
            -2.64901322e+08,  -4.74130250e+08,  -1.31106227e+08,
             2.01712653e+11,  -3.74983509e+07,   2.18167540e+08,
             7.28642326e+07,   1.22943502e+09,   1.51237588e+09,
            -3.74983509e+07,   2.03444619e+11,  -7.96441722e+07,
             6.53537902e+08,   3.87286176e+08,   3.02839294e+08,
             2.18167540e+08,  -7.96441722e+07,   2.02354566e+11])




```python
error_func(predictions_mech, np.vstack(y_test_mech))
```




    1.5286513932726207



Save the model using the pickel feature


```python
predictions_therm[0]
```




    array([ 80.11102966,   1.09318754,   0.19883517, -10.78158052,
            90.38448289, -11.85766343,   7.29640899,  -2.01917847,  78.26026394])




```python
np.vstack(y_test_therm)[0]
```




    array([  9.47497741e+01,   0.00000000e+00,   2.84217094e-14,
             0.00000000e+00,   9.47493095e+01,   0.00000000e+00,
             0.00000000e+00,   0.00000000e+00,   9.47499347e+01])




```python
error_func(predictions_therm, np.vstack(y_test_therm))
```




    1.3962803417568539




```python
import pickle

small_cnn_mech = mlp_etensor
model_name_mech = 'small_neural_net_test_etensor.sav'
pickle.dump(small_cnn_mech, open(model_name_mech, 'wb'))

small_cnn_therm = mlp_keff
model_name_therm = 'small_neural_net_test_keff.sav'
pickle.dump(small_cnn_therm, open(model_name_therm, 'wb'))
```
