<a href="https://colab.research.google.com/github/wesleybeckner/general_applications_of_neural_networks/blob/main/notebooks/S5_Time_Series_Analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# General Applications of Neural Networks <br> Session 5: Recurrent Neural Networks and Time Series Analysis

**Instructor**: Wesley Beckner

**Contact**: wesleybeckner@gmail.com

---

<br>

In this session, we'll be exploring NN as they apply to sequenced data, specifically time series data.

<br>

---

<br>

<a name='top'></a>

<a name='x.0'></a>

## 5.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 5.0.1 Import Packages

[back to top](#top)


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import random
from scipy.stats import gamma, norm, expon
from ipywidgets import interact
from statsmodels.tsa.stattools import pacf, acf
from sklearn.metrics import mean_squared_error

def melt_results(model, X, y):
  y_pred = model.predict(X)
  results = pd.DataFrame(y_pred, y)
  results = results.reset_index()
  results.index = data['Date'][window_size:]
  results = results.reset_index()
  results.columns=['Date', 'real', 'predicted']
  results = results.melt(id_vars='Date', var_name='Source', value_name='KG')
  return results

def process_data(Xy, window=3, time_cols=12, remove_null=False):
  """
  This function splits your time series data into the proper windows

  Parameters
  ----------
  Xy: array
    The input data. If there are non-time series columns, assumes they are on
    the left and time columns are on the right. 
  time_cols: int
    The number of time columns, default 12
  window: int
    The time window size, default 3

  Returns
  -------
  X_: array
    The independent variables, includes time and non-time series columns with
    the new window
  y_: array
    The dependent variable, selected from the time columns at the end of the 
    window
  labels:
    The time series labels, can be used in subsequent plot
  """
  # separate the non-time series columns
  X_cat = Xy[:,:-time_cols]

  # select the columns to apply the sweeping window
  X = Xy[:,-time_cols:]

  X_ = []
  y = []

  for i in range(X.shape[1]-window):
    # after attaching the current window to the non-time series 
    # columns, add it to a growing list
    X_.append(np.concatenate((X_cat, X[:, i:i+window]), axis=1))

    # add the next time delta after the window to the list of y
    # values
    y.append(X[:, i+window])

  # X_ is 3D: [number of replicates from sweeping window,
  #           length of input data, 
  #           size of new feature with categories and time]
  # we want to reshape X_ so that the replicates due to the sweeping window is 
  # a part of the same dimension as the instances of the input data
  X_ = np.array(X_).reshape(X.shape[0]*np.array(X_).shape[0],window+X_cat.shape[1])
  y = np.array(y).reshape(X.shape[0]*np.array(y).shape[0],)

  if remove_null:
    # remove training data where the target is 0 (may be unfair advantage)
    X_ = X_[np.where(~np.isnan(y.astype(float)))[0]]
    y = y[np.where(~np.isnan(y.astype(float)))[0]]

  # create labels that show the previous month values used to train the model
  labels = []
  for row in X_:
    labels.append("X: {}".format(np.array2string(row[-window:].astype(float).round())))
  return X_, y, labels

def train_test_process(data, train_test_val_ratios = [0.6, 0.8], window_size=3):
  # get the indices at the associated ratios
  idx_split1 = int(data.shape[1]*train_test_val_ratios[0])
  idx_split2 = int(data.shape[1]*train_test_val_ratios[1])

  # index the data to build the sets
  data_train = data[:,:idx_split1]
  data_val = data[:,idx_split1:idx_split2]
  data_test = data[:,idx_split2:]

  # build out the training sets with the sweeping window method
  X_train, y_train, labels = process_data(data_train, window=window_size, time_cols=132)
  X_val, y_val, labels = process_data(data_val, window=window_size, time_cols=132)
  X_test, y_test, labels = process_data(data_test, window=window_size, time_cols=132)

  print("train size: {}".format(X_train.shape[0]))
  print("val size: {}".format(X_val.shape[0]))
  print("test size: {}".format(X_test.shape[0]), end='\n\n')

  return X_train, y_train, X_val, y_val, X_test, y_test
```

<a name='x.0.2'></a>

### 5.0.2 Load Dataset

[back to top](#top)


```python
orders = pd.read_csv("https://raw.githubusercontent.com/wesleybeckner/"\
                     "truffletopia/main/truffletopia/data/12_year_orders.csv")

cat_cols = ['base_cake', 'truffle_type', 'primary_flavor', 'secondary_flavor',
       'color_group', 'customer']
       
time_cols = [i for i in orders.columns if i not in cat_cols]

# note that our data is 'untidy' if we wanted to tidy the data we would need to
# unpivot or 'melt' our date columns like so:
orders.melt(id_vars=cat_cols, var_name='date', value_name='kg')

# however the data as it is, is useful for our purposes of timeseries prediction 
# today
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>base_cake</th>
      <th>truffle_type</th>
      <th>primary_flavor</th>
      <th>secondary_flavor</th>
      <th>color_group</th>
      <th>customer</th>
      <th>date</th>
      <th>kg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cheese</td>
      <td>Candy Outer</td>
      <td>Horchata</td>
      <td>Vanilla</td>
      <td>Amethyst</td>
      <td>Perk-a-Cola</td>
      <td>1/2010</td>
      <td>12570.335165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Irish Cream</td>
      <td>Egg Nog</td>
      <td>Slate</td>
      <td>Dandy's Candies</td>
      <td>1/2010</td>
      <td>7922.970436</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sponge</td>
      <td>Chocolate Outer</td>
      <td>Ginger Ale</td>
      <td>Apple</td>
      <td>Slate</td>
      <td>Dandy's Candies</td>
      <td>1/2010</td>
      <td>10521.306722</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cheese</td>
      <td>Chocolate Outer</td>
      <td>Coffee</td>
      <td>Pear</td>
      <td>Opal</td>
      <td>Dandy's Candies</td>
      <td>1/2010</td>
      <td>4739.122200</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chiffon</td>
      <td>Jelly Filled</td>
      <td>Butter Toffee</td>
      <td>Apricot</td>
      <td>Olive</td>
      <td>Slugworth</td>
      <td>1/2010</td>
      <td>2756.891961</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13195</th>
      <td>Chiffon</td>
      <td>Chocolate Outer</td>
      <td>Acai Berry</td>
      <td>Tangerine</td>
      <td>Slate</td>
      <td>Fickelgruber</td>
      <td>12/2020</td>
      <td>25714.512372</td>
    </tr>
    <tr>
      <th>13196</th>
      <td>Butter</td>
      <td>Jelly Filled</td>
      <td>Plum</td>
      <td>Peppermint</td>
      <td>Olive</td>
      <td>Fickelgruber</td>
      <td>12/2020</td>
      <td>15043.303525</td>
    </tr>
    <tr>
      <th>13197</th>
      <td>Chiffon</td>
      <td>Chocolate Outer</td>
      <td>Wild Cherry Cream</td>
      <td>Peppermint</td>
      <td>Taupe</td>
      <td>Perk-a-Cola</td>
      <td>12/2020</td>
      <td>8769.613116</td>
    </tr>
    <tr>
      <th>13198</th>
      <td>Cheese</td>
      <td>Candy Outer</td>
      <td>Mango</td>
      <td>Mango</td>
      <td>Rose</td>
      <td>Dandy's Candies</td>
      <td>12/2020</td>
      <td>5065.975534</td>
    </tr>
    <tr>
      <th>13199</th>
      <td>Sponge</td>
      <td>Chocolate Outer</td>
      <td>Ginger Ale</td>
      <td>Passion Fruit</td>
      <td>Black</td>
      <td>Fickelgruber</td>
      <td>12/2020</td>
      <td>9466.712219</td>
    </tr>
  </tbody>
</table>
<p>13200 rows × 8 columns</p>
</div>



<a name='x.1'></a>

## 5.1 Why We Think in Sequences

[back to top](#top)

There are some problems that are best framed as a sequence in either the input or the output. For example, in our image classification we are performing a mapping of many-to-one: sequence input (the pixels) to a single output (classification). Other examples include:

* One-to-many: sequence output, e.x. word (if treated as a single input) to generate a picture
* Many-to-many: sequence input and output, e.x. machine translation (like english to mandarin)
* Synchronized many-to-many: synced sequence input and output, e.x. video classification

State of the art handling of sequences has occurred in a class of networks called recurrent neural networks

<a name='x.2'></a>

## 5.2 Recurrent Neural Networks

[back to top](#top)

Recurrent Neural Networks (RNNs) can be thought of as a FFNN with loops added into the architecture. This allows the network to retain information, create "memory" that can be associated with signals later in the sequence. 

We didn't go into much detail about the actual training algorithm of neural networks: **_back propagation_**. But what we will say here, is that this algorithm breaks down with recurrent neural networks because of the looped connections. A trick was created to overcome this, where the looped connections are unrolled, using a copy of the "unhooked" neuron to represent where the loop was initally fed back. This algorithm is called **_back propagation through time_**.

Another problem is introduced when training recurrent neural networks, in that the gradients calculated during back propagation can become very large, **_exploding gradients_**, or very small **_vanishing gradients_**. This problem is modulated in FNNNs by the ReLU, In RNNs, a more sophisticated gating mechanism is used in an architecture we call **_Long Short-Term Memory Networks_**

<p align=center>
<img src="https://miro.medium.com/max/4136/1*SKGAqkVVzT6co-sZ29ze-g.png"></img>
</p>
<small>LSTM shown in both typical and unfolded format</small>



### 5.2.1 Long Short-Term Memory Networks

[back to top](#top)

Long Short-Term Memory Networks (LSTMs) are a type of RNN that are trained using back propagation through time and overcome the vanishing/exploding gradient problem. Similar to CNNs, their architecture is composed of blocks, this time with memory blocks rather than convolutional blocks. A block is smarter than the classical neuron; it contains gates that manage the block's state and output. The gates are operated by a sigmoid function, determining whether they are open or closed (triggered or not trigerred). There are three types of gates within a memory block:

* Forget gate: decides what information is discarded
* Input gate: decides what information updates the memory state
* Output gate: decides what information to send forward depending on the input and memory state

These weights that configure these gates are learned during training, and their coordination allow each memory block to learn sophisticated relationships in and among sequenced data. 

> Big takeaway: memory blocks contain trainable parameters that allow the block to learn relationships between sequenced data 


<a name='x.3'></a>

## 5.3 Exploratory Data Analysis with Plotly/Pandas

[back to top](#top)


```python
orders.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>base_cake</th>
      <th>truffle_type</th>
      <th>primary_flavor</th>
      <th>secondary_flavor</th>
      <th>color_group</th>
      <th>customer</th>
      <th>1/2010</th>
      <th>2/2010</th>
      <th>3/2010</th>
      <th>4/2010</th>
      <th>5/2010</th>
      <th>6/2010</th>
      <th>7/2010</th>
      <th>8/2010</th>
      <th>9/2010</th>
      <th>10/2010</th>
      <th>11/2010</th>
      <th>12/2010</th>
      <th>1/2011</th>
      <th>2/2011</th>
      <th>3/2011</th>
      <th>4/2011</th>
      <th>5/2011</th>
      <th>6/2011</th>
      <th>7/2011</th>
      <th>8/2011</th>
      <th>9/2011</th>
      <th>10/2011</th>
      <th>11/2011</th>
      <th>12/2011</th>
      <th>1/2012</th>
      <th>2/2012</th>
      <th>3/2012</th>
      <th>4/2012</th>
      <th>5/2012</th>
      <th>6/2012</th>
      <th>7/2012</th>
      <th>8/2012</th>
      <th>9/2012</th>
      <th>10/2012</th>
      <th>...</th>
      <th>9/2017</th>
      <th>10/2017</th>
      <th>11/2017</th>
      <th>12/2017</th>
      <th>1/2018</th>
      <th>2/2018</th>
      <th>3/2018</th>
      <th>4/2018</th>
      <th>5/2018</th>
      <th>6/2018</th>
      <th>7/2018</th>
      <th>8/2018</th>
      <th>9/2018</th>
      <th>10/2018</th>
      <th>11/2018</th>
      <th>12/2018</th>
      <th>1/2019</th>
      <th>2/2019</th>
      <th>3/2019</th>
      <th>4/2019</th>
      <th>5/2019</th>
      <th>6/2019</th>
      <th>7/2019</th>
      <th>8/2019</th>
      <th>9/2019</th>
      <th>10/2019</th>
      <th>11/2019</th>
      <th>12/2019</th>
      <th>1/2020</th>
      <th>2/2020</th>
      <th>3/2020</th>
      <th>4/2020</th>
      <th>5/2020</th>
      <th>6/2020</th>
      <th>7/2020</th>
      <th>8/2020</th>
      <th>9/2020</th>
      <th>10/2020</th>
      <th>11/2020</th>
      <th>12/2020</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cheese</td>
      <td>Candy Outer</td>
      <td>Horchata</td>
      <td>Vanilla</td>
      <td>Amethyst</td>
      <td>Perk-a-Cola</td>
      <td>12570.335165</td>
      <td>11569.168746</td>
      <td>13616.812204</td>
      <td>11884.370881</td>
      <td>13950.332334</td>
      <td>12781.156536</td>
      <td>14256.210023</td>
      <td>12887.711960</td>
      <td>15038.574006</td>
      <td>12626.489306</td>
      <td>14611.291109</td>
      <td>13194.814300</td>
      <td>14921.016216</td>
      <td>13477.391457</td>
      <td>15409.211080</td>
      <td>13999.215069</td>
      <td>15597.436976</td>
      <td>14098.124978</td>
      <td>15596.818092</td>
      <td>14941.694032</td>
      <td>15715.347212</td>
      <td>14181.212142</td>
      <td>16282.098006</td>
      <td>14650.929410</td>
      <td>16433.209008</td>
      <td>15400.579034</td>
      <td>16756.981263</td>
      <td>15128.148250</td>
      <td>17523.979943</td>
      <td>15413.044691</td>
      <td>16366.264377</td>
      <td>14568.470959</td>
      <td>16901.111542</td>
      <td>14659.021365</td>
      <td>...</td>
      <td>20736.279239</td>
      <td>18617.387585</td>
      <td>20783.711234</td>
      <td>17470.755865</td>
      <td>20523.579840</td>
      <td>18796.936906</td>
      <td>20028.582493</td>
      <td>18677.535295</td>
      <td>20048.107422</td>
      <td>18929.248617</td>
      <td>20571.155902</td>
      <td>18207.204656</td>
      <td>20839.042892</td>
      <td>18966.532984</td>
      <td>20909.977545</td>
      <td>18589.807152</td>
      <td>21287.370123</td>
      <td>17987.976867</td>
      <td>21111.062685</td>
      <td>18538.311321</td>
      <td>21797.267132</td>
      <td>18935.352772</td>
      <td>21331.378420</td>
      <td>18783.759611</td>
      <td>22139.123373</td>
      <td>18553.797271</td>
      <td>21579.506284</td>
      <td>19726.433111</td>
      <td>21147.624131</td>
      <td>19232.360491</td>
      <td>21575.521051</td>
      <td>18856.178110</td>
      <td>20701.250676</td>
      <td>19406.448560</td>
      <td>22328.687163</td>
      <td>19384.824042</td>
      <td>21449.154890</td>
      <td>19554.405590</td>
      <td>21873.104938</td>
      <td>19572.860127</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tiramisu</td>
      <td>Chocolate Outer</td>
      <td>Irish Cream</td>
      <td>Egg Nog</td>
      <td>Slate</td>
      <td>Dandy's Candies</td>
      <td>7922.970436</td>
      <td>6464.558625</td>
      <td>6616.092291</td>
      <td>8244.991928</td>
      <td>6602.132649</td>
      <td>7032.700478</td>
      <td>8437.517865</td>
      <td>6919.862786</td>
      <td>7003.449554</td>
      <td>8516.767749</td>
      <td>7541.471510</td>
      <td>7145.880001</td>
      <td>8821.556334</td>
      <td>7325.240199</td>
      <td>7618.246523</td>
      <td>9385.832260</td>
      <td>7705.860411</td>
      <td>7709.843383</td>
      <td>9471.542415</td>
      <td>7791.968645</td>
      <td>8214.485181</td>
      <td>9393.875864</td>
      <td>7911.159820</td>
      <td>8181.208339</td>
      <td>9750.400651</td>
      <td>8084.792749</td>
      <td>8234.370603</td>
      <td>9777.179731</td>
      <td>8134.876166</td>
      <td>8449.426300</td>
      <td>9911.512891</td>
      <td>8686.751188</td>
      <td>8359.638696</td>
      <td>10147.391908</td>
      <td>...</td>
      <td>9709.721305</td>
      <td>12367.917469</td>
      <td>10109.247976</td>
      <td>10597.517266</td>
      <td>12723.546352</td>
      <td>10535.166596</td>
      <td>10043.352958</td>
      <td>12434.413543</td>
      <td>10594.590384</td>
      <td>10473.165191</td>
      <td>12323.923036</td>
      <td>10232.771159</td>
      <td>9973.322947</td>
      <td>12450.632426</td>
      <td>10247.199668</td>
      <td>10310.557749</td>
      <td>12527.881699</td>
      <td>10288.118368</td>
      <td>10792.495418</td>
      <td>12538.911064</td>
      <td>10564.257282</td>
      <td>10672.337286</td>
      <td>12442.348062</td>
      <td>10975.342816</td>
      <td>10504.218214</td>
      <td>12700.925307</td>
      <td>10853.622645</td>
      <td>10917.981718</td>
      <td>13005.963533</td>
      <td>10610.202654</td>
      <td>10145.394106</td>
      <td>13132.925131</td>
      <td>10821.805709</td>
      <td>10829.961838</td>
      <td>12995.340352</td>
      <td>10504.814195</td>
      <td>10617.199735</td>
      <td>13377.165673</td>
      <td>11065.835571</td>
      <td>11135.386324</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sponge</td>
      <td>Chocolate Outer</td>
      <td>Ginger Ale</td>
      <td>Apple</td>
      <td>Slate</td>
      <td>Dandy's Candies</td>
      <td>10521.306722</td>
      <td>5543.335645</td>
      <td>5294.892374</td>
      <td>11010.452413</td>
      <td>5267.190367</td>
      <td>5546.045669</td>
      <td>11394.362620</td>
      <td>5712.245098</td>
      <td>5798.349463</td>
      <td>11781.306993</td>
      <td>5918.299339</td>
      <td>5892.693500</td>
      <td>12298.538324</td>
      <td>6260.141878</td>
      <td>6244.742736</td>
      <td>12336.799353</td>
      <td>6201.024217</td>
      <td>6638.056331</td>
      <td>12661.775736</td>
      <td>6545.099808</td>
      <td>6536.149679</td>
      <td>12757.183923</td>
      <td>6717.449248</td>
      <td>6473.324997</td>
      <td>13467.205626</td>
      <td>6806.690857</td>
      <td>7052.340323</td>
      <td>13488.253357</td>
      <td>6613.263036</td>
      <td>7017.902839</td>
      <td>13676.458530</td>
      <td>6925.928151</td>
      <td>6931.425252</td>
      <td>13418.564207</td>
      <td>...</td>
      <td>8486.837094</td>
      <td>16402.754127</td>
      <td>8037.467789</td>
      <td>8184.225799</td>
      <td>16627.369839</td>
      <td>8287.853805</td>
      <td>8307.474776</td>
      <td>16869.445360</td>
      <td>8090.179106</td>
      <td>8258.503638</td>
      <td>16961.500038</td>
      <td>8261.880924</td>
      <td>8221.327715</td>
      <td>16468.858014</td>
      <td>8243.277819</td>
      <td>8381.807681</td>
      <td>16778.471019</td>
      <td>8673.021737</td>
      <td>8552.964237</td>
      <td>16902.958241</td>
      <td>8346.914162</td>
      <td>8431.816880</td>
      <td>17088.184916</td>
      <td>8928.738286</td>
      <td>8511.985934</td>
      <td>17273.307554</td>
      <td>8242.173578</td>
      <td>8755.268429</td>
      <td>16961.225344</td>
      <td>8686.594959</td>
      <td>8516.098910</td>
      <td>17498.911792</td>
      <td>8369.846849</td>
      <td>8334.206937</td>
      <td>17519.678690</td>
      <td>8595.378915</td>
      <td>8909.348040</td>
      <td>17234.636475</td>
      <td>9002.216839</td>
      <td>8794.467252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cheese</td>
      <td>Chocolate Outer</td>
      <td>Coffee</td>
      <td>Pear</td>
      <td>Opal</td>
      <td>Dandy's Candies</td>
      <td>4739.122200</td>
      <td>2733.281035</td>
      <td>4984.394797</td>
      <td>2750.709519</td>
      <td>5274.473185</td>
      <td>2737.736109</td>
      <td>5236.191952</td>
      <td>2807.504142</td>
      <td>5581.285441</td>
      <td>2500.882597</td>
      <td>5635.195267</td>
      <td>3035.782263</td>
      <td>5492.449149</td>
      <td>2987.850135</td>
      <td>6021.641513</td>
      <td>3141.406171</td>
      <td>5884.424448</td>
      <td>2898.492508</td>
      <td>5925.633348</td>
      <td>2990.291270</td>
      <td>6055.228068</td>
      <td>3587.204540</td>
      <td>6138.483396</td>
      <td>2997.648037</td>
      <td>6370.293979</td>
      <td>3442.123255</td>
      <td>6187.807429</td>
      <td>3140.255766</td>
      <td>6243.439310</td>
      <td>3568.780726</td>
      <td>6393.113315</td>
      <td>3246.908338</td>
      <td>6540.176631</td>
      <td>3383.552285</td>
      <td>...</td>
      <td>7756.552767</td>
      <td>4233.577541</td>
      <td>8119.681843</td>
      <td>4046.102552</td>
      <td>7738.053080</td>
      <td>4401.610595</td>
      <td>7675.782975</td>
      <td>4248.467389</td>
      <td>7835.989038</td>
      <td>3784.924538</td>
      <td>7778.705078</td>
      <td>4023.523503</td>
      <td>7996.384938</td>
      <td>4236.811930</td>
      <td>7780.903630</td>
      <td>4324.838890</td>
      <td>8706.750222</td>
      <td>4592.025934</td>
      <td>7864.924599</td>
      <td>3831.383360</td>
      <td>8121.771691</td>
      <td>4428.045665</td>
      <td>8216.481440</td>
      <td>4193.238269</td>
      <td>8316.693448</td>
      <td>4591.368108</td>
      <td>8184.213024</td>
      <td>4112.875085</td>
      <td>8342.099081</td>
      <td>4074.668047</td>
      <td>8093.541144</td>
      <td>4301.081977</td>
      <td>8235.616589</td>
      <td>4151.474242</td>
      <td>8213.665500</td>
      <td>4008.885583</td>
      <td>7912.641813</td>
      <td>4275.162782</td>
      <td>8031.227879</td>
      <td>4628.989194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Chiffon</td>
      <td>Jelly Filled</td>
      <td>Butter Toffee</td>
      <td>Apricot</td>
      <td>Olive</td>
      <td>Slugworth</td>
      <td>2756.891961</td>
      <td>1739.900797</td>
      <td>1791.975108</td>
      <td>1533.023665</td>
      <td>1735.868123</td>
      <td>1824.082183</td>
      <td>2637.470462</td>
      <td>1707.745100</td>
      <td>1621.994676</td>
      <td>1814.318681</td>
      <td>1669.867919</td>
      <td>1681.547977</td>
      <td>2830.947230</td>
      <td>1732.084718</td>
      <td>1739.759848</td>
      <td>1647.585142</td>
      <td>1750.293269</td>
      <td>1710.317714</td>
      <td>2805.902600</td>
      <td>1751.767883</td>
      <td>2009.631273</td>
      <td>1937.539054</td>
      <td>1735.097831</td>
      <td>1839.490125</td>
      <td>2799.159990</td>
      <td>1566.971234</td>
      <td>1752.097294</td>
      <td>1573.071123</td>
      <td>1760.829823</td>
      <td>1795.254784</td>
      <td>2754.540709</td>
      <td>1886.366523</td>
      <td>1807.204970</td>
      <td>1740.528223</td>
      <td>...</td>
      <td>1842.142055</td>
      <td>1500.057312</td>
      <td>1731.064289</td>
      <td>1675.758204</td>
      <td>2904.893170</td>
      <td>1789.992333</td>
      <td>1810.125486</td>
      <td>1795.379685</td>
      <td>1907.771505</td>
      <td>1973.079708</td>
      <td>2865.869167</td>
      <td>1574.850737</td>
      <td>1783.459110</td>
      <td>1787.164751</td>
      <td>1752.689655</td>
      <td>1734.478146</td>
      <td>2804.497744</td>
      <td>1956.641539</td>
      <td>1909.752412</td>
      <td>1693.251443</td>
      <td>1748.211959</td>
      <td>1842.843637</td>
      <td>2757.454985</td>
      <td>1674.184059</td>
      <td>1698.962332</td>
      <td>1631.735285</td>
      <td>1769.115620</td>
      <td>1663.851403</td>
      <td>2919.917902</td>
      <td>1830.186857</td>
      <td>1864.015449</td>
      <td>1800.566323</td>
      <td>1625.130275</td>
      <td>1908.316219</td>
      <td>2696.631511</td>
      <td>1859.017636</td>
      <td>1690.042699</td>
      <td>1764.410866</td>
      <td>1909.608709</td>
      <td>1711.780317</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 138 columns</p>
</div>




```python
data = pd.DataFrame(orders.loc[0, time_cols])
data = data.reset_index()
data.columns = ['Date', 'KG']
data
px.scatter(data, x='Date',  y='KG')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="9499d748-9bea-476f-9026-a591288a6ec7" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("9499d748-9bea-476f-9026-a591288a6ec7")) {
                    Plotly.newPlot(
                        '9499d748-9bea-476f-9026-a591288a6ec7',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "Date=%{x}<br>KG=%{y}", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "showlegend": false, "type": "scatter", "x": ["1/2010", "2/2010", "3/2010", "4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [12570.33516482565, 11569.168746227244, 13616.8122044598, 11884.3708810225, 13950.332334409884, 12781.156535682429, 14256.210023357236, 12887.711959877463, 15038.574005789536, 12626.48930557771, 14611.291109090684, 13194.81429999148, 14921.016215576235, 13477.391456909943, 15409.211079596587, 13999.215068692507, 15597.436975845374, 14098.12497823274, 15596.818092478728, 14941.69403166363, 15715.347212025836, 14181.212141927937, 16282.0980055455, 14650.929410064904, 16433.20900828632, 15400.579033515967, 16756.981262857273, 15128.148250492244, 17523.979943307248, 15413.0446914734, 16366.264377017458, 14568.470958551738, 16901.11154186154, 14659.021365286097, 16494.903960781197, 15398.721298130027, 17938.090871773184, 15850.35787113158, 18236.778754419982, 15956.750789202086, 17401.696472111977, 15890.103219350918, 17283.79073343649, 16302.509223010222, 17229.645014787257, 16223.309276278227, 17796.223621100053, 16344.001270241426, 17782.006164552513, 16326.588260101846, 18253.569321985724, 16818.123129181142, 18554.33980878632, 16900.704327264033, 18479.00603218699, 17042.963875823145, 18287.35559715585, 17244.887842050513, 18822.494484753846, 17603.725932131478, 18766.104076650663, 17170.126490680243, 19632.147600450644, 16856.921979192426, 18854.690380403008, 17880.884218985302, 19087.480847049384, 18196.112254637806, 19770.963054596545, 16488.739325030063, 19699.01989730995, 17194.707087425755, 19372.657901571318, 17715.24432224015, 19227.53144133251, 17691.136252909622, 20114.534506297117, 17926.25260490304, 19880.02532889845, 16690.02893115867, 19928.02694695529, 18553.766165315024, 20547.154033981024, 17301.11715078875, 19538.97650435099, 17902.44835514176, 21269.577926886348, 18842.69654955895, 20095.445399491346, 17670.300576591326, 20310.884287446843, 18754.84178182952, 20736.279238797022, 18617.387584546323, 20783.71123390676, 17470.755864944782, 20523.579839792714, 18796.93690580505, 20028.582492587037, 18677.535295190337, 20048.1074217522, 18929.24861718753, 20571.15590247796, 18207.20465623173, 20839.04289237627, 18966.53298378622, 20909.977545252816, 18589.807151786372, 21287.370122673103, 17987.976866769444, 21111.062684974826, 18538.311320658097, 21797.267132392342, 18935.35277235507, 21331.37841983855, 18783.75961074272, 22139.12337340894, 18553.79727063604, 21579.50628438568, 19726.43311123112, 21147.624131226225, 19232.360491469408, 21575.52105110441, 18856.1781102771, 20701.25067582265, 19406.448559709923, 22328.68716294986, 19384.824041986754, 21449.154889830093, 19554.40558950196, 21873.104938389297, 19572.860127015803], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "KG"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('9499d748-9bea-476f-9026-a591288a6ec7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>



```python
fig, ax = plt.subplots(1,1,figsize=(10,10))
pd.plotting.autocorrelation_plot(data['KG'], ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fb1b29f3610>




    
![png](S5_Time_Series_Analysis_files/S5_Time_Series_Analysis_13_1.png)
    


Normally with time series data, we'd want to try a host of preprocessing techniques and remove the trend (really create two separate analyses, one of the trend and one of the seasonality) but to keep things simple and to showcase the utility of machine learning, we are going to deviate from the stats-like approach and work with our data as is. 

For more details on the stats-like models you can perform a cursory search on _ARIMA_, _ARMA_, _SARIMA_

<a name='x.4'></a>

## 5.4 Modeling

[back to top](#top)


```python
from tensorflow import keras
from tensorflow.keras import layers
```

<a name='x.4.1'></a>

### 5.4.1 Sweeping (Rolling) Window

[back to top](#top)

We're going to revist this idea of a sweeping window from our feature engineering disucssion. It turns out, even though we are using a NN, there is still some preprocessing we need to do. In our case, each time delta is represented by a month. So we will choose some number of months to include in our feature set, this will in turn determine what our overall training data will look like. 

<p align=center>
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/11/3hotmk.gif"></img>
</p>


```python
Xy = orders.loc[[0], time_cols].values
# separate the non-time series columns
X_cat = Xy[:,:-120]

# select the columns to apply the sweeping window
X = Xy[:,-120:]
```

with a window size of 3, our X will have 3 features, the prior 3 months leading up to the month for which we will attempt to forecast. 


```python
window_size = 3
X, y, labels = process_data(orders.loc[[0], time_cols].values, window=window_size, time_cols=132)
X[:5]
```




    array([[12570.33516483, 11569.16874623, 13616.81220446],
           [11569.16874623, 13616.81220446, 11884.37088102],
           [13616.81220446, 11884.37088102, 13950.33233441],
           [11884.37088102, 13950.33233441, 12781.15653568],
           [13950.33233441, 12781.15653568, 14256.21002336]])



With a window size of 1, our X data will have a feature size of 1


```python
window_size = 1
X, y, labels = process_data(orders.loc[[0], time_cols].values, window=window_size, time_cols=132)
X[:5]
```




    array([[12570.33516483],
           [11569.16874623],
           [13616.81220446],
           [11884.37088102],
           [13950.33233441]])



and so on.

<a name='x.4.2'></a>

### 5.4.2 FFNN

[back to top](#top)

I'm going to start with a very simple FFNN model:


```python
model = keras.Sequential([
    layers.Dense(8, input_shape=[window_size]), # one layer, 8 nodes
    layers.Dense(1) # single output for the kg
])

model.compile(loss='mean_squared_error', optimizer='adam')
```


```python
early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='loss'
)
```


```python
history = model.fit(
    X, y,
    batch_size=10,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
```


```python
history_df = pd.DataFrame(history.history)
history_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>4250741.5</td>
    </tr>
    <tr>
      <th>24</th>
      <td>4245127.5</td>
    </tr>
    <tr>
      <th>25</th>
      <td>4246370.5</td>
    </tr>
    <tr>
      <th>26</th>
      <td>4248100.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>4242104.0</td>
    </tr>
  </tbody>
</table>
</div>



As we can see from the `y` vs `y_pred` the FFNN is just predicting the previous month's value:


```python
y_pred = model.predict(X)
pd.DataFrame(y_pred, y)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11569.168746</th>
      <td>12474.020508</td>
    </tr>
    <tr>
      <th>13616.812204</th>
      <td>11480.559570</td>
    </tr>
    <tr>
      <th>11884.370881</th>
      <td>13512.444336</td>
    </tr>
    <tr>
      <th>13950.332334</th>
      <td>11793.336914</td>
    </tr>
    <tr>
      <th>12781.156536</th>
      <td>13843.397461</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>19384.824042</th>
      <td>22157.271484</td>
    </tr>
    <tr>
      <th>21449.154890</th>
      <td>19236.064453</td>
    </tr>
    <tr>
      <th>19554.405590</th>
      <td>21284.505859</td>
    </tr>
    <tr>
      <th>21873.104938</th>
      <td>19404.339844</td>
    </tr>
    <tr>
      <th>19572.860127</th>
      <td>21705.197266</td>
    </tr>
  </tbody>
</table>
<p>131 rows × 1 columns</p>
</div>



We can try this with a more suitable window size


```python
window_size = 3
X, y, labels = process_data(orders.loc[[0], time_cols].values, window=window_size, time_cols=132)

model = keras.Sequential([
    # layers.Dense(8, input_shape=[window_size]),
    layers.Dense(1, input_shape=[window_size])
])

model.compile(loss='mean_squared_error', optimizer='adam')
```


```python
history = model.fit(
    X, y,
    batch_size=10,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
```


```python
history_df = pd.DataFrame(history.history)
history_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>703</th>
      <td>514088.96875</td>
    </tr>
    <tr>
      <th>704</th>
      <td>513142.15625</td>
    </tr>
    <tr>
      <th>705</th>
      <td>507798.15625</td>
    </tr>
    <tr>
      <th>706</th>
      <td>511337.62500</td>
    </tr>
    <tr>
      <th>707</th>
      <td>513890.40625</td>
    </tr>
  </tbody>
</table>
</div>



A cursory glance looks like our values are closer together


```python
results = melt_results(model, X, y)
```


```python
px.line(results, x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="d52a6cbd-a684-4ccd-bc70-41064f9370ef" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("d52a6cbd-a684-4ccd-bc70-41064f9370ef")) {
                    Plotly.newPlot(
                        'd52a6cbd-a684-4ccd-bc70-41064f9370ef',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=real<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=real", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Source=real", "showlegend": true, "type": "scatter", "x": ["4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [11884.3708810225, 13950.332334409884, 12781.156535682429, 14256.210023357236, 12887.711959877463, 15038.574005789536, 12626.48930557771, 14611.291109090684, 13194.81429999148, 14921.016215576235, 13477.391456909943, 15409.211079596587, 13999.215068692507, 15597.436975845374, 14098.12497823274, 15596.818092478728, 14941.69403166363, 15715.347212025836, 14181.212141927937, 16282.0980055455, 14650.929410064904, 16433.20900828632, 15400.579033515967, 16756.981262857273, 15128.148250492244, 17523.979943307248, 15413.0446914734, 16366.264377017458, 14568.470958551738, 16901.11154186154, 14659.021365286097, 16494.903960781197, 15398.721298130027, 17938.090871773184, 15850.35787113158, 18236.778754419982, 15956.750789202086, 17401.696472111977, 15890.103219350918, 17283.79073343649, 16302.509223010222, 17229.645014787257, 16223.309276278227, 17796.223621100053, 16344.001270241426, 17782.006164552513, 16326.588260101846, 18253.569321985724, 16818.123129181142, 18554.33980878632, 16900.704327264033, 18479.00603218699, 17042.963875823145, 18287.35559715585, 17244.887842050513, 18822.494484753846, 17603.725932131478, 18766.104076650663, 17170.126490680243, 19632.147600450644, 16856.921979192426, 18854.690380403008, 17880.884218985302, 19087.480847049384, 18196.112254637806, 19770.963054596545, 16488.739325030063, 19699.01989730995, 17194.707087425755, 19372.657901571318, 17715.24432224015, 19227.53144133251, 17691.136252909622, 20114.534506297117, 17926.25260490304, 19880.02532889845, 16690.02893115867, 19928.02694695529, 18553.766165315024, 20547.154033981024, 17301.11715078875, 19538.97650435099, 17902.44835514176, 21269.577926886348, 18842.69654955895, 20095.445399491346, 17670.300576591326, 20310.884287446843, 18754.84178182952, 20736.279238797022, 18617.387584546323, 20783.71123390676, 17470.755864944782, 20523.579839792714, 18796.93690580505, 20028.582492587037, 18677.535295190337, 20048.1074217522, 18929.24861718753, 20571.15590247796, 18207.20465623173, 20839.04289237627, 18966.53298378622, 20909.977545252816, 18589.807151786372, 21287.370122673103, 17987.976866769444, 21111.062684974826, 18538.311320658097, 21797.267132392342, 18935.35277235507, 21331.37841983855, 18783.75961074272, 22139.12337340894, 18553.79727063604, 21579.50628438568, 19726.43311123112, 21147.624131226225, 19232.360491469408, 21575.52105110441, 18856.1781102771, 20701.25067582265, 19406.448559709923, 22328.68716294986, 19384.824041986754, 21449.154889830093, 19554.40558950196, 21873.104938389297, 19572.860127015803], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=predicted<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=predicted", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Source=predicted", "showlegend": true, "type": "scatter", "x": ["4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [12355.0859375, 13640.7431640625, 12307.1044921875, 14332.3984375, 13131.0283203125, 14212.7783203125, 13552.763671875, 14700.265625, 12652.4365234375, 14801.03515625, 13574.015625, 14962.173828125, 13970.4462890625, 15578.1806640625, 14305.185546875, 15540.5673828125, 14297.0986328125, 16009.984375, 15129.0732421875, 15213.9912109375, 14734.306640625, 16403.990234375, 14958.9521484375, 16761.474609375, 15755.279296875, 16499.529296875, 15817.58203125, 17504.912109375, 14965.267578125, 15796.3193359375, 15130.4443359375, 16766.73828125, 14686.5009765625, 16811.646484375, 16451.060546875, 18008.30078125, 16300.5703125, 18110.662109375, 15727.3662109375, 17264.041015625, 16023.4013671875, 17446.83203125, 16421.3203125, 17137.51953125, 16727.669921875, 17762.119140625, 16536.51953125, 17676.62109375, 16819.26953125, 18412.998046875, 17204.791015625, 18482.048828125, 17078.1171875, 18459.734375, 17126.396484375, 18339.029296875, 17736.703125, 18938.0390625, 17749.787109375, 18435.4296875, 17923.501953125, 19243.39453125, 16717.30078125, 19337.271484375, 18180.85546875, 19215.458984375, 18766.697265625, 18614.162109375, 16836.208984375, 19861.880859375, 17306.4375, 19527.3203125, 17852.443359375, 19114.693359375, 18452.845703125, 20069.751953125, 18064.07421875, 18975.03515625, 17102.572265625, 20803.421875, 19135.88671875, 19629.9375, 17069.33984375, 19737.794921875, 19195.9765625, 21565.546875, 18428.048828125, 19303.3515625, 18109.71875, 20768.50390625, 19236.8671875, 20509.314453125, 18924.150390625, 19914.69140625, 17708.232421875, 21088.181640625, 18728.63671875, 19888.515625, 18887.47265625, 20122.76171875, 19426.662109375, 20018.087890625, 18674.275390625, 21098.416015625, 19263.26953125, 20539.78125, 19121.091796875, 20701.00390625, 18277.365234375, 21191.203125, 19286.3203125, 21770.748046875, 19003.40234375, 21054.41796875, 19604.93359375, 21716.5625, 18637.990234375, 22055.3515625, 19713.060546875, 20759.974609375, 19754.70703125, 21166.15234375, 18657.068359375, 20912.896484375, 20605.359375, 22080.318359375, 19207.404296875, 21405.28515625, 20073.166015625], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "KG"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('d52a6cbd-a684-4ccd-bc70-41064f9370ef');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


<a name='x.4.2.1'></a>

#### 🏋️ Exercise-Discussion 1: Varify that the model is linear

[back to top](#top)

We're having to change our way of thinking here with time series analysis. Recall that a model without an activation function can only encapsulate linear relationships. How come we can see non-linear relationships in our time series plot above? make a plot that showcases we are indeed still within a linear world.

This is an open ended question and I myself don't have the _best_ answer. Think about how you would attempt to show linearity of the model. (On Monday our model was only in 2D, and it was a binary classification task, so it was easier to view the decision boundaries and verify linearity).


```python
# Code cell for Exercise 1
```

<a name='x.4.2.2'></a>

#### 🏋️ Exercise 2: Vary model architecture and window size

[back to top](#top)

Create these three different models. Train on the whole dataset with a window size of 3. record the training loss for the last 5 epochs of each model

```
models = [
          keras.Sequential([
    layers.Dense(8, input_shape=[window_size]),
    layers.Dense(1)
]),
keras.Sequential([
    layers.Dense(8, activation='relu', input_shape=[window_size]),
    layers.Dense(1)
]),
keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[window_size]),
    layers.Dense(1)
])]
```

You can create the training sets with:

```
window_size = 3
X, y, labels = process_data(orders.loc[[0], time_cols].values, window=window_size, time_cols=132)
```

Use a batch size of 10 when training.

When you are finished training a model use `melt_results` and plotly to make a graph of your predictions vs actuals

```
df = melt_results(model, X, y)
px.line(df, x='Date', y='KG', color='Source')
```

You can use the same early_stopping and fit formula from 6.4.2


```python
# Code cell for exercise 2
```


```python
px.line(df, x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="15dad771-d18e-4fff-8d11-983538b1753d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("15dad771-d18e-4fff-8d11-983538b1753d")) {
                    Plotly.newPlot(
                        '15dad771-d18e-4fff-8d11-983538b1753d',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=real<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=real", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Source=real", "showlegend": true, "type": "scatter", "x": ["4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [11884.3708810225, 13950.332334409884, 12781.156535682429, 14256.210023357236, 12887.711959877463, 15038.574005789536, 12626.48930557771, 14611.291109090684, 13194.81429999148, 14921.016215576235, 13477.391456909943, 15409.211079596587, 13999.215068692507, 15597.436975845374, 14098.12497823274, 15596.818092478728, 14941.69403166363, 15715.347212025836, 14181.212141927937, 16282.0980055455, 14650.929410064904, 16433.20900828632, 15400.579033515967, 16756.981262857273, 15128.148250492244, 17523.979943307248, 15413.0446914734, 16366.264377017458, 14568.470958551738, 16901.11154186154, 14659.021365286097, 16494.903960781197, 15398.721298130027, 17938.090871773184, 15850.35787113158, 18236.778754419982, 15956.750789202086, 17401.696472111977, 15890.103219350918, 17283.79073343649, 16302.509223010222, 17229.645014787257, 16223.309276278227, 17796.223621100053, 16344.001270241426, 17782.006164552513, 16326.588260101846, 18253.569321985724, 16818.123129181142, 18554.33980878632, 16900.704327264033, 18479.00603218699, 17042.963875823145, 18287.35559715585, 17244.887842050513, 18822.494484753846, 17603.725932131478, 18766.104076650663, 17170.126490680243, 19632.147600450644, 16856.921979192426, 18854.690380403008, 17880.884218985302, 19087.480847049384, 18196.112254637806, 19770.963054596545, 16488.739325030063, 19699.01989730995, 17194.707087425755, 19372.657901571318, 17715.24432224015, 19227.53144133251, 17691.136252909622, 20114.534506297117, 17926.25260490304, 19880.02532889845, 16690.02893115867, 19928.02694695529, 18553.766165315024, 20547.154033981024, 17301.11715078875, 19538.97650435099, 17902.44835514176, 21269.577926886348, 18842.69654955895, 20095.445399491346, 17670.300576591326, 20310.884287446843, 18754.84178182952, 20736.279238797022, 18617.387584546323, 20783.71123390676, 17470.755864944782, 20523.579839792714, 18796.93690580505, 20028.582492587037, 18677.535295190337, 20048.1074217522, 18929.24861718753, 20571.15590247796, 18207.20465623173, 20839.04289237627, 18966.53298378622, 20909.977545252816, 18589.807151786372, 21287.370122673103, 17987.976866769444, 21111.062684974826, 18538.311320658097, 21797.267132392342, 18935.35277235507, 21331.37841983855, 18783.75961074272, 22139.12337340894, 18553.79727063604, 21579.50628438568, 19726.43311123112, 21147.624131226225, 19232.360491469408, 21575.52105110441, 18856.1781102771, 20701.25067582265, 19406.448559709923, 22328.68716294986, 19384.824041986754, 21449.154889830093, 19554.40558950196, 21873.104938389297, 19572.860127015803], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=predicted<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=predicted", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Source=predicted", "showlegend": true, "type": "scatter", "x": ["4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [11825.6875, 13559.1416015625, 12137.75390625, 13947.8779296875, 12989.833984375, 14231.40625, 13158.73828125, 14926.837890625, 12870.529296875, 14588.9521484375, 13427.3798828125, 14895.6435546875, 13730.8017578125, 15391.9677734375, 14224.8291015625, 15569.8564453125, 14314.1650390625, 15647.681640625, 15102.7060546875, 15677.5302734375, 14454.068359375, 16251.0849609375, 14895.8544921875, 16456.4453125, 15615.7265625, 16722.203125, 15433.7666015625, 17458.36328125, 15580.2919921875, 16309.3955078125, 14863.07421875, 16818.68359375, 14903.3330078125, 16513.052734375, 15724.498046875, 17878.58203125, 16155.3134765625, 18159.625, 16172.05078125, 17382.794921875, 16107.341796875, 17313.5390625, 16483.404296875, 17252.341796875, 16464.197265625, 17786.560546875, 16568.794921875, 17770.6953125, 16597.220703125, 18251.64453125, 17074.072265625, 18531.99609375, 17140.28515625, 18475.177734375, 17254.068359375, 18316.220703125, 17492.2578125, 18841.234375, 17812.646484375, 18745.146484375, 17495.0234375, 19517.84765625, 17125.201171875, 18900.5859375, 18097.98046875, 19135.197265625, 18450.5703125, 19601.57421875, 16863.998046875, 19617.59765625, 17484.56640625, 19359.19140625, 17953.666015625, 19218.322265625, 18016.2109375, 20058.076171875, 18202.548828125, 19723.52734375, 17070.009765625, 19954.546875, 18845.3046875, 20389.919921875, 17590.349609375, 19529.072265625, 18316.64453125, 21206.703125, 19056.572265625, 20005.666015625, 18007.3671875, 20317.125, 19044.81640625, 20686.09765625, 18918.6328125, 20623.279296875, 17837.052734375, 20518.95703125, 19015.080078125, 20039.18359375, 18911.501953125, 20081.958984375, 19192.466796875, 20493.732421875, 18547.373046875, 20818.744140625, 19251.302734375, 20841.6015625, 18938.953125, 21136.314453125, 18364.234375, 21031.203125, 18937.564453125, 21695.7734375, 19253.24609375, 21248.470703125, 19193.845703125, 21972.640625, 18921.76171875, 21569.33203125, 19967.01953125, 21113.927734375, 19555.97265625, 21477.498046875, 19123.1015625, 20727.06640625, 19791.390625, 22219.658203125, 19673.5703125, 21425.2734375, 19877.919921875], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "KG"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('15dad771-d18e-4fff-8d11-983538b1753d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


<a name='x.4.3'></a>

### 5.4.3 LSTM NN

[back to top](#top)

Our data preparation for the LSTM NN includes time steps. The parameter `input_dim` tells our `LSTM` block how man time steps we have in the input data. This is a reframing (and a more appropriate reframing) of the same problem. The LSTM model is viewing the input feature w/ multiple time steps as a single feature at different times, rather than separate features. We could, for instance, have a second dimension that includes non-time related information, such as the customer name or truffle types (or other featurse that also vary through time, multiple feed rates or T/P, etc).


```python
window_size = 6
batch_size = 10

X, y, labels = process_data(orders.loc[[0], time_cols].values, window=window_size, time_cols=132)

X = X.reshape(-1, 1, window_size)
y = y.reshape(-1, 1, 1)

model = keras.Sequential([
    layers.LSTM(8, activation='relu', input_dim=window_size),
    layers.Dense(8),
    layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
```


```python
history = model.fit(
    X, y,
    batch_size=batch_size,
    epochs=1000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
```


```python
history_df = pd.DataFrame(history.history)
history_df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>411</th>
      <td>322288.40625</td>
    </tr>
    <tr>
      <th>412</th>
      <td>361232.25000</td>
    </tr>
    <tr>
      <th>413</th>
      <td>336341.78125</td>
    </tr>
    <tr>
      <th>414</th>
      <td>326199.90625</td>
    </tr>
    <tr>
      <th>415</th>
      <td>320201.21875</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = melt_results(model, X, y.flatten())
```


```python
px.line(results, x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="60577dfc-eefb-4e74-806a-184d8dbf0f50" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("60577dfc-eefb-4e74-806a-184d8dbf0f50")) {
                    Plotly.newPlot(
                        '60577dfc-eefb-4e74-806a-184d8dbf0f50',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=real<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=real", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Source=real", "showlegend": true, "type": "scatter", "x": ["7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [14256.210023357236, 12887.711959877463, 15038.574005789536, 12626.48930557771, 14611.291109090684, 13194.81429999148, 14921.016215576235, 13477.391456909943, 15409.211079596587, 13999.215068692507, 15597.436975845374, 14098.12497823274, 15596.818092478728, 14941.69403166363, 15715.347212025836, 14181.212141927937, 16282.0980055455, 14650.929410064904, 16433.20900828632, 15400.579033515967, 16756.981262857273, 15128.148250492244, 17523.979943307248, 15413.0446914734, 16366.264377017458, 14568.470958551738, 16901.11154186154, 14659.021365286097, 16494.903960781197, 15398.721298130027, 17938.090871773184, 15850.35787113158, 18236.778754419982, 15956.750789202086, 17401.696472111977, 15890.103219350918, 17283.79073343649, 16302.509223010222, 17229.645014787257, 16223.309276278227, 17796.223621100053, 16344.001270241426, 17782.006164552513, 16326.588260101846, 18253.569321985724, 16818.123129181142, 18554.33980878632, 16900.704327264033, 18479.00603218699, 17042.963875823145, 18287.35559715585, 17244.887842050513, 18822.494484753846, 17603.725932131478, 18766.104076650663, 17170.126490680243, 19632.147600450644, 16856.921979192426, 18854.690380403008, 17880.884218985302, 19087.480847049384, 18196.112254637806, 19770.963054596545, 16488.739325030063, 19699.01989730995, 17194.707087425755, 19372.657901571318, 17715.24432224015, 19227.53144133251, 17691.136252909622, 20114.534506297117, 17926.25260490304, 19880.02532889845, 16690.02893115867, 19928.02694695529, 18553.766165315024, 20547.154033981024, 17301.11715078875, 19538.97650435099, 17902.44835514176, 21269.577926886348, 18842.69654955895, 20095.445399491346, 17670.300576591326, 20310.884287446843, 18754.84178182952, 20736.279238797022, 18617.387584546323, 20783.71123390676, 17470.755864944782, 20523.579839792714, 18796.93690580505, 20028.582492587037, 18677.535295190337, 20048.1074217522, 18929.24861718753, 20571.15590247796, 18207.20465623173, 20839.04289237627, 18966.53298378622, 20909.977545252816, 18589.807151786372, 21287.370122673103, 17987.976866769444, 21111.062684974826, 18538.311320658097, 21797.267132392342, 18935.35277235507, 21331.37841983855, 18783.75961074272, 22139.12337340894, 18553.79727063604, 21579.50628438568, 19726.43311123112, 21147.624131226225, 19232.360491469408, 21575.52105110441, 18856.1781102771, 20701.25067582265, 19406.448559709923, 22328.68716294986, 19384.824041986754, 21449.154889830093, 19554.40558950196, 21873.104938389297, 19572.860127015803], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=predicted<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=predicted", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Source=predicted", "showlegend": true, "type": "scatter", "x": ["7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [13180.5244140625, 11941.052734375, 13897.1728515625, 12522.2275390625, 14309.3193359375, 13096.4521484375, 14802.939453125, 13020.638671875, 14922.9716796875, 13130.0478515625, 14894.341796875, 13569.859375, 15329.9990234375, 14020.2109375, 15622.0634765625, 14315.18359375, 15828.01953125, 14755.5458984375, 15874.12890625, 14772.2138671875, 16116.365234375, 14683.439453125, 16555.751953125, 15248.7373046875, 16811.826171875, 15624.388671875, 17435.9453125, 15604.052734375, 17160.962890625, 15268.173828125, 16771.892578125, 14809.287109375, 16804.845703125, 15244.541015625, 17437.248046875, 16029.177734375, 18312.34765625, 16288.9140625, 17983.26171875, 16235.626953125, 17556.31640625, 16354.9541015625, 17489.228515625, 16529.060546875, 17760.345703125, 16555.396484375, 17977.400390625, 16594.361328125, 18230.009765625, 16900.791015625, 18635.373046875, 17209.576171875, 18730.90234375, 17256.3203125, 18582.009765625, 17442.232421875, 18845.5390625, 17683.0703125, 19098.67578125, 17734.451171875, 19381.06640625, 17376.384765625, 19372.57421875, 17649.228515625, 19386.1328125, 18351.927734375, 19727.396484375, 17688.015625, 19858.5234375, 17253.630859375, 19738.755859375, 17737.013671875, 19527.080078125, 18018.39453125, 20046.962890625, 18187.865234375, 20121.90234375, 17573.572265625, 20144.68359375, 18101.638671875, 20523.671875, 18151.841796875, 20122.90625, 17993.490234375, 20761.791015625, 18826.2734375, 20869.994140625, 18526.685546875, 20408.02734375, 18580.732421875, 20915.5, 19076.302734375, 20948.8515625, 18448.703125, 20821.078125, 18561.212890625, 20518.23046875, 19026.8984375, 20366.970703125, 19106.486328125, 20562.0234375, 18892.654296875, 20958.59375, 18943.490234375, 21216.8046875, 19141.826171875, 21336.763671875, 18609.283203125, 21359.861328125, 18677.748046875, 21713.556640625, 19097.056640625, 21832.57421875, 19257.25390625, 21906.306640625, 19140.806640625, 22075.396484375, 19541.294921875, 21693.501953125, 19897.388671875, 21610.060546875, 19310.5859375, 21347.126953125, 19513.626953125, 21818.173828125, 19804.263671875], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "KG"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('60577dfc-eefb-4e74-806a-184d8dbf0f50');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


<a name='x.3.4.1'></a>

#### 🏋️ Exercise 3: Compare LSTM with FFNN using Train/Val/Test sets and 3 Month Window

[back to top](#top)


```python
### YOUR OPT WINDOW SIZE FROM EXERCISE 2 ###
window_size = 3
batch_size = 10
patience = 50

# training on single order history
data = orders.loc[[0], time_cols].values

# describes the split train 0-.6/val .6-.8/test .8-1
train_test_val_ratios = [0.6, 0.8]

X_train, y_train, X_val, y_val, X_test, y_test = train_test_process(data,
                                                        train_test_val_ratios)


### YOUR EARLY STOPPING FORMULA ###

### YOUR MODEL FROM EX 6.3.3.2 ###

# compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=10000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
  )

print(pd.DataFrame(history.history).tail())
```

    train size: 76
    val size: 23
    test size: 24
    
                 loss     val_loss
    635  272437.62500  815874.4375
    636  274113.68750  810266.5000
    637  275288.75000  817768.3750
    638  274128.71875  800769.0625
    639  272784.25000  800416.8125


We'll then record the mse performance of the model to later compare with the LSTM


```python
results = []
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
results.append(['Dense', mse])
results
```




    [['Dense', 415929.951056031]]



We'll use the same parameters (window size, batch size, and early stopping to train the LSTM and compare the optimum FFNN architecture we previously used)


```python
X_train = X_train.reshape(-1, 1, window_size)
y_train = y_train.reshape(-1, 1, 1)
X_val = X_val.reshape(-1, 1, window_size)
y_val = y_val.reshape(-1, 1, 1)
X_test = X_test.reshape(-1, 1, window_size)
y_test = y_test.reshape(-1, 1, 1)

model = keras.Sequential([
    layers.LSTM(8, activation='relu', input_dim=window_size),
    layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=10000,
    callbacks=[early_stopping],
    verbose=0, 
  )

print(pd.DataFrame(history.history).tail())
```

                  loss     val_loss
    1286  267132.96875  767315.0625
    1287  266946.25000  770766.9375
    1288  270205.12500  748175.3750
    1289  267703.06250  749915.8125
    1290  266674.15625  758844.8125



```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
results.append(['LSTM', mse])
```

Comparison of results:


```python
pd.DataFrame(results, columns=['Model', 'Test MSE']).set_index('Model').astype(int)
```




    [['Dense', 415929.951056031], ['LSTM', 393786.0529972827]]



As a last visualization in this exercise we'll look at the trian/val/test predictions along the actual


```python
data = orders.loc[[0], time_cols].values
idx_split1 = int(data.shape[1]*train_test_val_ratios[0])
idx_split2 = int(data.shape[1]*train_test_val_ratios[1])

y_p_train = model.predict(X_train)
y_p_val = model.predict(X_val)
y_p_test = model.predict(X_test)
new = orders.loc[[0], time_cols].T.reset_index()
new.columns = ['Date', 'Real']
new['Train'] = np.nan
new.iloc[window_size:idx_split1,2] = y_p_train
new['Val'] = np.nan
new.iloc[idx_split1+window_size:idx_split2,3] = y_p_val
new['Test'] = np.nan
new.iloc[idx_split2+window_size:,4] = y_p_test
new = new.melt(id_vars='Date', var_name='Source', value_name='KG')
```


```python
px.line(new, x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>
                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>    
            <div id="36fed7ea-f858-4788-9a4e-ee6fe8319e47" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">

                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("36fed7ea-f858-4788-9a4e-ee6fe8319e47")) {
                    Plotly.newPlot(
                        '36fed7ea-f858-4788-9a4e-ee6fe8319e47',
                        [{"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=Real<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=Real", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "Source=Real", "showlegend": true, "type": "scatter", "x": ["1/2010", "2/2010", "3/2010", "4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [12570.33516482565, 11569.168746227244, 13616.8122044598, 11884.3708810225, 13950.332334409884, 12781.156535682429, 14256.210023357236, 12887.711959877463, 15038.574005789536, 12626.48930557771, 14611.291109090684, 13194.81429999148, 14921.016215576235, 13477.391456909943, 15409.211079596587, 13999.215068692507, 15597.436975845374, 14098.12497823274, 15596.818092478728, 14941.69403166363, 15715.347212025836, 14181.212141927937, 16282.0980055455, 14650.929410064904, 16433.20900828632, 15400.579033515967, 16756.981262857273, 15128.148250492244, 17523.979943307248, 15413.0446914734, 16366.264377017458, 14568.470958551738, 16901.11154186154, 14659.021365286097, 16494.903960781197, 15398.721298130027, 17938.090871773184, 15850.35787113158, 18236.778754419982, 15956.750789202086, 17401.696472111977, 15890.103219350918, 17283.79073343649, 16302.509223010222, 17229.645014787257, 16223.309276278227, 17796.223621100053, 16344.001270241426, 17782.006164552513, 16326.588260101846, 18253.569321985724, 16818.123129181142, 18554.33980878632, 16900.704327264033, 18479.00603218699, 17042.963875823145, 18287.35559715585, 17244.887842050513, 18822.494484753846, 17603.725932131478, 18766.104076650663, 17170.126490680243, 19632.147600450644, 16856.921979192426, 18854.690380403008, 17880.884218985302, 19087.480847049384, 18196.112254637806, 19770.963054596545, 16488.739325030063, 19699.01989730995, 17194.707087425755, 19372.657901571318, 17715.24432224015, 19227.53144133251, 17691.136252909622, 20114.534506297117, 17926.25260490304, 19880.02532889845, 16690.02893115867, 19928.02694695529, 18553.766165315024, 20547.154033981024, 17301.11715078875, 19538.97650435099, 17902.44835514176, 21269.577926886348, 18842.69654955895, 20095.445399491346, 17670.300576591326, 20310.884287446843, 18754.84178182952, 20736.279238797022, 18617.387584546323, 20783.71123390676, 17470.755864944782, 20523.579839792714, 18796.93690580505, 20028.582492587037, 18677.535295190337, 20048.1074217522, 18929.24861718753, 20571.15590247796, 18207.20465623173, 20839.04289237627, 18966.53298378622, 20909.977545252816, 18589.807151786372, 21287.370122673103, 17987.976866769444, 21111.062684974826, 18538.311320658097, 21797.267132392342, 18935.35277235507, 21331.37841983855, 18783.75961074272, 22139.12337340894, 18553.79727063604, 21579.50628438568, 19726.43311123112, 21147.624131226225, 19232.360491469408, 21575.52105110441, 18856.1781102771, 20701.25067582265, 19406.448559709923, 22328.68716294986, 19384.824041986754, 21449.154889830093, 19554.40558950196, 21873.104938389297, 19572.860127015803], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=Train<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=Train", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "Source=Train", "showlegend": true, "type": "scatter", "x": ["1/2010", "2/2010", "3/2010", "4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [null, null, null, 11530.80859375, 13547.0107421875, 12017.6533203125, 13778.6494140625, 12893.5078125, 14261.6748046875, 12927.7666015625, 15076.15234375, 12940.1923828125, 14507.783203125, 13324.0244140625, 14885.705078125, 13578.6259765625, 15323.1728515625, 14155.8232421875, 15607.4013671875, 14294.25, 15480.1943359375, 15074.0634765625, 15926.88671875, 14279.1435546875, 16202.197265625, 14831.9873046875, 16323.2197265625, 15521.9248046875, 16855.90234375, 15203.2568359375, 17469.328125, 15860.4794921875, 16587.314453125, 14690.3681640625, 16879.89453125, 14974.7041015625, 16384.044921875, 15324.7646484375, 17848.638671875, 16040.9189453125, 18219.875, 16360.8349609375, 17463.84375, 16121.544921875, 17262.85546875, 16494.96875, 17322.7890625, 16306.2822265625, 17820.662109375, 16557.044921875, 17838.72265625, 16453.333984375, 18195.025390625, 16977.841796875, 18581.884765625, 17140.587890625, 18504.537109375, 17291.974609375, 18320.216796875, 17343.390625, 18812.208984375, 17820.373046875, 18920.37109375, 17241.39453125, 19696.1484375, 17286.443359375, 18702.0078125, 18034.021484375, 19108.53515625, 18266.767578125, 20135.396484375, 16819.486328125, 19538.892578125, 17530.896484375, 19302.970703125, 17974.005859375, 19292.216796875, 17759.21484375, 20087.013671875, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=Val<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=Val", "line": {"color": "#00cc96", "dash": "solid"}, "mode": "lines", "name": "Source=Val", "showlegend": true, "type": "scatter", "x": ["1/2010", "2/2010", "3/2010", "4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 19562.25, 18666.6640625, 20812.396484375, 17802.416015625, 19452.72265625, 17828.015625, 21070.75, 19337.732421875, 20385.990234375, 17909.234375, 20121.26171875, 18914.283203125, 20805.439453125, 18875.658203125, 21021.876953125, 17844.279296875, 20268.6875, 19130.21875, 20132.666015625, 18896.615234375, 20078.298828125, 19047.263671875, 20762.689453125, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], "yaxis": "y"}, {"hoverlabel": {"namelength": 0}, "hovertemplate": "Source=Test<br>Date=%{x}<br>KG=%{y}", "legendgroup": "Source=Test", "line": {"color": "#ab63fa", "dash": "solid"}, "mode": "lines", "name": "Source=Test", "showlegend": true, "type": "scatter", "x": ["1/2010", "2/2010", "3/2010", "4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 21025.208984375, 18800.91015625, 21401.697265625, 18349.783203125, 20994.505859375, 18708.802734375, 21705.333984375, 19330.486328125, 21383.529296875, 18933.072265625, 22155.6640625, 19004.689453125, 21361.583984375, 20062.9765625, 21315.884765625, 19415.896484375, 21672.541015625, 19315.009765625, 20655.81640625, 19342.04296875, 22334.814453125, 19861.693359375, 21464.390625, 19739.919921875], "yaxis": "y"}],
                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "KG"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('36fed7ea-f858-4788-9a4e-ee6fe8319e47');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };

            </script>
        </div>
</body>
</html>


<a name='x.5'></a>

## 5.5 Model Extensibility

[back to top](#top)


```python
from ipywidgets import interact
```

<a name='x.5.1'></a>

### 🏋️ Exercise 4: Apply Model to Other Orders

Take the last LSTM model and apply it to other orders in the dataset. What do you notice?

[back to top](#top)


```python
def apply_model(dataset=orders.index, window_size=3):
  window_size = window_size
  data = pd.DataFrame(orders.loc[dataset, time_cols])
  data = data.reset_index()
  data.columns = ['Date', 'KG']

  

  X, y, labels = process_data(orders.loc[[dataset], 
                              time_cols].values, 
                              window=window_size, 
                              time_cols=132)

  y_pred = model.predict(X.reshape(-1, 1, window_size)).flatten()

  results = pd.DataFrame(y_pred,  y)
  results = results.reset_index()
  results.index = data['Date'][window_size:]
  results = results.reset_index()
  results.columns=['Date', 'real', 'predicted']
  results = results.melt(id_vars='Date', var_name='Source', value_name='KG')

  fig = px.line(results, x='Date', y='KG', color='Source')
  return fig
```


```python
interact(apply_model)
```


    interactive(children=(Dropdown(description='dataset', options=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1…





    <function __main__.apply_model>



<a name='x.5.2'></a>

### 🏋️ Exercise-Discussion 5.1: How Would You Create a General Forecast Model?

[back to top](#top)

> After exploring how your model does on other order histories, what do you think is a good strategy for developing company wide order forecasts?

Some possible questions:

* should you create a single model for the whole company?
* could you embed meta data about the order in this all-inclusive model?
* should you make models specific to certain customers, products, etc. 
  * what kind of analysis could you do before hand to determine how your models should be grouped?


```python
melted = orders.melt(id_vars=['base_cake', 'truffle_type', 'primary_flavor', 'secondary_flavor',
       'color_group', 'customer'], var_name='month', value_name='kg')
```


```python
def my_eda(color=cat_cols):
  fig = px.line(melted, x='month', y='kg', color=color)
  return fig
```


```python
interact(my_eda)
```


    interactive(children=(Dropdown(description='color', options=('base_cake', 'truffle_type', 'primary_flavor', 's…





    <function __main__.my_eda>



<a name='x.5.2.1'></a>

### 🏋️ Exercise 5.2: EDA

[back to top](#top)

In our quest to create a model that works well for all orders to truffltopia. I tell you that there are some orders with patterned behavior, according to their meta data. Your first task, is to find out which categorical variable best separates the data. You can use any statistical or visual method you like

```
# recall the categorical variables:
['base_cake', 'truffle_type', 'primary_flavor', 'secondary_flavor', 'color_group', 'customer']
```

From C1 S6, it may be useful to think of this diagram:

<img src="https://cdn.scribbr.com/wp-content/uploads//2020/01/flowchart-for-choosing-a-statistical-test.png" width=800px></img>

<a name='x.5.2.2'></a>

### 🏋️ Exercise 5.3: Decide on Model

[back to top](#top)

Will you model the whole dataset together? Will you create a number of submodels? Choose based on the groupings you determined statistically significant in the data. 

As a base comparison I have provided a formula that trains a model on the entire order history:

```
data = orders
data = data[time_cols].values

batch_size = 256
window_size = 12

print("batch size: {}".format(batch_size))
print("window size: {}".format(window_size), end='\n\n')

# describes the split train 0-.6/val .6-.8/test .8-1
train_test_val_ratios = [0.8, 0.9]

X_train, y_train, X_val, y_val, X_test, y_test = train_test_process(data,
                                                        train_test_val_ratios,
                                                        window_size)

early_stopping = keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='loss'
)

model = keras.Sequential([
    layers.Dense(8, input_shape=[window_size]),
    layers.Dense(16),
    layers.Dense(32),
    layers.Dense(16),
    layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=10000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
  )

print(pd.DataFrame(history.history).tail())
```


```python
data = orders
data = data[time_cols].values

batch_size = 256
window_size = 12

print("batch size: {}".format(batch_size))
print("window size: {}".format(window_size), end='\n\n')

# describes the split train 0-.6/val .6-.8/test .8-1
train_test_val_ratios = [0.8, 0.9]

X_train, y_train, X_val, y_val, X_test, y_test = train_test_process(data,
                                                        train_test_val_ratios,
                                                        window_size)

early_stopping = keras.callbacks.EarlyStopping(
    patience=50,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='loss'
)

model = keras.Sequential([
    layers.Dense(8, input_shape=[window_size]),
    layers.Dense(16),
    layers.Dense(32),
    layers.Dense(16),
    layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=10000,
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
  )

print(pd.DataFrame(history.history).tail())
```

    batch size: 256
    window size: 12
    
    train size: 9300
    val size: 100
    test size: 200
    
                 loss     val_loss
    326  279111.15625  953265.0625
    327  322529.15625  580780.2500
    328  285901.56250  476007.4375
    329  302237.68750  496192.8125
    330  281779.40625  480916.6250


And a history of the loss with the following settings:
```
batch size: 256
window size: 12

train size: 9300
val size: 100
test size: 200

             loss     val_loss
326  279111.15625  953265.0625
327  322529.15625  580780.2500
328  285901.56250  476007.4375
329  302237.68750  496192.8125
330  281779.40625  480916.6250
```


```python
interact(apply_model, window_size=window_size)
```


    interactive(children=(Dropdown(description='dataset', options=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1…





    <function __main__.apply_model>




```python

```