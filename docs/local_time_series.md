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
      <th>...</th>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
      <td>...</td>
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
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script> 
                          <div id="98732d03-d58b-4130-9666-2568a26317b9" class="plotly-graph-div" style="height:525px; width:100%;"></div>            
                          <script type="text/javascript">                window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("98732d03-d58b-4130-9666-2568a26317b9")) {                    Plotly.newPlot(                        "98732d03-d58b-4130-9666-2568a26317b9",                        [{"hovertemplate": "Date=%{x}<br>KG=%{y}<extra></extra>", "legendgroup": "", "marker": {"color": "#636efa", "symbol": "circle"}, "mode": "markers", "name": "", "orientation": "v", "showlegend": false, "type": "scatter", "x": ["1/2010", "2/2010", "3/2010", "4/2010", "5/2010", "6/2010", "7/2010", "8/2010", "9/2010", "10/2010", "11/2010", "12/2010", "1/2011", "2/2011", "3/2011", "4/2011", "5/2011", "6/2011", "7/2011", "8/2011", "9/2011", "10/2011", "11/2011", "12/2011", "1/2012", "2/2012", "3/2012", "4/2012", "5/2012", "6/2012", "7/2012", "8/2012", "9/2012", "10/2012", "11/2012", "12/2012", "1/2013", "2/2013", "3/2013", "4/2013", "5/2013", "6/2013", "7/2013", "8/2013", "9/2013", "10/2013", "11/2013", "12/2013", "1/2014", "2/2014", "3/2014", "4/2014", "5/2014", "6/2014", "7/2014", "8/2014", "9/2014", "10/2014", "11/2014", "12/2014", "1/2015", "2/2015", "3/2015", "4/2015", "5/2015", "6/2015", "7/2015", "8/2015", "9/2015", "10/2015", "11/2015", "12/2015", "1/2016", "2/2016", "3/2016", "4/2016", "5/2016", "6/2016", "7/2016", "8/2016", "9/2016", "10/2016", "11/2016", "12/2016", "1/2017", "2/2017", "3/2017", "4/2017", "5/2017", "6/2017", "7/2017", "8/2017", "9/2017", "10/2017", "11/2017", "12/2017", "1/2018", "2/2018", "3/2018", "4/2018", "5/2018", "6/2018", "7/2018", "8/2018", "9/2018", "10/2018", "11/2018", "12/2018", "1/2019", "2/2019", "3/2019", "4/2019", "5/2019", "6/2019", "7/2019", "8/2019", "9/2019", "10/2019", "11/2019", "12/2019", "1/2020", "2/2020", "3/2020", "4/2020", "5/2020", "6/2020", "7/2020", "8/2020", "9/2020", "10/2020", "11/2020", "12/2020"], "xaxis": "x", "y": [12570.33516482565, 11569.168746227244, 13616.8122044598, 11884.3708810225, 13950.332334409884, 12781.156535682429, 14256.210023357236, 12887.711959877463, 15038.574005789536, 12626.48930557771, 14611.291109090684, 13194.81429999148, 14921.016215576235, 13477.391456909943, 15409.211079596587, 13999.215068692507, 15597.436975845374, 14098.12497823274, 15596.818092478728, 14941.69403166363, 15715.347212025836, 14181.212141927937, 16282.0980055455, 14650.929410064904, 16433.20900828632, 15400.579033515967, 16756.981262857273, 15128.148250492244, 17523.979943307248, 15413.0446914734, 16366.264377017458, 14568.470958551738, 16901.11154186154, 14659.021365286097, 16494.903960781197, 15398.721298130027, 17938.090871773184, 15850.35787113158, 18236.778754419982, 15956.750789202086, 17401.696472111977, 15890.103219350918, 17283.79073343649, 16302.509223010222, 17229.645014787257, 16223.309276278227, 17796.223621100053, 16344.001270241426, 17782.006164552513, 16326.588260101846, 18253.569321985724, 16818.123129181142, 18554.33980878632, 16900.704327264033, 18479.00603218699, 17042.963875823145, 18287.35559715585, 17244.887842050513, 18822.494484753846, 17603.725932131478, 18766.104076650663, 17170.126490680243, 19632.147600450644, 16856.921979192426, 18854.690380403008, 17880.884218985302, 19087.480847049384, 18196.112254637806, 19770.963054596545, 16488.739325030063, 19699.01989730995, 17194.707087425755, 19372.657901571318, 17715.24432224015, 19227.53144133251, 17691.136252909622, 20114.534506297117, 17926.25260490304, 19880.02532889845, 16690.02893115867, 19928.02694695529, 18553.766165315024, 20547.154033981024, 17301.11715078875, 19538.97650435099, 17902.44835514176, 21269.577926886348, 18842.69654955895, 20095.445399491346, 17670.300576591326, 20310.884287446843, 18754.84178182952, 20736.279238797022, 18617.387584546323, 20783.71123390676, 17470.755864944782, 20523.579839792714, 18796.93690580505, 20028.582492587037, 18677.535295190337, 20048.1074217522, 18929.24861718753, 20571.15590247796, 18207.20465623173, 20839.04289237627, 18966.53298378622, 20909.977545252816, 18589.807151786372, 21287.370122673103, 17987.976866769444, 21111.062684974826, 18538.311320658097, 21797.267132392342, 18935.35277235507, 21331.37841983855, 18783.75961074272, 22139.12337340894, 18553.79727063604, 21579.50628438568, 19726.43311123112, 21147.624131226225, 19232.360491469408, 21575.52105110441, 18856.1781102771, 20701.25067582265, 19406.448559709923, 22328.68716294986, 19384.824041986754, 21449.154889830093, 19554.40558950196, 21873.104938389297, 19572.860127015803], "yaxis": "y"}],                        {"legend": {"tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "Date"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "KG"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('98732d03-d58b-4130-9666-2568a26317b9');
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

                        })                };                            
                        </script>        </div> 
                        </body> </html>

