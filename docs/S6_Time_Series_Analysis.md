# General Applications of Neural Networks <br> Session 6: Recurrent Neural Networks and Time Series Analysis

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

## 6.0 Preparing Environment and Importing Data

[back to top](#top)

<a name='x.0.1'></a>

### 6.0.1 Import Packages

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

def melt_results(model, X, y, window_size):
  y_pred = model.predict(X)
  results = pd.DataFrame(y_pred, y)
  results = results.reset_index()
  results.index = orders.loc[0, time_cols].index[window_size:]
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

### 6.0.2 Load Dataset

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
<p>13200 rows ?? 8 columns</p>
</div>



<a name='x.1'></a>

## 6.1 Why We Think in Sequences

[back to top](#top)

There are some problems that are best framed as a sequence in either the input or the output. For example, in our image classification we are performing a mapping of many-to-one: sequence input (the pixels) to a single output (classification). Other examples include:

* One-to-many: sequence output, e.x. word (if treated as a single input) to generate a picture
* Many-to-many: sequence input and output, e.x. machine translation (like english to mandarin)
* Synchronized many-to-many: synced sequence input and output, e.x. video classification

State of the art handling of sequences has occurred in a class of networks called recurrent neural networks

<a name='x.2'></a>

## 6.2 Recurrent Neural Networks

[back to top](#top)

Recurrent Neural Networks (RNNs) can be thought of as a FFNN with loops added into the architecture. This allows the network to retain information, create "memory" that can be associated with signals later in the sequence. 

We didn't go into much detail about the actual training algorithm of neural networks: **_back propagation_**. But what we will say here, is that this algorithm breaks down with recurrent neural networks because of the looped connections. A trick was created to overcome this, where the looped connections are unrolled, using a copy of the "unhooked" neuron to represent where the loop was initally fed back. This algorithm is called **_back propagation through time_**.

Another problem is introduced when training recurrent neural networks, in that the gradients calculated during back propagation can become very large, **_exploding gradients_**, or very small **_vanishing gradients_**. This problem is modulated in FNNNs by the ReLU, In RNNs, a more sophisticated gating mechanism is used in an architecture we call **_Long Short-Term Memory Networks_**

<p align=center>
<img src="https://miro.medium.com/max/4136/1*SKGAqkVVzT6co-sZ29ze-g.png"></img>
</p>
<small>LSTM shown in both typical and unfolded format</small>



### 6.2.1 Long Short-Term Memory Networks

[back to top](#top)

Long Short-Term Memory Networks (LSTMs) are a type of RNN that are trained using back propagation through time and overcome the vanishing/exploding gradient problem. Similar to CNNs, their architecture is composed of blocks, this time with memory blocks rather than convolutional blocks. A block is smarter than the classical neuron; it contains gates that manage the block's state and output. The gates are operated by a sigmoid function, determining whether they are open or closed (triggered or not trigerred). There are three types of gates within a memory block:

* Forget gate: decides what information is discarded
* Input gate: decides what information updates the memory state
* Output gate: decides what information to send forward depending on the input and memory state

These weights that configure these gates are learned during training, and their coordination allow each memory block to learn sophisticated relationships in and among sequenced data. 

> Big takeaway: memory blocks contain trainable parameters that allow the block to learn relationships between sequenced data 


<a name='x.3'></a>

## 6.3 Exploratory Data Analysis with Plotly/Pandas

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
<p>5 rows ?? 138 columns</p>
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
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="78860497-0227-4f11-9696-e7100d1cd14d" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("78860497-0227-4f11-9696-e7100d1cd14d")) {                    Plotly.newPlot(                        "78860497-0227-4f11-9696-e7100d1cd14d",                        [{"hovertemplate":"Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":["1/2010","2/2010","3/2010","4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[12570.33516482565,11569.168746227244,13616.8122044598,11884.3708810225,13950.332334409886,12781.156535682429,14256.210023357236,12887.711959877464,15038.574005789536,12626.48930557771,14611.291109090684,13194.81429999148,14921.016215576235,13477.391456909943,15409.211079596587,13999.215068692509,15597.436975845374,14098.12497823274,15596.818092478728,14941.69403166363,15715.347212025836,14181.212141927936,16282.0980055455,14650.929410064906,16433.209008286325,15400.579033515967,16756.981262857273,15128.148250492244,17523.979943307248,15413.044691473402,16366.26437701746,14568.470958551738,16901.11154186154,14659.021365286097,16494.903960781197,15398.721298130027,17938.090871773184,15850.35787113158,18236.778754419985,15956.750789202086,17401.696472111977,15890.10321935092,17283.79073343649,16302.509223010222,17229.64501478726,16223.309276278227,17796.223621100053,16344.001270241426,17782.006164552513,16326.588260101846,18253.569321985724,16818.12312918114,18554.33980878632,16900.704327264033,18479.00603218699,17042.963875823145,18287.35559715585,17244.887842050513,18822.494484753846,17603.725932131478,18766.104076650663,17170.12649068024,19632.147600450644,16856.921979192426,18854.690380403008,17880.884218985302,19087.480847049384,18196.112254637803,19770.963054596545,16488.739325030063,19699.01989730995,17194.707087425755,19372.65790157132,17715.24432224015,19227.53144133251,17691.136252909622,20114.53450629712,17926.252604903035,19880.02532889845,16690.02893115867,19928.02694695529,18553.766165315024,20547.154033981024,17301.11715078875,19538.97650435099,17902.44835514176,21269.577926886348,18842.69654955895,20095.445399491346,17670.300576591326,20310.884287446843,18754.84178182952,20736.279238797026,18617.387584546323,20783.71123390676,17470.755864944782,20523.579839792717,18796.936905805047,20028.582492587037,18677.535295190337,20048.1074217522,18929.24861718753,20571.15590247796,18207.204656231734,20839.04289237627,18966.53298378622,20909.977545252816,18589.807151786372,21287.370122673103,17987.976866769444,21111.062684974822,18538.311320658097,21797.26713239234,18935.35277235507,21331.37841983855,18783.75961074272,22139.12337340894,18553.79727063604,21579.50628438568,19726.43311123112,21147.624131226225,19232.360491469408,21575.52105110441,18856.1781102771,20701.25067582265,19406.448559709923,22328.687162949856,19384.824041986754,21449.154889830097,19554.40558950196,21873.104938389297,19572.860127015803],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"KG"}},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('78860497-0227-4f11-9696-e7100d1cd14d');
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

                        })                };                            </script>        </div>
</body>
</html>



```python
fig, ax = plt.subplots(1,1,figsize=(10,10))
pd.plotting.autocorrelation_plot(data['KG'], ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff3c213eb50>




    
![png](S6_Time_Series_Analysis_files/S6_Time_Series_Analysis_12_1.png)
    


Normally with time series data, we'd want to try a host of preprocessing techniques and remove the trend (really create two separate analyses, one of the trend and one of the seasonality) but to keep things simple and to showcase the utility of machine learning, we are going to deviate from the stats-like approach and work with our data as is. 

For more details on the stats-like models you can perform a cursory search on _ARIMA_, _ARMA_, _SARIMA_

<a name='x.4'></a>

## 6.4 Modeling

[back to top](#top)


```python
from tensorflow import keras
from tensorflow.keras import layers
```

<a name='x.4.1'></a>

### 6.4.1 Sweeping (Rolling) Window

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

### 6.4.2 FFNN

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
      <th>19</th>
      <td>4.243438e+06</td>
    </tr>
    <tr>
      <th>20</th>
      <td>4.240187e+06</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4.240501e+06</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4.247284e+06</td>
    </tr>
    <tr>
      <th>23</th>
      <td>4.238702e+06</td>
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
      <td>12562.842773</td>
    </tr>
    <tr>
      <th>13616.812204</th>
      <td>11562.307617</td>
    </tr>
    <tr>
      <th>11884.370881</th>
      <td>13608.662109</td>
    </tr>
    <tr>
      <th>13950.332334</th>
      <td>11877.311523</td>
    </tr>
    <tr>
      <th>12781.156536</th>
      <td>13941.969727</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>19384.824042</th>
      <td>22315.048828</td>
    </tr>
    <tr>
      <th>21449.154890</th>
      <td>19373.041016</td>
    </tr>
    <tr>
      <th>19554.405590</th>
      <td>21436.068359</td>
    </tr>
    <tr>
      <th>21873.104938</th>
      <td>19542.517578</td>
    </tr>
    <tr>
      <th>19572.860127</th>
      <td>21859.757812</td>
    </tr>
  </tbody>
</table>
<p>131 rows ?? 1 columns</p>
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





  <div id="df-afaaa018-2d7b-4c78-aa8c-d622f9dadb69">
    <div class="colab-df-container">
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
      <th>752</th>
      <td>550135.6875</td>
    </tr>
    <tr>
      <th>753</th>
      <td>555000.3750</td>
    </tr>
    <tr>
      <th>754</th>
      <td>550800.3125</td>
    </tr>
    <tr>
      <th>755</th>
      <td>551368.6250</td>
    </tr>
    <tr>
      <th>756</th>
      <td>548760.4375</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-afaaa018-2d7b-4c78-aa8c-d622f9dadb69')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-afaaa018-2d7b-4c78-aa8c-d622f9dadb69 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-afaaa018-2d7b-4c78-aa8c-d622f9dadb69');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




A cursory glance looks like our values are closer together


```python
results = melt_results(model, X, y, window_size)
```


```python
px.line(results, x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="82ca7aaa-b25e-4fa4-9917-a70105e71bfe" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("82ca7aaa-b25e-4fa4-9917-a70105e71bfe")) {                    Plotly.newPlot(                        "82ca7aaa-b25e-4fa4-9917-a70105e71bfe",                        [{"hovertemplate":"Source=real<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"real","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"real","orientation":"v","showlegend":true,"x":["4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[11884.3708810225,13950.332334409886,12781.156535682429,14256.210023357236,12887.711959877464,15038.574005789536,12626.48930557771,14611.291109090684,13194.81429999148,14921.016215576235,13477.391456909943,15409.211079596587,13999.215068692509,15597.436975845374,14098.12497823274,15596.818092478728,14941.69403166363,15715.347212025836,14181.212141927936,16282.0980055455,14650.929410064906,16433.209008286325,15400.579033515967,16756.981262857273,15128.148250492244,17523.979943307248,15413.044691473402,16366.26437701746,14568.470958551738,16901.11154186154,14659.021365286097,16494.903960781197,15398.721298130027,17938.090871773184,15850.35787113158,18236.778754419985,15956.750789202086,17401.696472111977,15890.10321935092,17283.79073343649,16302.509223010222,17229.64501478726,16223.309276278227,17796.223621100053,16344.001270241426,17782.006164552513,16326.588260101846,18253.569321985724,16818.12312918114,18554.33980878632,16900.704327264033,18479.00603218699,17042.963875823145,18287.35559715585,17244.887842050513,18822.494484753846,17603.725932131478,18766.104076650663,17170.12649068024,19632.147600450644,16856.921979192426,18854.690380403008,17880.884218985302,19087.480847049384,18196.112254637803,19770.963054596545,16488.739325030063,19699.01989730995,17194.707087425755,19372.65790157132,17715.24432224015,19227.53144133251,17691.136252909622,20114.53450629712,17926.252604903035,19880.02532889845,16690.02893115867,19928.02694695529,18553.766165315024,20547.154033981024,17301.11715078875,19538.97650435099,17902.44835514176,21269.577926886348,18842.69654955895,20095.445399491346,17670.300576591326,20310.884287446843,18754.84178182952,20736.279238797026,18617.387584546323,20783.71123390676,17470.755864944782,20523.579839792717,18796.936905805047,20028.582492587037,18677.535295190337,20048.1074217522,18929.24861718753,20571.15590247796,18207.204656231734,20839.04289237627,18966.53298378622,20909.977545252816,18589.807151786372,21287.370122673103,17987.976866769444,21111.062684974822,18538.311320658097,21797.26713239234,18935.35277235507,21331.37841983855,18783.75961074272,22139.12337340894,18553.79727063604,21579.50628438568,19726.43311123112,21147.624131226225,19232.360491469408,21575.52105110441,18856.1781102771,20701.25067582265,19406.448559709923,22328.687162949856,19384.824041986754,21449.154889830097,19554.40558950196,21873.104938389297,19572.860127015803],"yaxis":"y","type":"scatter"},{"hovertemplate":"Source=predicted<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"predicted","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"predicted","orientation":"v","showlegend":true,"x":["4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[11259.46875,13349.3271484375,11999.1884765625,13402.828125,12855.837890625,14155.4228515625,12754.638671875,15048.845703125,13186.6015625,14235.9033203125,13297.19921875,14713.466796875,13498.7080078125,15067.93359375,14165.580078125,15497.2734375,14365.888671875,15148.046875,15068.068359375,16118.8349609375,14179.611328125,15954.45703125,14863.759765625,16005.7275390625,15471.76171875,16870.5546875,15047.203125,17263.318359375,16314.33203125,16796.021484375,14613.373046875,16736.271484375,15206.1015625,16066.9560546875,14936.6484375,17584.765625,16049.8486328125,18068.611328125,16724.56640625,17411.38671875,16229.265625,17066.962890625,16557.314453125,17299.4609375,16180.7421875,17704.82421875,16630.2109375,17771.55078125,16378.6552734375,17948.849609375,16954.154296875,18469.576171875,17242.130859375,18381.76953125,17418.310546875,18195.595703125,17229.048828125,18623.435546875,17894.810546875,18993.314453125,17053.271484375,19668.8359375,17656.671875,18290.076171875,18008.35546875,18951.130859375,18099.71484375,20576.140625,16999.568359375,19164.525390625,17750.412109375,19035.54296875,18096.880859375,19223.251953125,17561.724609375,19894.462890625,18409.365234375,20418.21484375,17134.646484375,18833.755859375,18542.748046875,21094.056640625,18265.861328125,19157.732421875,17378.361328125,20617.12109375,19811.525390625,20679.423828125,17959.744140625,19659.283203125,18858.236328125,20747.884765625,18962.271484375,21262.759765625,18083.052734375,19712.927734375,19362.912109375,20106.318359375,18954.232421875,19932.1015625,18940.099609375,20899.845703125,18468.845703125,20349.244140625,19276.36328125,21042.38671875,18796.822265625,21450.583984375,18562.283203125,20671.611328125,18622.966796875,21422.41015625,19609.720703125,21309.9609375,18808.9453125,22059.017578125,19347.857421875,20853.78125,20282.482421875,21394.27734375,19376.556640625,21669.943359375,19711.716796875,20395.39453125,18906.919921875,22195.814453125,20270.896484375,21309.14453125,19700.73828125],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"KG"}},"legend":{"title":{"text":"Source"},"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('82ca7aaa-b25e-4fa4-9917-a70105e71bfe');
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

                        })                };                            </script>        </div>
</body>
</html>


<a name='x.4.2.1'></a>

#### ??????? Exercise-Discussion 1: Varify that the model is linear

[back to top](#top)

We're having to change our way of thinking here with time series analysis. Recall that a model without an activation function can only encapsulate linear relationships. How come we can see non-linear relationships in our time series plot above? make a plot that showcases we are indeed still within a linear world.

This is an open ended question, think about how you would attempt to show linearity of the model. (In [Lab 1](https://wesleybeckner.github.io/general_applications_of_neural_networks/labs/L1_Neural_Network_Linearity/) our model predicted on only 2 dimensions (vs 3, in this case), and it was a binary classification task, so it was easier to view the decision boundaries and verify linearity).


```python
# Code cell for Exercise 1
```

<a name='x.4.2.2'></a>

#### ??????? Exercise 2: Vary model architecture and window size

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
df = melt_results(model, X, y, window_size)
px.line(df, x='Date', y='KG', color='Source')
```

You can use the same early_stopping and fit formula from 6.4.2


```python
# Code cell for exercise 2
window_size = 3
batch_size = 10

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


X, y, labels = process_data(orders.loc[[0], time_cols].values, window=window_size, time_cols=132)

dfs = []
for model in models:
  model.compile(loss='mean_squared_error', optimizer='adam')
  history = model.fit(
      X, y,
      batch_size=batch_size,
      epochs=int(1e4),
      callbacks=[early_stopping],
      verbose=0, # hide the output because we have so many epochs
  )
  print(history.history['loss'][-10:])
  df = melt_results(model, X, y, window_size)
  dfs.append(df)
  px.line(df, x='Date', y='KG', color='Source')
```

    [449815.4375, 438517.75, 434076.40625, 436526.9375, 431749.71875, 432751.5, 431072.125, 433903.71875, 434614.8125, 434704.71875]
    [319843872.0, 319843424.0, 319842976.0, 319842528.0, 319842080.0, 319841632.0, 319841120.0, 319840704.0, 319840224.0, 319839776.0]
    [398923.6875, 398456.15625, 399380.0, 399915.5, 406269.09375, 400187.28125, 397825.96875, 412889.375, 399718.75, 402859.40625]



```python
px.line(dfs[2], x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="c14194e1-23b5-4c35-8582-9e0f03ea3eb3" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("c14194e1-23b5-4c35-8582-9e0f03ea3eb3")) {                    Plotly.newPlot(                        "c14194e1-23b5-4c35-8582-9e0f03ea3eb3",                        [{"hovertemplate":"Source=real<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"real","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"real","orientation":"v","showlegend":true,"x":["4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[11884.3708810225,13950.332334409886,12781.156535682429,14256.210023357236,12887.711959877464,15038.574005789536,12626.48930557771,14611.291109090684,13194.81429999148,14921.016215576235,13477.391456909943,15409.211079596587,13999.215068692509,15597.436975845374,14098.12497823274,15596.818092478728,14941.69403166363,15715.347212025836,14181.212141927936,16282.0980055455,14650.929410064906,16433.209008286325,15400.579033515967,16756.981262857273,15128.148250492244,17523.979943307248,15413.044691473402,16366.26437701746,14568.470958551738,16901.11154186154,14659.021365286097,16494.903960781197,15398.721298130027,17938.090871773184,15850.35787113158,18236.778754419985,15956.750789202086,17401.696472111977,15890.10321935092,17283.79073343649,16302.509223010222,17229.64501478726,16223.309276278227,17796.223621100053,16344.001270241426,17782.006164552513,16326.588260101846,18253.569321985724,16818.12312918114,18554.33980878632,16900.704327264033,18479.00603218699,17042.963875823145,18287.35559715585,17244.887842050513,18822.494484753846,17603.725932131478,18766.104076650663,17170.12649068024,19632.147600450644,16856.921979192426,18854.690380403008,17880.884218985302,19087.480847049384,18196.112254637803,19770.963054596545,16488.739325030063,19699.01989730995,17194.707087425755,19372.65790157132,17715.24432224015,19227.53144133251,17691.136252909622,20114.53450629712,17926.252604903035,19880.02532889845,16690.02893115867,19928.02694695529,18553.766165315024,20547.154033981024,17301.11715078875,19538.97650435099,17902.44835514176,21269.577926886348,18842.69654955895,20095.445399491346,17670.300576591326,20310.884287446843,18754.84178182952,20736.279238797026,18617.387584546323,20783.71123390676,17470.755864944782,20523.579839792717,18796.936905805047,20028.582492587037,18677.535295190337,20048.1074217522,18929.24861718753,20571.15590247796,18207.204656231734,20839.04289237627,18966.53298378622,20909.977545252816,18589.807151786372,21287.370122673103,17987.976866769444,21111.062684974822,18538.311320658097,21797.26713239234,18935.35277235507,21331.37841983855,18783.75961074272,22139.12337340894,18553.79727063604,21579.50628438568,19726.43311123112,21147.624131226225,19232.360491469408,21575.52105110441,18856.1781102771,20701.25067582265,19406.448559709923,22328.687162949856,19384.824041986754,21449.154889830097,19554.40558950196,21873.104938389297,19572.860127015803],"yaxis":"y","type":"scatter"},{"hovertemplate":"Source=predicted<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"predicted","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"predicted","orientation":"v","showlegend":true,"x":["4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[11787.9921875,13528.689453125,12110.7001953125,13907.7119140625,12961.4736328125,14203.09375,13122.7568359375,14902.7314453125,12854.0654296875,14552.697265625,13398.21875,14863.3095703125,13698.328125,15354.958984375,14196.06640625,15539.146484375,14288.1796875,15605.3486328125,15073.185546875,15660.1328125,14419.06640625,16213.244140625,14866.4833984375,16413.986328125,15582.1884765625,16695.1640625,15393.87890625,17421.05859375,15570.0146484375,16292.0771484375,14827.8798828125,16785.5859375,14882.7314453125,16470.625,15673.4736328125,17837.8828125,16121.4560546875,18123.783203125,16155.7373046875,17351.328125,16079.8515625,17274.8125,16454.083984375,17221.455078125,16425.328125,17751.4140625,16538.830078125,17737.751953125,16559.697265625,18209.806640625,17038.30078125,18496.013671875,17110.263671875,18438.421875,17225.5703125,18278.9296875,17451.978515625,18800.451171875,17780.96484375,18716.90625,17449.77734375,19485.955078125,17106.3671875,18849.30078125,18061.234375,19094.646484375,18406.181640625,19591.318359375,16834.88671875,19569.560546875,17458.021484375,19314.787109375,17923.216796875,19182.78125,17969.66015625,20016.71875,18173.166015625,19705.658203125,17038.638671875,19887.98828125,18801.3125,20371.0625,17574.3515625,19483.0859375,18256.52734375,21152.060546875,19040.1953125,19986.482421875,17971.458984375,20262.103515625,19003.498046875,20649.45703125,18883.546875,20602.298828125,17809.0546875,20459.7109375,18988.08984375,20003.669921875,18876.6484375,20040.59375,19149.24609375,20466.580078125,18509.63671875,20767.818359375,19215.16796875,20808.3828125,18898.787109375,21105.818359375,18333.9609375,20982.98046875,18892.697265625,21648.6484375,19225.6796875,21210.89453125,19146.6171875,21934.666015625,18896.458984375,21510.51953125,19937.326171875,21082.162109375,19513.771484375,21442.978515625,19102.048828125,20679.7578125,19730.041015625,22178.12890625,19651.638671875,21382.49609375,19835.1796875],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"KG"}},"legend":{"title":{"text":"Source"},"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('c14194e1-23b5-4c35-8582-9e0f03ea3eb3');
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

                        })                };                            </script>        </div>
</body>
</html>


<a name='x.4.3'></a>

### 6.4.3 LSTM NN

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

    WARNING:tensorflow:Layer lstm will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.



```python
history = model.fit(
    X, y,
    batch_size=batch_size,
    epochs=int(1e4),
    callbacks=[early_stopping],
    verbose=0, # hide the output because we have so many epochs
)
```


```python
history_df = pd.DataFrame(history.history)
history_df.tail()
```





  <div id="df-1c67f9fb-60da-4b60-a4a2-c57ecffd7221">
    <div class="colab-df-container">
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
      <th>4050</th>
      <td>4906831.5</td>
    </tr>
    <tr>
      <th>4051</th>
      <td>4906659.0</td>
    </tr>
    <tr>
      <th>4052</th>
      <td>4906790.5</td>
    </tr>
    <tr>
      <th>4053</th>
      <td>4906763.0</td>
    </tr>
    <tr>
      <th>4054</th>
      <td>4906620.5</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1c67f9fb-60da-4b60-a4a2-c57ecffd7221')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1c67f9fb-60da-4b60-a4a2-c57ecffd7221 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1c67f9fb-60da-4b60-a4a2-c57ecffd7221');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
results = melt_results(model, X, y.flatten(), window_size)
```


```python
px.line(results, x='Date', y='KG', color='Source')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="667e8358-593d-4c9c-84a1-9d04379b0ab1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("667e8358-593d-4c9c-84a1-9d04379b0ab1")) {                    Plotly.newPlot(                        "667e8358-593d-4c9c-84a1-9d04379b0ab1",                        [{"hovertemplate":"Source=real<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"real","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"real","orientation":"v","showlegend":true,"x":["7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[14256.210023357236,12887.711959877464,15038.574005789536,12626.48930557771,14611.291109090684,13194.81429999148,14921.016215576235,13477.391456909943,15409.211079596587,13999.215068692509,15597.436975845374,14098.12497823274,15596.818092478728,14941.69403166363,15715.347212025836,14181.212141927936,16282.0980055455,14650.929410064906,16433.209008286325,15400.579033515967,16756.981262857273,15128.148250492244,17523.979943307248,15413.044691473402,16366.26437701746,14568.470958551738,16901.11154186154,14659.021365286097,16494.903960781197,15398.721298130027,17938.090871773184,15850.35787113158,18236.778754419985,15956.750789202086,17401.696472111977,15890.10321935092,17283.79073343649,16302.509223010222,17229.64501478726,16223.309276278227,17796.223621100053,16344.001270241426,17782.006164552513,16326.588260101846,18253.569321985724,16818.12312918114,18554.33980878632,16900.704327264033,18479.00603218699,17042.963875823145,18287.35559715585,17244.887842050513,18822.494484753846,17603.725932131478,18766.104076650663,17170.12649068024,19632.147600450644,16856.921979192426,18854.690380403008,17880.884218985302,19087.480847049384,18196.112254637803,19770.963054596545,16488.739325030063,19699.01989730995,17194.707087425755,19372.65790157132,17715.24432224015,19227.53144133251,17691.136252909622,20114.53450629712,17926.252604903035,19880.02532889845,16690.02893115867,19928.02694695529,18553.766165315024,20547.154033981024,17301.11715078875,19538.97650435099,17902.44835514176,21269.577926886348,18842.69654955895,20095.445399491346,17670.300576591326,20310.884287446843,18754.84178182952,20736.279238797026,18617.387584546323,20783.71123390676,17470.755864944782,20523.579839792717,18796.936905805047,20028.582492587037,18677.535295190337,20048.1074217522,18929.24861718753,20571.15590247796,18207.204656231734,20839.04289237627,18966.53298378622,20909.977545252816,18589.807151786372,21287.370122673103,17987.976866769444,21111.062684974822,18538.311320658097,21797.26713239234,18935.35277235507,21331.37841983855,18783.75961074272,22139.12337340894,18553.79727063604,21579.50628438568,19726.43311123112,21147.624131226225,19232.360491469408,21575.52105110441,18856.1781102771,20701.25067582265,19406.448559709923,22328.687162949856,19384.824041986754,21449.154889830097,19554.40558950196,21873.104938389297,19572.860127015803],"yaxis":"y","type":"scatter"},{"hovertemplate":"Source=predicted<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"predicted","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"predicted","orientation":"v","showlegend":true,"x":["7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625,17969.009765625],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"KG"}},"legend":{"title":{"text":"Source"},"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('667e8358-593d-4c9c-84a1-9d04379b0ab1');
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

                        })                };                            </script>        </div>
</body>
</html>


<a name='x.3.4.1'></a>

#### ??????? Exercise 3: Compare LSTM with FFNN using Train/Val/Test sets and 3 Month Window

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

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    monitor='loss'
)

### YOUR MODEL FROM EX 6.3.3.2 ###

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[window_size]),
    layers.Dense(1)
])

### UNCOMMENT THE BELOW ###
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
    796  274950.81250  705562.1875
    797  278971.28125  726105.4375
    798  283036.78125  696195.4375
    799  284960.03125  723620.4375
    800  279106.31250  691688.0000


We'll then record the mse performance of the model to later compare with the LSTM


```python
results = []
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
results.append(['Dense', mse])
results
```




    [['Dense', 444380.32838419516]]



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

    WARNING:tensorflow:Layer lstm_1 will not use cuDNN kernels since it doesn't meet the criteria. It will use a generic GPU kernel as fallback when running on GPU.
                loss     val_loss
    836  273390.4375  673023.0625
    837  271066.7500  694139.5625
    838  275661.9375  705827.1875
    839  274106.7500  680028.2500
    840  270606.5000  691417.3750



```python
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test.flatten(), y_pred.flatten())
results.append(['LSTM', mse])
```

Comparison of results:


```python
pd.DataFrame(results, columns=['Model', 'Test MSE']).set_index('Model').astype(int)
```





  <div id="df-fed8b43f-e700-4717-b722-1ea75feecdf1">
    <div class="colab-df-container">
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
      <th>Test MSE</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dense</th>
      <td>444380</td>
    </tr>
    <tr>
      <th>LSTM</th>
      <td>424835</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fed8b43f-e700-4717-b722-1ea75feecdf1')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fed8b43f-e700-4717-b722-1ea75feecdf1 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fed8b43f-e700-4717-b722-1ea75feecdf1');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




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
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>                <div id="7609a4ed-f908-4f31-9cac-ebbc76aaae8f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("7609a4ed-f908-4f31-9cac-ebbc76aaae8f")) {                    Plotly.newPlot(                        "7609a4ed-f908-4f31-9cac-ebbc76aaae8f",                        [{"hovertemplate":"Source=Real<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"Real","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Real","orientation":"v","showlegend":true,"x":["1/2010","2/2010","3/2010","4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[12570.33516482565,11569.168746227244,13616.8122044598,11884.3708810225,13950.332334409886,12781.156535682429,14256.210023357236,12887.711959877464,15038.574005789536,12626.48930557771,14611.291109090684,13194.81429999148,14921.016215576235,13477.391456909943,15409.211079596587,13999.215068692509,15597.436975845374,14098.12497823274,15596.818092478728,14941.69403166363,15715.347212025836,14181.212141927936,16282.0980055455,14650.929410064906,16433.209008286325,15400.579033515967,16756.981262857273,15128.148250492244,17523.979943307248,15413.044691473402,16366.26437701746,14568.470958551738,16901.11154186154,14659.021365286097,16494.903960781197,15398.721298130027,17938.090871773184,15850.35787113158,18236.778754419985,15956.750789202086,17401.696472111977,15890.10321935092,17283.79073343649,16302.509223010222,17229.64501478726,16223.309276278227,17796.223621100053,16344.001270241426,17782.006164552513,16326.588260101846,18253.569321985724,16818.12312918114,18554.33980878632,16900.704327264033,18479.00603218699,17042.963875823145,18287.35559715585,17244.887842050513,18822.494484753846,17603.725932131478,18766.104076650663,17170.12649068024,19632.147600450644,16856.921979192426,18854.690380403008,17880.884218985302,19087.480847049384,18196.112254637803,19770.963054596545,16488.739325030063,19699.01989730995,17194.707087425755,19372.65790157132,17715.24432224015,19227.53144133251,17691.136252909622,20114.53450629712,17926.252604903035,19880.02532889845,16690.02893115867,19928.02694695529,18553.766165315024,20547.154033981024,17301.11715078875,19538.97650435099,17902.44835514176,21269.577926886348,18842.69654955895,20095.445399491346,17670.300576591326,20310.884287446843,18754.84178182952,20736.279238797026,18617.387584546323,20783.71123390676,17470.755864944782,20523.579839792717,18796.936905805047,20028.582492587037,18677.535295190337,20048.1074217522,18929.24861718753,20571.15590247796,18207.204656231734,20839.04289237627,18966.53298378622,20909.977545252816,18589.807151786372,21287.370122673103,17987.976866769444,21111.062684974822,18538.311320658097,21797.26713239234,18935.35277235507,21331.37841983855,18783.75961074272,22139.12337340894,18553.79727063604,21579.50628438568,19726.43311123112,21147.624131226225,19232.360491469408,21575.52105110441,18856.1781102771,20701.25067582265,19406.448559709923,22328.687162949856,19384.824041986754,21449.154889830097,19554.40558950196,21873.104938389297,19572.860127015803],"yaxis":"y","type":"scatter"},{"hovertemplate":"Source=Train<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"Train","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Train","orientation":"v","showlegend":true,"x":["1/2010","2/2010","3/2010","4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[null,null,null,11909.3662109375,13577.0400390625,12182.8759765625,14002.0751953125,13029.9931640625,14241.2119140625,13230.0498046875,14909.6240234375,12874.17578125,14624.0224609375,13470.0166015625,14915.193359375,13784.998046875,15425.3427734375,14260.6826171875,15579.6259765625,14339.052734375,15704.4052734375,15129.6240234375,15640.1455078125,14514.5126953125,16280.8583984375,14931.6865234375,16506.08984375,15658.6220703125,16711.865234375,15508.2880859375,17475.73828125,15539.1826171875,16266.123046875,14923.8681640625,16823.8359375,14909.078125,16561.744140625,15837.341796875,17905.666015625,16204.8701171875,18166.705078125,16152.7568359375,17385.224609375,16126.83984375,17345.96484375,16503.3671875,17257.6328125,16522.794921875,17800.07421875,16594.74609375,17776.615234375,16653.330078125,18286.029296875,17119.927734375,18542.703125,17164.484375,18490.673828125,17269.587890625,18337.6875,17550.1953125,18870.525390625,17835.525390625,18728.25,17577.498046875,19499.7265625,17114.029296875,18968.1015625,18137.283203125,19164.779296875,18517.556640625,19503.587890625,16899.98046875,19657.294921875,17499.73828125,19394.65234375,17974.37890625,19224.732421875,18100.0703125,20074.755859375,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],"yaxis":"y","type":"scatter"},{"hovertemplate":"Source=Val<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"Val","line":{"color":"#00cc96","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Val","orientation":"v","showlegend":true,"x":["1/2010","2/2010","3/2010","4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,20062.798828125,18912.208984375,20317.857421875,17568.76953125,19569.275390625,18453.818359375,21261.365234375,19020.208984375,19943.580078125,18056.06640625,20385.130859375,19101.236328125,20683.533203125,18955.59765625,20556.7578125,17862.564453125,20599.138671875,19015.669921875,20042.546875,18941.142578125,20107.35546875,19251.904296875,20457.21484375,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],"yaxis":"y","type":"scatter"},{"hovertemplate":"Source=Test<br>Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"Test","line":{"color":"#ab63fa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines","name":"Test","orientation":"v","showlegend":true,"x":["1/2010","2/2010","3/2010","4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,20824.638671875,18997.853515625,21100.205078125,18395.37109375,21063.302734375,19017.439453125,21718.04296875,19264.138671875,21242.572265625,19281.326171875,21955.626953125,18931.779296875,21641.244140625,19973.392578125,21093.765625,19615.673828125,21458.291015625,19107.515625,20768.154296875,19921.154296875,22218.95703125,19659.841796875,21441.87109375,19937.548828125],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"KG"}},"legend":{"title":{"text":"Source"},"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('7609a4ed-f908-4f31-9cac-ebbc76aaae8f');
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

                        })                };                            </script>        </div>
</body>
</html>


<a name='x.5'></a>

## 6.5 Model Extensibility

[back to top](#top)


```python
from ipywidgets import interact
```

<a name='x.5.1'></a>

### ??????? Exercise 4: Apply Model to Other Orders

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


    interactive(children=(Dropdown(description='dataset', options=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1???





    <function __main__.apply_model>



<a name='x.5.2'></a>

### ??????? Exercise-Discussion 5.1: How Would You Create a General Forecast Model?

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


    interactive(children=(Dropdown(description='color', options=('base_cake', 'truffle_type', 'primary_flavor', 's???





    <function __main__.my_eda>



<a name='x.5.2.1'></a>

### ??????? Exercise 5.2: EDA

[back to top](#top)

In our quest to create a model that works well for all orders to truffltopia. I tell you that there are some orders with patterned behavior, according to their meta data. Your first task, is to find out which categorical variable best separates the data. You can use any statistical or visual method you like

```
# recall the categorical variables:
['base_cake', 'truffle_type', 'primary_flavor', 'secondary_flavor', 'color_group', 'customer']
```

From C1 S6, it may be useful to think of this diagram:

<img src="https://cdn.scribbr.com/wp-content/uploads//2020/01/flowchart-for-choosing-a-statistical-test.png" width=800px></img>

<a name='x.5.2.2'></a>

### ??????? Exercise 5.3: Decide on Model

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
    
                 loss      val_loss
    178  273904.93750  533661.50000
    179  288718.50000  519464.28125
    180  295474.71875  513898.46875
    181  299524.78125  664799.06250
    182  283324.56250  509953.53125


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
interact(apply_model)
```
