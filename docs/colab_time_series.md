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
!pip install plotly==5.6
```

    Collecting plotly==5.6
      Downloading plotly-5.6.0-py2.py3-none-any.whl (27.7 MB)
    [K     |████████████████████████████████| 27.7 MB 39.1 MB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from plotly==5.6) (1.15.0)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.7/dist-packages (from plotly==5.6) (8.0.1)
    Installing collected packages: plotly
      Attempting uninstall: plotly
        Found existing installation: plotly 4.14.3
        Uninstalling plotly-4.14.3:
          Successfully uninstalled plotly-4.14.3
    Successfully installed plotly-5.6.0



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

    /usr/local/lib/python3.7/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm


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





  <div id="df-d65e9f28-ef7d-4d9a-82af-25a97dc82a5e">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-d65e9f28-ef7d-4d9a-82af-25a97dc82a5e')"
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
          document.querySelector('#df-d65e9f28-ef7d-4d9a-82af-25a97dc82a5e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d65e9f28-ef7d-4d9a-82af-25a97dc82a5e');
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





  <div id="df-caf1d81d-b751-4dff-bd59-c9b4f396032e">
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
      <button class="colab-df-convert" onclick="convertToInteractive('df-caf1d81d-b751-4dff-bd59-c9b4f396032e')"
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
          document.querySelector('#df-caf1d81d-b751-4dff-bd59-c9b4f396032e button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-caf1d81d-b751-4dff-bd59-c9b4f396032e');
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
        <div id="fbb6d422-ab24-406a-bee5-14d68f583b3f" class="plotly-graph-div" style="height:525px; width:100%;"></div>            
        <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("fbb6d422-ab24-406a-bee5-14d68f583b3f")) {                    Plotly.newPlot(                        "fbb6d422-ab24-406a-bee5-14d68f583b3f",                        [{"hovertemplate":"Date=%{x}<br>KG=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"","orientation":"v","showlegend":false,"x":["1/2010","2/2010","3/2010","4/2010","5/2010","6/2010","7/2010","8/2010","9/2010","10/2010","11/2010","12/2010","1/2011","2/2011","3/2011","4/2011","5/2011","6/2011","7/2011","8/2011","9/2011","10/2011","11/2011","12/2011","1/2012","2/2012","3/2012","4/2012","5/2012","6/2012","7/2012","8/2012","9/2012","10/2012","11/2012","12/2012","1/2013","2/2013","3/2013","4/2013","5/2013","6/2013","7/2013","8/2013","9/2013","10/2013","11/2013","12/2013","1/2014","2/2014","3/2014","4/2014","5/2014","6/2014","7/2014","8/2014","9/2014","10/2014","11/2014","12/2014","1/2015","2/2015","3/2015","4/2015","5/2015","6/2015","7/2015","8/2015","9/2015","10/2015","11/2015","12/2015","1/2016","2/2016","3/2016","4/2016","5/2016","6/2016","7/2016","8/2016","9/2016","10/2016","11/2016","12/2016","1/2017","2/2017","3/2017","4/2017","5/2017","6/2017","7/2017","8/2017","9/2017","10/2017","11/2017","12/2017","1/2018","2/2018","3/2018","4/2018","5/2018","6/2018","7/2018","8/2018","9/2018","10/2018","11/2018","12/2018","1/2019","2/2019","3/2019","4/2019","5/2019","6/2019","7/2019","8/2019","9/2019","10/2019","11/2019","12/2019","1/2020","2/2020","3/2020","4/2020","5/2020","6/2020","7/2020","8/2020","9/2020","10/2020","11/2020","12/2020"],"xaxis":"x","y":[12570.33516482565,11569.168746227244,13616.8122044598,11884.3708810225,13950.332334409886,12781.156535682429,14256.210023357236,12887.711959877464,15038.574005789536,12626.48930557771,14611.291109090684,13194.81429999148,14921.016215576235,13477.391456909943,15409.211079596587,13999.215068692509,15597.436975845374,14098.12497823274,15596.818092478728,14941.69403166363,15715.347212025836,14181.212141927936,16282.0980055455,14650.929410064906,16433.209008286325,15400.579033515967,16756.981262857273,15128.148250492244,17523.979943307248,15413.044691473402,16366.26437701746,14568.470958551738,16901.11154186154,14659.021365286097,16494.903960781197,15398.721298130027,17938.090871773184,15850.35787113158,18236.778754419985,15956.750789202086,17401.696472111977,15890.10321935092,17283.79073343649,16302.509223010222,17229.64501478726,16223.309276278227,17796.223621100053,16344.001270241426,17782.006164552513,16326.588260101846,18253.569321985724,16818.12312918114,18554.33980878632,16900.704327264033,18479.00603218699,17042.963875823145,18287.35559715585,17244.887842050513,18822.494484753846,17603.725932131478,18766.104076650663,17170.12649068024,19632.147600450644,16856.921979192426,18854.690380403008,17880.884218985302,19087.480847049384,18196.112254637803,19770.963054596545,16488.739325030063,19699.01989730995,17194.707087425755,19372.65790157132,17715.24432224015,19227.53144133251,17691.136252909622,20114.53450629712,17926.252604903035,19880.02532889845,16690.02893115867,19928.02694695529,18553.766165315024,20547.154033981024,17301.11715078875,19538.97650435099,17902.44835514176,21269.577926886348,18842.69654955895,20095.445399491346,17670.300576591326,20310.884287446843,18754.84178182952,20736.279238797026,18617.387584546323,20783.71123390676,17470.755864944782,20523.579839792717,18796.936905805047,20028.582492587037,18677.535295190337,20048.1074217522,18929.24861718753,20571.15590247796,18207.204656231734,20839.04289237627,18966.53298378622,20909.977545252816,18589.807151786372,21287.370122673103,17987.976866769444,21111.062684974822,18538.311320658097,21797.26713239234,18935.35277235507,21331.37841983855,18783.75961074272,22139.12337340894,18553.79727063604,21579.50628438568,19726.43311123112,21147.624131226225,19232.360491469408,21575.52105110441,18856.1781102771,20701.25067582265,19406.448559709923,22328.687162949856,19384.824041986754,21449.154889830097,19554.40558950196,21873.104938389297,19572.860127015803],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":0.05},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Date"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"KG"}},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('fbb6d422-ab24-406a-bee5-14d68f583b3f');
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
</body>
</html>



