# Data Cleaning


```python
import numpy as np
import pandas as pd
import re
from functools import reduce
```


```python
df = pd.read_csv("../data/original.csv")
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Pos</th>
      <th>Name</th>
      <th>Club</th>
      <th>Cat</th>
      <th>Swim</th>
      <th>T1</th>
      <th>Bike</th>
      <th>T2</th>
      <th>Run</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>FRODENO Jan (2)</td>
      <td>Laz  Saarbruecken</td>
      <td>MPRO - 1</td>
      <td>00:22:501</td>
      <td>00:01:2710</td>
      <td>02:02:011</td>
      <td>00:01:5221</td>
      <td>01:11:252</td>
      <td>03:39:35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>CLAVEL Maurice (24)</td>
      <td>NaN</td>
      <td>MPRO - 2</td>
      <td>00:23:3311</td>
      <td>00:01:3015</td>
      <td>02:04:543</td>
      <td>00:01:323</td>
      <td>01:12:144</td>
      <td>03:43:43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>TAAGHOLT Miki  Morck (21)</td>
      <td>Ttsdu</td>
      <td>MPRO - 3</td>
      <td>00:22:574</td>
      <td>00:01:141</td>
      <td>02:05:526</td>
      <td>00:02:0148</td>
      <td>01:14:238</td>
      <td>03:46:27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>STRATMANN Jan (13)</td>
      <td>Triathlon  Team  Witten</td>
      <td>MPRO - 4</td>
      <td>00:22:502</td>
      <td>00:01:3118</td>
      <td>02:08:208</td>
      <td>00:01:5633</td>
      <td>01:13:336</td>
      <td>03:48:10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>MOLINARI Giulio (22)</td>
      <td>C. S. Carabinieri</td>
      <td>MPRO - 5</td>
      <td>00:22:585</td>
      <td>00:01:4968</td>
      <td>02:05:114</td>
      <td>00:02:0768</td>
      <td>01:17:2117</td>
      <td>03:49:26</td>
    </tr>
  </tbody>
</table>
</div>



### Remove Uninteresting Columns

We initially remove the unnamed column, the position, the athlete name, and the club.


```python
cols = df.columns
df.drop(columns=[cols[0], 'Pos', 'Name', 'Club'], inplace=True)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cat</th>
      <th>Swim</th>
      <th>T1</th>
      <th>Bike</th>
      <th>T2</th>
      <th>Run</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MPRO - 1</td>
      <td>00:22:501</td>
      <td>00:01:2710</td>
      <td>02:02:011</td>
      <td>00:01:5221</td>
      <td>01:11:252</td>
      <td>03:39:35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MPRO - 2</td>
      <td>00:23:3311</td>
      <td>00:01:3015</td>
      <td>02:04:543</td>
      <td>00:01:323</td>
      <td>01:12:144</td>
      <td>03:43:43</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MPRO - 3</td>
      <td>00:22:574</td>
      <td>00:01:141</td>
      <td>02:05:526</td>
      <td>00:02:0148</td>
      <td>01:14:238</td>
      <td>03:46:27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MPRO - 4</td>
      <td>00:22:502</td>
      <td>00:01:3118</td>
      <td>02:08:208</td>
      <td>00:01:5633</td>
      <td>01:13:336</td>
      <td>03:48:10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MPRO - 5</td>
      <td>00:22:585</td>
      <td>00:01:4968</td>
      <td>02:05:114</td>
      <td>00:02:0768</td>
      <td>01:17:2117</td>
      <td>03:49:26</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Helper functions for processing/filtering

def all(ls: list) -> bool:
    return reduce(lambda a, b: a and b, ls)

def startswith(s: str, prefix: str) -> bool:
    s_ = str(s)
    return (len(s_) >= len(prefix)) and all(map(lambda tup: tup[0] == tup[1], zip(s_,prefix)))

def indexof(v, arr) -> int:
    for i, x in enumerate(arr):
        if v == x:
            return i
    return -1
```

### Convert 'Cat' column into 'Gender' and 'AgeGroup' columns


```python
# Remove the Category placing suffix

def parsecategory(s: str) -> str:
    s_ = str(s)
    res = re.split(r"\W+", s_)
    return res[0] if len(res) > 0 else ""

# return 0 for female, 1 for male
def parsegender(cat: str) -> int:
    if cat[0] == 'F':
        return 0
    elif cat[0] == 'M':
        return 1
    else:
        return -1

parsed_cats = df['Cat'].apply(parsecategory)
df['Gender'] = parsed_cats.apply(parsegender)

agegroups = parsed_cats.apply(lambda ag: ag[1:])
df['AgeGroup'] = agegroups.apply(lambda v: indexof(v, np.unique(agegroups)))

df = df.loc[df['Gender'] != -1]

df.drop(columns=['Cat'], inplace=True)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Swim</th>
      <th>T1</th>
      <th>Bike</th>
      <th>T2</th>
      <th>Run</th>
      <th>Time</th>
      <th>Gender</th>
      <th>AgeGroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00:22:501</td>
      <td>00:01:2710</td>
      <td>02:02:011</td>
      <td>00:01:5221</td>
      <td>01:11:252</td>
      <td>03:39:35</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00:23:3311</td>
      <td>00:01:3015</td>
      <td>02:04:543</td>
      <td>00:01:323</td>
      <td>01:12:144</td>
      <td>03:43:43</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00:22:574</td>
      <td>00:01:141</td>
      <td>02:05:526</td>
      <td>00:02:0148</td>
      <td>01:14:238</td>
      <td>03:46:27</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00:22:502</td>
      <td>00:01:3118</td>
      <td>02:08:208</td>
      <td>00:01:5633</td>
      <td>01:13:336</td>
      <td>03:48:10</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00:22:585</td>
      <td>00:01:4968</td>
      <td>02:05:114</td>
      <td>00:02:0768</td>
      <td>01:17:2117</td>
      <td>03:49:26</td>
      <td>1</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



### Parse the time fields into seconds


```python
def parsetime(s: str) -> float:
    res = re.split(':', s)
    if len(res) != 3:
        return 0
    else:
        hrs, mins, secs = res
        secs = secs[0:2]
        return 60*60*int(hrs) + 60*int(mins) + int(secs)

```


```python
timecols = ['Swim', 'T1', 'Bike', 'T2', 'Run', 'Time']
timecols

for timecol in timecols:
    df[timecol] = df[timecol].apply(parsetime)
```


```python
# remove any rows with values of 0

df = df.loc[df[timecols].apply(lambda row: all(map(lambda col: col != 0, [row[col] for col in timecols])), axis=1)]
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Swim</th>
      <th>T1</th>
      <th>Bike</th>
      <th>T2</th>
      <th>Run</th>
      <th>Time</th>
      <th>Gender</th>
      <th>AgeGroup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1370</td>
      <td>87</td>
      <td>7321</td>
      <td>112</td>
      <td>4285</td>
      <td>13175</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1413</td>
      <td>90</td>
      <td>7494</td>
      <td>92</td>
      <td>4334</td>
      <td>13423</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1377</td>
      <td>74</td>
      <td>7552</td>
      <td>121</td>
      <td>4463</td>
      <td>13587</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1370</td>
      <td>91</td>
      <td>7700</td>
      <td>116</td>
      <td>4413</td>
      <td>13690</td>
      <td>1</td>
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1378</td>
      <td>109</td>
      <td>7511</td>
      <td>127</td>
      <td>4641</td>
      <td>13766</td>
      <td>1</td>
      <td>9</td>
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
      <th>795</th>
      <td>2345</td>
      <td>213</td>
      <td>9887</td>
      <td>208</td>
      <td>6624</td>
      <td>19277</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>796</th>
      <td>2201</td>
      <td>200</td>
      <td>9913</td>
      <td>237</td>
      <td>6727</td>
      <td>19278</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>797</th>
      <td>2481</td>
      <td>338</td>
      <td>9599</td>
      <td>163</td>
      <td>6698</td>
      <td>19279</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>798</th>
      <td>2436</td>
      <td>128</td>
      <td>9676</td>
      <td>168</td>
      <td>6878</td>
      <td>19286</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>799</th>
      <td>2230</td>
      <td>278</td>
      <td>9702</td>
      <td>292</td>
      <td>6785</td>
      <td>19286</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>797 rows × 8 columns</p>
</div>



### Output Clean Data to CSV


```python
df.to_csv('../data/cleaned.csv', index=False)
```
