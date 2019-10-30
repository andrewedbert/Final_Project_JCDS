![purwadhikaLogo](https://d1ah56qj523gwb.cloudfront.net/uploads/organizations/logos/1538557444-kcgv11HXelvcOnlyrGcEpfwAf6hbPMhC.png)

# Poisonous Mushroom Classification Web App - Purwadhika Job Connector Data Science Batch 05 Final Project

The terms "mushroom" and "toadstool" go back centuries and were never precisely defined, nor was there consensus on application. Between 1400 and 1600 AD, the terms mushrom, mushrum, muscheron, mousheroms, mussheron, or musserouns were used. The term "mushroom" and its variations may have been derived from the French word mousseron in reference to moss (mousse). Delineation between edible and poisonous fungi is not clear-cut, so a "mushroom" may be edible, poisonous, or unpalatable. Cultural or social phobias of mushrooms and fungi may be related. The term "fungophobia" was coined by William Delisle Hay of England, who noted a national superstition or fear of "toadstools". The word "toadstool" has apparent analogies in Dutch padde(n)stoel (toad-stool/chair, mushroom) and German Krötenschwamm (toad-fungus, alt. word for panther cap). In German folklore and old fairy tales, toads are often depicted sitting on toadstool mushrooms and catching, with their tongues, the flies that are said to be drawn to the Fliegenpilz, a German name for the toadstool, meaning "flies' mushroom". This is how the mushroom got another of its names, Krötenstuhl (a less-used German name for the mushroom), literally translating to "toad-stool".

However, there are some concern of whether these wild mushrooms are safe to eat or not. Hence, the term **Mushroom Poisoning** became popular. Mushroom poisoning refers to harmful effects from ingestion of toxic substances present in a mushroom. These symptoms can vary from slight gastrointestinal discomfort to death in about 10 days. The toxins present are secondary metabolites produced by the fungus. Mushroom poisoning is usually the result of ingestion of wild mushrooms after misidentification of a toxic mushroom as an edible species. The most common reason for this misidentification is close resemblance in terms of colour and general morphology of the toxic mushrooms species with edible species. To prevent mushroom poisoning, mushroom gatherers familiarize themselves with the mushrooms they intend to collect, as well as with any similar-looking toxic species.

**This simple web app(Flask) project is made to show a prediction whether a mushroom is poisonous or not.**

This project falls into the topic of classification with machine learning model.
The dataset used for model training is obtainable from [kaggle](https://www.kaggle.com/uciml/mushroom-classification) with original research data from several universities in [UCI Machine Learning Repositories](https://archive.ics.uci.edu/ml/datasets/Mushroom). Random Forest Classifier was chosen to process all the mushroom data up until the prediction. The front-end interface was made with the help of [Flask](https://palletsprojects.com/p/flask/) and other miscellaneous things.

As the preparation, the dataset were all cleaned, and visualised with the help of [Pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/). The model were dumped using joblib by *MushroomsMachineLearning.ipynb*. Further on, the barplot which is available on the prediction page were made by utilizing [Seaborn](https://seaborn.pydata.org/)


## Data Visualisation and Modelling Process

# Import Libraries & Data


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
```


```python
dfmushroom = pd.read_csv('mushrooms.csv')
dfmushroom
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <td>1</td>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <td>2</td>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <td>3</td>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <td>4</td>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <td>8119</td>
      <td>e</td>
      <td>k</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <td>8120</td>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <td>8121</td>
      <td>e</td>
      <td>f</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>b</td>
      <td>c</td>
      <td>l</td>
    </tr>
    <tr>
      <td>8122</td>
      <td>p</td>
      <td>k</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>y</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>b</td>
      <td>...</td>
      <td>k</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>w</td>
      <td>v</td>
      <td>l</td>
    </tr>
    <tr>
      <td>8123</td>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>a</td>
      <td>c</td>
      <td>b</td>
      <td>y</td>
      <td>...</td>
      <td>s</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>o</td>
      <td>p</td>
      <td>o</td>
      <td>c</td>
      <td>l</td>
    </tr>
  </tbody>
</table>
<p>8124 rows × 23 columns</p>
</div>



## Attribute Information: 
1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
4. bruises?: bruises=t,no=f
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
6. gill-attachment: attached=a,descending=d,free=f,notched=n
7. gill-spacing: close=c,crowded=w,distant=d
8. gill-size: broad=b,narrow=n
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
10. stalk-shape: enlarging=e,tapering=t
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing data (NaN)=?
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
16. veil-type: partial=p,universal=u
17. veil-color: brown=n,orange=o,white=w,yellow=y
18. ring-number: none=n,one=o,two=t
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
23. class: edible=e,poisonous=p


```python
dfmushroom['class'] = dfmushroom['class'].apply(
    lambda x: 'poisonous' if x == 'p' else (
        'edible' if x == 'e' else np.NaN)
)
dfmushroom['cap-shape'] = dfmushroom['cap-shape'].apply(
    lambda x: 'bell' if x == 'b' else (
        'conical' if x == 'c' else (
        'convex' if x == 'x' else (
        'flat' if x == 'f' else (
        'knobbed' if x == 'k' else(
        'sunken' if x == 's' else np.NaN)))))
)
dfmushroom['cap-surface'] = dfmushroom['cap-surface'].apply(
    lambda x: 'fibrous' if x == 'f' else (
    'grooves' if x == 'g' else(
    'scaly' if x == 'y' else(
    'smooth' if x == 's' else np.NaN)))
)
dfmushroom['cap-color'] = dfmushroom['cap-color'].apply(
    lambda x: 'brown' if x == 'n' else(
    'buff' if x == 'b' else(
    'cinnamon' if x == 'c' else(
    'gray' if x == 'g' else(
    'green' if x == 'r' else(
    'pink' if x == 'p' else(
    'purple' if x == 'u' else(
    'red' if x == 'e' else(
    'white' if x == 'w' else(
    'yellow' if x == 'y' else np.NaN
    )))))))))
)
dfmushroom['bruises'] = dfmushroom['bruises'].apply(
    lambda x: 'bruises' if x == 't' else(
    'no' if x == 'f' else np.NaN)
)
dfmushroom['odor'] = dfmushroom['odor'].apply(
    lambda x: 'almond' if x == 'a' else(
    'anise' if x == 'l' else(
    'creosote' if x == 'c' else(
    'fishy' if x == 'y' else(
    'foul' if x == 'f' else(
    'musty' if x == 'm' else(
    'none' if x == 'n' else(
    'pungent' if x == 'p' else(
    'spicy' if x == 's' else np.NaN
    ))))))))
)
dfmushroom['gill-attachment'] = dfmushroom['gill-attachment'].apply(
    lambda x: 'attached' if x == 'a' else(
    'descending' if x == 'd' else(
    'free' if x == 'f' else(
    'notched' if x == 'n' else np.NaN
    )))
)
dfmushroom['gill-spacing'] = dfmushroom['gill-spacing'].apply(
    lambda x: 'close' if x == 'c' else(
    'crowded' if x == 'w' else(
    'distant' if x == 'd' else np.NaN))
)
dfmushroom['gill-size'] = dfmushroom['gill-size'].apply(
    lambda x: 'broad' if x == 'b' else(
    'narrow' if x == 'n' else np.NaN)
)
dfmushroom['gill-color'] = dfmushroom['gill-color'].apply(
    lambda x: 'black' if x == 'k' else(
    'brown' if x == 'n' else(
    'buff' if x == 'b' else(
    'chocolate' if x == 'h' else(
    'gray' if x == 'g' else(
    'green' if x == 'r' else(
    'orange' if x == 'o' else(
    'pink' if x == 'p' else(
    'purple' if x == 'u' else(
    'red' if x == 'e' else(
    'white' if x == 'w' else(
    'yellow' if x == 'y' else np.NaN
    )))))))))))
)
dfmushroom['stalk-shape'] = dfmushroom['stalk-shape'].apply(
    lambda x: 'enlarging' if x == 'e' else(
    'tapering' if x == 't' else np.NaN)
)
dfmushroom['stalk-root'] = dfmushroom['stalk-root'].apply(
    lambda x: 'bulbous' if x == 'b' else(
    'club' if x == 'c' else(
    'cup' if x == 'u' else(
    'equal' if x == 'e' else(
    'rhizomorphs' if x == 'z' else(
    'rooted' if x == 'r' else np.NaN
    )))))
)
dfmushroom['stalk-surface-above-ring'] = dfmushroom['stalk-surface-above-ring'].apply(
    lambda x: 'fibrous' if x == 'f' else(
    'scaly' if x == 'y' else(
    'silky' if x == 'k' else(
    'smooth' if x == 's' else np.NaN)))
)
dfmushroom['stalk-surface-below-ring'] = dfmushroom['stalk-surface-below-ring'].apply(
    lambda x: 'fibrous' if x == 'f' else(
    'scaly' if x == 'y' else(
    'silky' if x == 'k' else(
    'smooth' if x == 's' else np.NaN)))
)
dfmushroom['stalk-color-above-ring'] = dfmushroom['stalk-color-above-ring'].apply(
    lambda x: 'brown' if x == 'n' else(
    'buff' if x == 'b' else(
    'cinnamon' if x == 'c' else(
    'gray' if x == 'g' else(
    'orange' if x == 'o' else(
    'pink' if x == 'p' else(
    'red' if x == 'e' else(
    'white' if x == 'w' else(
    'yellow' if x == 'y' else np.NaN
    ))))))))
)
dfmushroom['stalk-color-below-ring'] = dfmushroom['stalk-color-below-ring'].apply(
    lambda x: 'brown' if x == 'n' else(
    'buff' if x == 'b' else(
    'cinnamon' if x == 'c' else(
    'gray' if x == 'g' else(
    'orange' if x == 'o' else(
    'pink' if x == 'p' else(
    'red' if x == 'e' else(
    'white' if x == 'w' else(
    'yellow' if x == 'y' else np.NaN
    ))))))))
)
dfmushroom['veil-type'] = dfmushroom['veil-type'].apply(
    lambda x: 'partial' if x == 'p' else(
    'universal' if x == 'u' else np.NaN)
)
dfmushroom['veil-color'] = dfmushroom['veil-color'].apply(
    lambda x: 'brown' if x == 'n' else(
    'orange' if x == 'o' else(
    'white' if x == 'w' else(
    'yellow' if x == 'y' else np.NaN)))
)
dfmushroom['ring-number'] = dfmushroom['ring-number'].apply(
    lambda x: 'none' if x == 'n' else(
    'one' if x == 'o' else(
    'two' if x == 't' else np.NaN))
)
dfmushroom['ring-type'] = dfmushroom['ring-type'].apply(
    lambda x: 'cobwebby' if x == 'c' else(
    'evanescent' if x == 'e' else(
    'flaring' if x == 'f' else(
    'large' if x == 'l' else(
    'none' if x == 'n' else(
    'pendant' if x == 'p' else(
    'sheathing' if x == 's' else(
    'zone' if x == 'z' else np.NaN)))))))
)
dfmushroom['spore-print-color'] = dfmushroom['spore-print-color'].apply(
    lambda x: 'black' if x == 'k' else(
    'brown' if x == 'n' else(
    'buff' if x == 'b' else(
    'chocolate' if x == 'h' else(
    'green' if x == 'r' else(
    'orange' if x == 'o' else(
    'purple' if x == 'u' else(
    'white' if x == 'w' else(
    'yellow' if x == 'y' else np.NaN
    ))))))))
)
dfmushroom['population'] = dfmushroom['population'].apply(
    lambda x: 'abundant' if x == 'a' else(
    'clustered' if x == 'c' else(
    'numerous' if x == 'n' else(
    'scattered' if x == 's' else(
    'several' if x == 'v' else(
    'solitary' if x == 'y' else np.NaN
    )))))
)
dfmushroom['habitat'] = dfmushroom['habitat'].apply(
    lambda x: 'grasses' if x == 'g' else(
    'leaves' if x == 'l' else(
    'meadows' if x == 'm' else(
    'paths' if x == 'p' else(
    'urban' if x == 'u' else(
    'waste' if x == 'w' else(
    'woods' if x == 'd' else np.NaN))))))
)
dfmushroom
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>poisonous</td>
      <td>convex</td>
      <td>smooth</td>
      <td>brown</td>
      <td>bruises</td>
      <td>pungent</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>black</td>
      <td>scattered</td>
      <td>urban</td>
    </tr>
    <tr>
      <td>1</td>
      <td>edible</td>
      <td>convex</td>
      <td>smooth</td>
      <td>yellow</td>
      <td>bruises</td>
      <td>almond</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>brown</td>
      <td>numerous</td>
      <td>grasses</td>
    </tr>
    <tr>
      <td>2</td>
      <td>edible</td>
      <td>bell</td>
      <td>smooth</td>
      <td>white</td>
      <td>bruises</td>
      <td>anise</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>brown</td>
      <td>numerous</td>
      <td>meadows</td>
    </tr>
    <tr>
      <td>3</td>
      <td>poisonous</td>
      <td>convex</td>
      <td>scaly</td>
      <td>white</td>
      <td>bruises</td>
      <td>pungent</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>black</td>
      <td>scattered</td>
      <td>urban</td>
    </tr>
    <tr>
      <td>4</td>
      <td>edible</td>
      <td>convex</td>
      <td>smooth</td>
      <td>gray</td>
      <td>no</td>
      <td>none</td>
      <td>free</td>
      <td>crowded</td>
      <td>broad</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>evanescent</td>
      <td>brown</td>
      <td>abundant</td>
      <td>grasses</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <td>8119</td>
      <td>edible</td>
      <td>knobbed</td>
      <td>smooth</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>attached</td>
      <td>close</td>
      <td>broad</td>
      <td>yellow</td>
      <td>...</td>
      <td>smooth</td>
      <td>orange</td>
      <td>orange</td>
      <td>partial</td>
      <td>orange</td>
      <td>one</td>
      <td>pendant</td>
      <td>buff</td>
      <td>clustered</td>
      <td>leaves</td>
    </tr>
    <tr>
      <td>8120</td>
      <td>edible</td>
      <td>convex</td>
      <td>smooth</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>attached</td>
      <td>close</td>
      <td>broad</td>
      <td>yellow</td>
      <td>...</td>
      <td>smooth</td>
      <td>orange</td>
      <td>orange</td>
      <td>partial</td>
      <td>brown</td>
      <td>one</td>
      <td>pendant</td>
      <td>buff</td>
      <td>several</td>
      <td>leaves</td>
    </tr>
    <tr>
      <td>8121</td>
      <td>edible</td>
      <td>flat</td>
      <td>smooth</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>attached</td>
      <td>close</td>
      <td>broad</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>orange</td>
      <td>orange</td>
      <td>partial</td>
      <td>orange</td>
      <td>one</td>
      <td>pendant</td>
      <td>buff</td>
      <td>clustered</td>
      <td>leaves</td>
    </tr>
    <tr>
      <td>8122</td>
      <td>poisonous</td>
      <td>knobbed</td>
      <td>scaly</td>
      <td>brown</td>
      <td>no</td>
      <td>fishy</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>buff</td>
      <td>...</td>
      <td>silky</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>evanescent</td>
      <td>white</td>
      <td>several</td>
      <td>leaves</td>
    </tr>
    <tr>
      <td>8123</td>
      <td>edible</td>
      <td>convex</td>
      <td>smooth</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>attached</td>
      <td>close</td>
      <td>broad</td>
      <td>yellow</td>
      <td>...</td>
      <td>smooth</td>
      <td>orange</td>
      <td>orange</td>
      <td>partial</td>
      <td>orange</td>
      <td>one</td>
      <td>pendant</td>
      <td>orange</td>
      <td>clustered</td>
      <td>leaves</td>
    </tr>
  </tbody>
</table>
<p>8124 rows × 23 columns</p>
</div>



# EDA


```python
dfmushroom.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 8124 entries, 0 to 8123
    Data columns (total 23 columns):
    class                       8124 non-null object
    cap-shape                   8124 non-null object
    cap-surface                 8124 non-null object
    cap-color                   8124 non-null object
    bruises                     8124 non-null object
    odor                        8124 non-null object
    gill-attachment             8124 non-null object
    gill-spacing                8124 non-null object
    gill-size                   8124 non-null object
    gill-color                  8124 non-null object
    stalk-shape                 8124 non-null object
    stalk-root                  3908 non-null object
    stalk-surface-above-ring    8124 non-null object
    stalk-surface-below-ring    8124 non-null object
    stalk-color-above-ring      8124 non-null object
    stalk-color-below-ring      8124 non-null object
    veil-type                   8124 non-null object
    veil-color                  8124 non-null object
    ring-number                 8124 non-null object
    ring-type                   8124 non-null object
    spore-print-color           8124 non-null object
    population                  8124 non-null object
    habitat                     8124 non-null object
    dtypes: object(23)
    memory usage: 1.4+ MB
    


```python
dfmushroom.describe()
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>...</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <td>top</td>
      <td>edible</td>
      <td>convex</td>
      <td>scaly</td>
      <td>brown</td>
      <td>no</td>
      <td>none</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>buff</td>
      <td>...</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>partial</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>white</td>
      <td>several</td>
      <td>woods</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>...</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 23 columns</p>
</div>




```python
listItem = []
for col in dfmushroom.columns :
    listItem.append([col, dfmushroom[col].dtype, dfmushroom[col].isna().sum(), round((dfmushroom[col].isna().sum()/len(dfmushroom[col])) * 100,2),
                    dfmushroom[col].nunique(), list(dfmushroom[col].unique()[:2])]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
dfDesc
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
      <th>dataFeatures</th>
      <th>dataType</th>
      <th>null</th>
      <th>nullPct</th>
      <th>unique</th>
      <th>uniqueSample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>class</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[poisonous, edible]</td>
    </tr>
    <tr>
      <td>1</td>
      <td>cap-shape</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>[convex, bell]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>cap-surface</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>[smooth, scaly]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>cap-color</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>10</td>
      <td>[brown, yellow]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>bruises</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[bruises, no]</td>
    </tr>
    <tr>
      <td>5</td>
      <td>odor</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>[pungent, almond]</td>
    </tr>
    <tr>
      <td>6</td>
      <td>gill-attachment</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[free, attached]</td>
    </tr>
    <tr>
      <td>7</td>
      <td>gill-spacing</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[close, crowded]</td>
    </tr>
    <tr>
      <td>8</td>
      <td>gill-size</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[narrow, broad]</td>
    </tr>
    <tr>
      <td>9</td>
      <td>gill-color</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>12</td>
      <td>[black, brown]</td>
    </tr>
    <tr>
      <td>10</td>
      <td>stalk-shape</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>2</td>
      <td>[enlarging, tapering]</td>
    </tr>
    <tr>
      <td>11</td>
      <td>stalk-root</td>
      <td>object</td>
      <td>4216</td>
      <td>51.9</td>
      <td>6</td>
      <td>[equal, club]</td>
    </tr>
    <tr>
      <td>12</td>
      <td>stalk-surface-above-ring</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>[smooth, fibrous]</td>
    </tr>
    <tr>
      <td>13</td>
      <td>stalk-surface-below-ring</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>[smooth, fibrous]</td>
    </tr>
    <tr>
      <td>14</td>
      <td>stalk-color-above-ring</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>[white, gray]</td>
    </tr>
    <tr>
      <td>15</td>
      <td>stalk-color-below-ring</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>[white, pink]</td>
    </tr>
    <tr>
      <td>16</td>
      <td>veil-type</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>[partial]</td>
    </tr>
    <tr>
      <td>17</td>
      <td>veil-color</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>4</td>
      <td>[white, brown]</td>
    </tr>
    <tr>
      <td>18</td>
      <td>ring-number</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>3</td>
      <td>[one, two]</td>
    </tr>
    <tr>
      <td>19</td>
      <td>ring-type</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>5</td>
      <td>[pendant, evanescent]</td>
    </tr>
    <tr>
      <td>20</td>
      <td>spore-print-color</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>9</td>
      <td>[black, brown]</td>
    </tr>
    <tr>
      <td>21</td>
      <td>population</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>6</td>
      <td>[scattered, numerous]</td>
    </tr>
    <tr>
      <td>22</td>
      <td>habitat</td>
      <td>object</td>
      <td>0</td>
      <td>0.0</td>
      <td>7</td>
      <td>[urban, grasses]</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(dfmushroom.isnull(),yticklabels=False,cbar=False,cmap='viridis')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586b943e88>




![png](output_9_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',data=dfmushroom,palette='RdBu_r')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586ceefbc8>




![png](output_10_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='cap-shape',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586d46d288>




![png](output_11_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='cap-surface',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586d46a488>




![png](output_12_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='cap-color',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586de41e88>




![png](output_13_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='bruises',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586de8aac8>




![png](output_14_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='odor',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586df05c48>




![png](output_15_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='gill-attachment',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586df7ddc8>




![png](output_16_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='gill-spacing',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e03ac08>




![png](output_17_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='gill-size',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586cf2d4c8>




![png](output_18_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='gill-color',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e0f7248>




![png](output_19_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='stalk-shape',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e1a3a48>




![png](output_20_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='stalk-root',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e1f1888>




![png](output_21_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='stalk-surface-above-ring',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e2860c8>




![png](output_22_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='stalk-surface-below-ring',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e2f4ec8>




![png](output_23_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='stalk-color-above-ring',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e3609c8>




![png](output_24_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='stalk-color-below-ring',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e3db348>




![png](output_25_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='veil-type',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e491ac8>




![png](output_26_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='veil-color',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586e4f70c8>




![png](output_27_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='ring-number',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586f53c0c8>




![png](output_28_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='ring-type',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586f5a3288>




![png](output_29_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='spore-print-color',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586f60cb08>




![png](output_30_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='population',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586f6a6a08>




![png](output_31_1.png)



```python
sns.set_style('whitegrid')
sns.countplot(x='class',hue='habitat',data=dfmushroom)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2586b50a4c8>




![png](output_32_1.png)



```python
dfmushroom.drop(['stalk-root','veil-type'],axis=1,inplace=True)
```


```python
dfmushroom.head()
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>poisonous</td>
      <td>convex</td>
      <td>smooth</td>
      <td>brown</td>
      <td>bruises</td>
      <td>pungent</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>black</td>
      <td>scattered</td>
      <td>urban</td>
    </tr>
    <tr>
      <td>1</td>
      <td>edible</td>
      <td>convex</td>
      <td>smooth</td>
      <td>yellow</td>
      <td>bruises</td>
      <td>almond</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>brown</td>
      <td>numerous</td>
      <td>grasses</td>
    </tr>
    <tr>
      <td>2</td>
      <td>edible</td>
      <td>bell</td>
      <td>smooth</td>
      <td>white</td>
      <td>bruises</td>
      <td>anise</td>
      <td>free</td>
      <td>close</td>
      <td>broad</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>brown</td>
      <td>numerous</td>
      <td>meadows</td>
    </tr>
    <tr>
      <td>3</td>
      <td>poisonous</td>
      <td>convex</td>
      <td>scaly</td>
      <td>white</td>
      <td>bruises</td>
      <td>pungent</td>
      <td>free</td>
      <td>close</td>
      <td>narrow</td>
      <td>brown</td>
      <td>...</td>
      <td>smooth</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>white</td>
      <td>one</td>
      <td>pendant</td>
      <td>black</td>
      <td>scattered</td>
      <td>urban</td>
    </tr>
    <tr>
      <td>4</td>
      <td>edible</td>
      <td>convex</td>
      <td>smooth</td>
      <td>gray</td>
      <td>no</td>
      <td>none</td>
      <td>free</td>
      <td>crowded</td>
      <td>broad</td>
      <td>black</td>
      <td>...</td>
      <td>smooth</td>
      <td>smooth</td>
      <td>white</td>
      <td>white</td>
      <td>white</td>
      <td>one</td>
      <td>evanescent</td>
      <td>brown</td>
      <td>abundant</td>
      <td>grasses</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# Label Encoding


```python
from sklearn.preprocessing import LabelEncoder
```


```python
labelencoder=LabelEncoder()
for col in dfmushroom.columns:
    dfmushroom[col] = labelencoder.fit_transform(dfmushroom[col])
dfmushroom.head()
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
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-above-ring</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



# Train Test Split


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(dfmushroom.drop('class',axis=1), 
                                                    dfmushroom['class'], test_size=0.30, 
                                                    random_state=101)
```

# Train The Model


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
rfc = RandomForestClassifier(n_estimators=100, random_state=50)
rfc.fit(X_train,y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=50, verbose=0,
                           warm_start=False)



# Feature Importances


```python
coef1 = pd.Series(rfc.feature_importances_, X_train.columns).sort_values(ascending = False)
coef1.plot(kind='bar', title='Feature Importances')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x258702f4c08>




![png](output_45_1.png)


# Evaluate The Model

### Training Data


```python
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, auc, log_loss, roc_auc_score, f1_score
```


```python
predictTrain = rfc.predict(X_train)
predictTrain
```




    array([0, 0, 0, ..., 1, 0, 1])




```python
len(predictTrain)
```




    5686




```python
sum(predictTrain)
```




    2752




```python
con = pd.DataFrame(data=confusion_matrix(y_train,predictTrain), columns=['P No', 'P Yes'], index=['A No', 'A Yes']);
print(con)
```

           P No  P Yes
    A No   2934      0
    A Yes     0   2752
    


```python
print(classification_report(y_train,predictTrain))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      2934
               1       1.00      1.00      1.00      2752
    
        accuracy                           1.00      5686
       macro avg       1.00      1.00      1.00      5686
    weighted avg       1.00      1.00      1.00      5686
    
    


```python
print('Accuracy : ' + str(accuracy_score(y_train,predictTrain)))
```

    Accuracy : 1.0
    


```python
predictProbTrain = rfc.predict_proba(X_train)
predictProbTrain
```




    array([[1., 0.],
           [1., 0.],
           [1., 0.],
           ...,
           [0., 1.],
           [1., 0.],
           [0., 1.]])




```python
# calculate the fpr and tpr for all thresholds of the classification
preds = predictProbTrain[:,1]
fpr, tpr, threshold = roc_curve(y_train, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```


![png](output_56_0.png)



```python
print(len(fpr))
print(len(tpr))
print(len(threshold))
```

    9
    9
    9
    


```python
print(fpr[-5:])
print(tpr[-5:])
print(threshold[-5:])
```

    [0.00000000e+00 3.40831629e-04 1.70415815e-03 7.49829584e-03
     1.00000000e+00]
    [1. 1. 1. 1. 1.]
    [0.92 0.05 0.02 0.01 0.  ]
    


```python
log_loss(y_train, predictProbTrain[:,1])
```




    0.00012135526821815287



### Testing Data


```python
predictTest = rfc.predict(X_test)
predictTest
```




    array([1, 0, 1, ..., 0, 0, 1])




```python
sum(predictTest)
```




    1164




```python
con = pd.DataFrame(data=confusion_matrix(y_test,predictTest), columns=['P No', 'P Yes'], index=['A No', 'A Yes']);
print(con)
```

           P No  P Yes
    A No   1274      0
    A Yes     0   1164
    


```python
print(classification_report(y_test,predictTest))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      1274
               1       1.00      1.00      1.00      1164
    
        accuracy                           1.00      2438
       macro avg       1.00      1.00      1.00      2438
    weighted avg       1.00      1.00      1.00      2438
    
    


```python
print('Accuracy : ' + str(accuracy_score(y_test,predictTest)))
```

    Accuracy : 1.0
    


```python
predictProbTest = rfc.predict_proba(X_test)
predictProbTest
```




    array([[0., 1.],
           [1., 0.],
           [0., 1.],
           ...,
           [1., 0.],
           [1., 0.],
           [0., 1.]])




```python
# calculate the fpr and tpr for all thresholds of the classification
preds = predictProbTest[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = {}'.format(round(roc_auc,2)))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```


![png](output_67_0.png)



```python
log_loss(y_test, predictProbTest[:,1])
```




    0.00032508551262899976



# K Fold


```python
def calc_train_error(X_train, y_train, model):
#     '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    predictProba = model.predict_proba(X_train)
    accuracy = accuracy_score(y_train, predictions)
    f1 = f1_score(y_train, predictions, average='macro')
    roc_auc = roc_auc_score(y_train, predictProba[:,1])
    logloss = log_loss(y_train,predictProba)
    report = classification_report(y_train, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'logloss': logloss
    }
    
def calc_validation_error(X_test, y_test, model):
#     '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    predictProba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')
    roc_auc = roc_auc_score(y_test, predictProba[:,1])
    logloss = log_loss(y_test,predictProba)
    report = classification_report(y_test, predictions)
    return { 
        'report': report, 
        'f1' : f1, 
        'roc': roc_auc, 
        'accuracy': accuracy,
        'logloss': logloss
    }
    
def calc_metrics(X_train, y_train, X_test, y_test, model):
#     '''fits model and returns the in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error = calc_train_error(X_train, y_train, model)
    validation_error = calc_validation_error(X_test, y_test, model)
    return train_error, validation_error
```


```python
from sklearn.model_selection import KFold

K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=42)
```


```python
data = dfmushroom.drop('class',axis=1)
target = dfmushroom['class']
```


```python
train_errors = []
validation_errors = []
for train_index, val_index in kf.split(data, target):
    
    # split data
    X_train, X_val = data.iloc[train_index], data.iloc[val_index]
    y_train, y_val = target.iloc[train_index], target.iloc[val_index]
    
    print(len(X_val), (len(X_train) + len(X_val)))

    # instantiate model
    rfc = RandomForestClassifier(n_estimators=100)

    #calculate errors
    train_error, val_error = calc_metrics(X_train, y_train, X_val, y_val, rfc)

    # append to appropriate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
```

    813 8124
    813 8124
    813 8124
    813 8124
    812 8124
    812 8124
    812 8124
    812 8124
    812 8124
    812 8124
    


```python
listItem = []

for tr,val in zip(train_errors,validation_errors) :
    listItem.append([tr['accuracy'],val['accuracy'],tr['roc'],val['roc'],tr['f1'],val['f1'],
                     tr['logloss'],val['logloss']])
    
dfEvaluate = pd.DataFrame(listItem, 
                    columns=['Train Accuracy', 
                            'Test Accuracy', 
                            'Train ROC AUC', 
                            'Test ROC AUC', 
                            'Train F1 Score',
                            'Test F1 Score',
                            'Train Log Loss',
                            'Test Log Loss'])
dfEvaluate
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
      <th>Train Accuracy</th>
      <th>Test Accuracy</th>
      <th>Train ROC AUC</th>
      <th>Test ROC AUC</th>
      <th>Train F1 Score</th>
      <th>Test F1 Score</th>
      <th>Train Log Loss</th>
      <th>Test Log Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000074</td>
      <td>0.000186</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000073</td>
      <td>0.000136</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000072</td>
      <td>0.000325</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000092</td>
      <td>0.000374</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000062</td>
      <td>0.000099</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000069</td>
      <td>0.000211</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000087</td>
      <td>0.000199</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000074</td>
      <td>0.000162</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000065</td>
      <td>0.000112</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000081</td>
      <td>0.000186</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfEvaluate.mean()
```




    Train Accuracy    1.000000
    Test Accuracy     1.000000
    Train ROC AUC     1.000000
    Test ROC AUC      1.000000
    Train F1 Score    1.000000
    Test F1 Score     1.000000
    Train Log Loss    0.000075
    Test Log Loss     0.000199
    dtype: float64




```python
for rep in validation_errors :
    print(rep['report'])
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       433
               1       1.00      1.00      1.00       380
    
        accuracy                           1.00       813
       macro avg       1.00      1.00      1.00       813
    weighted avg       1.00      1.00      1.00       813
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       410
               1       1.00      1.00      1.00       403
    
        accuracy                           1.00       813
       macro avg       1.00      1.00      1.00       813
    weighted avg       1.00      1.00      1.00       813
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       414
               1       1.00      1.00      1.00       399
    
        accuracy                           1.00       813
       macro avg       1.00      1.00      1.00       813
    weighted avg       1.00      1.00      1.00       813
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       423
               1       1.00      1.00      1.00       390
    
        accuracy                           1.00       813
       macro avg       1.00      1.00      1.00       813
    weighted avg       1.00      1.00      1.00       813
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       407
               1       1.00      1.00      1.00       405
    
        accuracy                           1.00       812
       macro avg       1.00      1.00      1.00       812
    weighted avg       1.00      1.00      1.00       812
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       410
               1       1.00      1.00      1.00       402
    
        accuracy                           1.00       812
       macro avg       1.00      1.00      1.00       812
    weighted avg       1.00      1.00      1.00       812
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       408
               1       1.00      1.00      1.00       404
    
        accuracy                           1.00       812
       macro avg       1.00      1.00      1.00       812
    weighted avg       1.00      1.00      1.00       812
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       457
               1       1.00      1.00      1.00       355
    
        accuracy                           1.00       812
       macro avg       1.00      1.00      1.00       812
    weighted avg       1.00      1.00      1.00       812
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       415
               1       1.00      1.00      1.00       397
    
        accuracy                           1.00       812
       macro avg       1.00      1.00      1.00       812
    weighted avg       1.00      1.00      1.00       812
    
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       431
               1       1.00      1.00      1.00       381
    
        accuracy                           1.00       812
       macro avg       1.00      1.00      1.00       812
    weighted avg       1.00      1.00      1.00       812
    
    

# Dumping Model Into Joblib File


```python
import joblib as jb
```


```python
jb.dump(rfc,'modelmushroom')
```




    ['modelmushroom']




```python

```


## Interface (Flask)

**How to run the Web App**
1. Clone/download this repo.
2. Open *appMushroom.py*.
3. To include the data export to MySQL, make sure you have the same account profile and database, or just delete lines that have to do with MySQL syntax.
4. Run *appMushroom.py*.
5. The server will run on 127.0.0.1:5000 which bring you to the login page:
    ![Login](screenshots/LoginPage.png)
    This page serves as the landing page with a few buttons and inputs such as:
    - Username (input bar): enter registered username
    - Password (input bar): enter registered password
    - Login (button): button to log in to the specific user database
    - Signup (button): button to sign up new account

    Notes: signup are required to create a new account before login with existing account. Whenever a person create new account, a new table in 'mushroom' MySQL database is created. MySQL was used only to store prediction data which the user has input through homepage on Predicting Category. *404 Error* also included on the template.
    ![Signup](screenshots/Signup.png)

6. After Signing up/Login process, the user will be directed to the main page which consists of some option pane and menu navigation bar:
    ![Main](screenshots/MainMenu.png)

    In this app, there are several pages can found in NavBar:
    
    ![NavBar](screenshots/NavBar.png)

    1. Poisonous Prediction (Main Menu)
        The main menu is prediction menu. The corresponding user can try to predict whether the mushroom is poisonous or not.

        If a mushroom is indicated non-poisonous, then the following page will be prompted:
        ![Safe](screenshots/Safe.png)
        If a mushroom is indicated poisonous, then the following page will be prompted:
        ![Danger](screenshots/Unsafe.png)
    2. About Mushrooms in general
        In this page, user can learn some small insight of mushrooms and general world of Fungus (Biological Fungi Kingdom)
        ![About1](screenshots/About1.png)
        ![About2](screenshots/About2.png)
        ![About3](screenshots/About3.png)
    3. Test History
        In this page, user can check any predicting activity that has occured
        ![History](screenshots/History.png)
        Whenever a user try to predict a mushroom, the input will be recorded into the respective user data in MySQL. This can help the user to get previous information. This page load the MySQL table into html.

    

___
### Thank you for your attention. I hope you enjoy my debrief about this small project. 😊

#### ✉ _andrew.edbert@yahoo.com_