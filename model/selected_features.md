# Selected features

```
Index(['day', 'CLC1(t-20)', 'Topic5(t-19)', 'CLC4(t-19)', 'CLC1(t-19)',
       'Topic5(t-18)', 'CLC4(t-18)', 'Decay_Topic1(t-17)', 'CLC1(t-17)',
       'Decay_Topic3(t-14)', 'Topic1(t-12)', 'Decay_Topic5(t-8)', 'CLC1(t-3)',
       'Subjectivity(t-2)', 'Topic5(t-2)', 'CLC1(t-2)', 'Topic3(t-1)',
       'CLC3(t-1)', 'CLC2(t-1)', 'CLC1(t-1)'],
      dtype='object')
```

```python
# stepwise ridge 20
Index(['day', 'CLC1(t-20)', 'Topic5(t-19)', 'CLC4(t-19)', 'CLC1(t-19)',
       'Topic5(t-18)', 'CLC4(t-18)', 'Decay_Topic1(t-17)', 'CLC1(t-17)',
       'Decay_Topic3(t-14)', 'Topic1(t-12)', 'Decay_Topic5(t-8)', 'CLC1(t-3)',
       'Subjectivity(t-2)', 'Topic5(t-2)', 'CLC1(t-2)', 'Topic3(t-1)',
       'CLC3(t-1)', 'CLC2(t-1)', 'CLC1(t-1)'],
      dtype='object')
```

```py
Index(['day', 'CLC1(t-20)', 'Topic5(t-19)', 'CLC4(t-19)', 'CLC1(t-19)',
       'Topic5(t-18)', 'CLC4(t-18)', 'Decay_Topic1(t-17)', 'CLC1(t-17)',
       'Decay_Topic3(t-14)', 'Topic1(t-12)', 'Decay_Topic5(t-8)', 'CLC1(t-3)',
       'Subjectivity(t-2)', 'Topic5(t-2)', 'CLC1(t-2)', 'Topic3(t-1)',
       'CLC3(t-1)', 'CLC2(t-1)', 'CLC1(t-1)'],
      dtype='object')
```

lag 25
Lasso RFE 10
0.5065304794570896,
0.6792731495603385,
0.008987044761449986
0.9936145945456947

```py
Index(['Decay_Topic2(t-25)', 'Decay_Topic2(t-22)', 'Decay_Topic2(t-15)',
       'Decay_Topic2(t-4)', 'Decay_Topic2(t-3)', 'War_Threats(t-3)',
       'Terrorist_Threats(t-3)', 'Decay_Topic2(t-1)',
       'Decay_Terrorist_Acts (t-1)', 'CLC1(t-1)'],
      dtype='object')
```

lag 25
Granger 10
0.5059846964184894,
0.6788853833928368,
0.008977540471098897
0.993621882740354
```py
Index(['Decay_Topic4(t-23)', 'Decay_Topic4(t-22)', 'Decay_Topic4(t-21)',
       'Decay_Terrorist_Acts (t-23)', 'Decay_Terrorist_Acts (t-24)',
       'Decay_Terrorist_Acts (t-13)', 'Decay_Terrorist_Acts (t-12)',
       'Decay_Terrorist_Acts (t-11)', 'Decay_Terrorist_Acts (t-22)',
       'Decay_Terrorist_Acts (t-21)'],
      dtype='object')
```