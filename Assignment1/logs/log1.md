Run the command:

```
date '+%X'
python3 main.py --foreign ../data/combination.f --source ../data/combination.e --wa ../data/test.wa.nonullalign --ibm IBM-M2-1 --iter-1 1 --iter-2 1 --export weights
date '+%X'
```

```
04:24:48 PM
IBM model 1
Resetting weights
Perplexity = 2309252.1051808055, Log-likelihood = 159788.52464723182
Recall = 0.03813769192669639, Precision = 0.06917092146851446, AER = 0.9410469667318982
Model 1 instance: <ibm.Model object at 0x7fa48b7ad630>
IBM model 2 initialized by IBM-M1
Perplexity = 2308670.8588592764, Log-likelihood = 160369.77096875323
Recall = 0.03813769192669639, Precision = 0.06908294818778886, AER = 0.9411092985318108
Exporting weights...
04:50:22 PM
```

Run the command:

```
date '+%X'
python3 main.py --foreign ../data/combination.f --source ../data/combination.e --wa ../data/test.wa.nonullalign --ibm IBM-M1 --iter-1 1 --export weights
date '+%X'
```

```
04:57:29 PM
IBM model 1
Resetting weights
Perplexity = 2309705.1642425186, Log-likelihood = 159335.4655855216
Recall = 0.03838533927686974, Precision = 0.07170875214513361, AER = 0.9393243686454575
Model 1 instance: <ibm.Model object at 0x7f7fe27316a0>
Exporting weights...
05:04:58 PM
```

| Model | Time consumption |
|-------|------------------|
|IBM-M1 | 00:07:29         |
|IBM-M2 | 00:18:05         |
