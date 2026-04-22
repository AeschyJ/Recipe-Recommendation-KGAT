# Experiment Logs (Simplified)

## File: output\simplified_for_llm\logs\baseline\bprmf_20260408_173547.txt

```text
Training started. Args: Namespace(model='BPR-MF', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline')
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.6719
Epoch 1 Evaluation - HR@[10,20,50]: [0.3738, 0.4746, 0.6629] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 2 Avg Loss: 0.5871
Epoch 2 Evaluation - HR@[10,20,50]: [0.4868, 0.6086, 0.7962] | Precision@[10,20,50]: [0.0487, 0.0304, 0.0159] | NDCG@[10,20,50]: [0.3219, 0.3526, 0.3898]
Epoch 3 Avg Loss: 0.5413
Epoch 3 Evaluation - HR@[10,20,50]: [0.4250, 0.5313, 0.7102] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 4 Avg Loss: 0.5159
Epoch 4 Evaluation - HR@[10,20,50]: [0.5344, 0.6660, 0.8484] | Precision@[10,20,50]: [0.0534, 0.0333, 0.0170] | NDCG@[10,20,50]: [0.3509, 0.3841, 0.4204]
Epoch 5 Avg Loss: 0.4995
Epoch 5 Evaluation - HR@[10,20,50]: [0.4397, 0.5489, 0.7205] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 6 Avg Loss: 0.4874
Epoch 6 Evaluation - HR@[10,20,50]: [0.5571, 0.6910, 0.8670] | Precision@[10,20,50]: [0.0557, 0.0346, 0.0173] | NDCG@[10,20,50]: [0.3644, 0.3983, 0.4334]
Epoch 7 Avg Loss: 0.4788
Epoch 7 Evaluation - HR@[10,20,50]: [0.4467, 0.5578, 0.7265] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 8 Avg Loss: 0.4717
Epoch 8 Evaluation - HR@[10,20,50]: [0.5745, 0.7083, 0.8785] | Precision@[10,20,50]: [0.0575, 0.0354, 0.0176] | NDCG@[10,20,50]: [0.3753, 0.4091, 0.4431]
Epoch 9 Avg Loss: 0.4660
Epoch 9 Evaluation - HR@[10,20,50]: [0.4518, 0.5631, 0.7291] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 10 Avg Loss: 0.4616
Epoch 10 Evaluation - HR@[10,20,50]: [0.5852, 0.7195, 0.8856] | Precision@[10,20,50]: [0.0585, 0.0360, 0.0177] | NDCG@[10,20,50]: [0.3819, 0.4158, 0.4490]
```

## File: output\simplified_for_llm\logs\baseline\bprmf_20260409_161909.txt

```text
Training started. Args: Namespace(model='BPR-MF', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline', eval_only=False, resume=None, experiment_id=2)
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.6705
Epoch 1 Evaluation - HR@[10,20,50]: [0.3774, 0.4771, 0.6650] | Precision@[10,20,50]: [0.0377, 0.0239, 0.0133] | NDCG@[10,20,50]: [0.2531, 0.2781, 0.3152]
Epoch 2 Avg Loss: 0.5858
Epoch 2 Evaluation - HR@[10,20,50]: [0.4105, 0.5141, 0.6993] | Precision@[10,20,50]: [0.0411, 0.0257, 0.0140] | NDCG@[10,20,50]: [0.2732, 0.2993, 0.3359]
Epoch 3 Avg Loss: 0.5409
Epoch 3 Evaluation - HR@[10,20,50]: [0.4261, 0.5327, 0.7109] | Precision@[10,20,50]: [0.0426, 0.0266, 0.0142] | NDCG@[10,20,50]: [0.2819, 0.3088, 0.3440]
Epoch 4 Avg Loss: 0.5148
Epoch 4 Evaluation - HR@[10,20,50]: [0.4344, 0.5447, 0.7178] | Precision@[10,20,50]: [0.0434, 0.0272, 0.0144] | NDCG@[10,20,50]: [0.2861, 0.3139, 0.3482]
Epoch 5 Avg Loss: 0.4983
Epoch 5 Evaluation - HR@[10,20,50]: [0.4408, 0.5512, 0.7222] | Precision@[10,20,50]: [0.0441, 0.0276, 0.0144] | NDCG@[10,20,50]: [0.2901, 0.3179, 0.3518]
Epoch 6 Avg Loss: 0.4869
Epoch 6 Evaluation - HR@[10,20,50]: [0.4449, 0.5565, 0.7247] | Precision@[10,20,50]: [0.0445, 0.0278, 0.0145] | NDCG@[10,20,50]: [0.2921, 0.3202, 0.3536]
Epoch 7 Avg Loss: 0.4782
Epoch 7 Evaluation - HR@[10,20,50]: [0.4469, 0.5594, 0.7263] | Precision@[10,20,50]: [0.0447, 0.0280, 0.0145] | NDCG@[10,20,50]: [0.2932, 0.3216, 0.3548]
Epoch 8 Avg Loss: 0.4715
Epoch 8 Evaluation - HR@[10,20,50]: [0.4510, 0.5624, 0.7289] | Precision@[10,20,50]: [0.0451, 0.0281, 0.0146] | NDCG@[10,20,50]: [0.2961, 0.3242, 0.3572]
Epoch 9 Avg Loss: 0.4661
Epoch 9 Evaluation - HR@[10,20,50]: [0.4530, 0.5654, 0.7306] | Precision@[10,20,50]: [0.0453, 0.0283, 0.0146] | NDCG@[10,20,50]: [0.2970, 0.3253, 0.3581]
Epoch 10 Avg Loss: 0.4614
Epoch 10 Evaluation - HR@[10,20,50]: [0.4544, 0.5671, 0.7319] | Precision@[10,20,50]: [0.0454, 0.0284, 0.0146] | NDCG@[10,20,50]: [0.2982, 0.3266, 0.3593]
```

## File: output\simplified_for_llm\logs\baseline\bprmf_20260409_220451.txt

```text
Training started. Args: Namespace(model='BPR-MF', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline', eval_only=False, resume=None, experiment_id=3)
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.6708
Epoch 1 Evaluation - HR@[10,20,50]: [0.3777, 0.4782, 0.6653] | Precision@[10,20,50]: [0.0378, 0.0239, 0.0133] | NDCG@[10,20,50]: [0.2537, 0.2789, 0.3159]
Epoch 2 Avg Loss: 0.5861
Epoch 2 Evaluation - HR@[10,20,50]: [0.4121, 0.5156, 0.6997] | Precision@[10,20,50]: [0.0412, 0.0258, 0.0140] | NDCG@[10,20,50]: [0.2741, 0.3002, 0.3365]
Epoch 3 Avg Loss: 0.5408
Epoch 3 Evaluation - HR@[10,20,50]: [0.4277, 0.5347, 0.7114] | Precision@[10,20,50]: [0.0428, 0.0267, 0.0142] | NDCG@[10,20,50]: [0.2826, 0.3096, 0.3445]
Epoch 4 Avg Loss: 0.5144
Epoch 4 Evaluation - HR@[10,20,50]: [0.4372, 0.5463, 0.7190] | Precision@[10,20,50]: [0.0437, 0.0273, 0.0144] | NDCG@[10,20,50]: [0.2880, 0.3155, 0.3496]
Epoch 5 Avg Loss: 0.4977
Epoch 5 Evaluation - HR@[10,20,50]: [0.4428, 0.5530, 0.7231] | Precision@[10,20,50]: [0.0443, 0.0277, 0.0145] | NDCG@[10,20,50]: [0.2916, 0.3195, 0.3532]
Epoch 6 Avg Loss: 0.4861
Epoch 6 Evaluation - HR@[10,20,50]: [0.4481, 0.5579, 0.7257] | Precision@[10,20,50]: [0.0448, 0.0279, 0.0145] | NDCG@[10,20,50]: [0.2939, 0.3216, 0.3549]
Epoch 7 Avg Loss: 0.4772
Epoch 7 Evaluation - HR@[10,20,50]: [0.4497, 0.5615, 0.7276] | Precision@[10,20,50]: [0.0450, 0.0281, 0.0146] | NDCG@[10,20,50]: [0.2958, 0.3241, 0.3570]
Epoch 8 Avg Loss: 0.4703
Epoch 8 Evaluation - HR@[10,20,50]: [0.4528, 0.5643, 0.7299] | Precision@[10,20,50]: [0.0453, 0.0282, 0.0146] | NDCG@[10,20,50]: [0.2972, 0.3253, 0.3582]
Epoch 9 Avg Loss: 0.4648
Epoch 9 Evaluation - HR@[10,20,50]: [0.4548, 0.5670, 0.7318] | Precision@[10,20,50]: [0.0455, 0.0283, 0.0146] | NDCG@[10,20,50]: [0.2982, 0.3265, 0.3593]
Epoch 10 Avg Loss: 0.4600
Epoch 10 Evaluation - HR@[10,20,50]: [0.4562, 0.5680, 0.7328] | Precision@[10,20,50]: [0.0456, 0.0284, 0.0147] | NDCG@[10,20,50]: [0.2998, 0.3280, 0.3608]
```

## File: output\simplified_for_llm\logs\baseline\lightgcn_20260408_173719.txt

```text
Training started. Args: Namespace(model='LightGCN', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline')
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.6552
Epoch 1 Evaluation - HR@[10,20,50]: [0.5350, 0.6685, 0.8532] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 2 Avg Loss: 0.6242
Epoch 2 Evaluation - HR@[10,20,50]: [0.5436, 0.6804, 0.8644] | Precision@[10,20,50]: [0.0544, 0.0340, 0.0173] | NDCG@[10,20,50]: [0.3553, 0.3899, 0.4265]
Epoch 3 Avg Loss: 0.6169
Epoch 3 Evaluation - HR@[10,20,50]: [0.5365, 0.6714, 0.8549] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 4 Avg Loss: 0.6128
Epoch 4 Evaluation - HR@[10,20,50]: [0.5424, 0.6800, 0.8646] | Precision@[10,20,50]: [0.0542, 0.0340, 0.0173] | NDCG@[10,20,50]: [0.3543, 0.3891, 0.4259]
Epoch 5 Avg Loss: 0.6094
Epoch 5 Evaluation - HR@[10,20,50]: [0.5363, 0.6702, 0.8551] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 6 Avg Loss: 0.6067
Epoch 6 Evaluation - HR@[10,20,50]: [0.5450, 0.6811, 0.8651] | Precision@[10,20,50]: [0.0545, 0.0341, 0.0173] | NDCG@[10,20,50]: [0.3553, 0.3897, 0.4263]
Epoch 7 Avg Loss: 0.6030
Epoch 7 Evaluation - HR@[10,20,50]: [0.5374, 0.6708, 0.8555] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 8 Avg Loss: 0.5993
Epoch 8 Evaluation - HR@[10,20,50]: [0.5480, 0.6872, 0.8712] | Precision@[10,20,50]: [0.0548, 0.0344, 0.0174] | NDCG@[10,20,50]: [0.3565, 0.3917, 0.4284]
Epoch 9 Avg Loss: 0.5969
Epoch 9 Evaluation - HR@[10,20,50]: [0.5386, 0.6721, 0.8564] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 10 Avg Loss: 0.5942
Epoch 10 Evaluation - HR@[10,20,50]: [0.5490, 0.6890, 0.8730] | Precision@[10,20,50]: [0.0549, 0.0344, 0.0175] | NDCG@[10,20,50]: [0.3570, 0.3924, 0.4291]
```

## File: output\simplified_for_llm\logs\baseline\lightgcn_20260409_162109.txt

```text
Training started. Args: Namespace(model='LightGCN', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline', eval_only=False, resume=None, experiment_id=2)
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.6549
Epoch 1 Evaluation - HR@[10,20,50]: [0.5357, 0.6697, 0.8543] | Precision@[10,20,50]: [0.0536, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3511, 0.3850, 0.4217]
Epoch 2 Avg Loss: 0.6242
Epoch 2 Evaluation - HR@[10,20,50]: [0.5371, 0.6718, 0.8560] | Precision@[10,20,50]: [0.0537, 0.0336, 0.0171] | NDCG@[10,20,50]: [0.3516, 0.3856, 0.4223]
Epoch 3 Avg Loss: 0.6169
Epoch 3 Evaluation - HR@[10,20,50]: [0.5352, 0.6701, 0.8543] | Precision@[10,20,50]: [0.0535, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3512, 0.3852, 0.4219]
Epoch 4 Avg Loss: 0.6120
Epoch 4 Evaluation - HR@[10,20,50]: [0.5348, 0.6693, 0.8556] | Precision@[10,20,50]: [0.0535, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3505, 0.3845, 0.4216]
Epoch 5 Avg Loss: 0.6091
Epoch 5 Evaluation - HR@[10,20,50]: [0.5348, 0.6694, 0.8557] | Precision@[10,20,50]: [0.0535, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3503, 0.3843, 0.4214]
Epoch 6 Avg Loss: 0.6065
Epoch 6 Evaluation - HR@[10,20,50]: [0.5352, 0.6686, 0.8550] | Precision@[10,20,50]: [0.0535, 0.0334, 0.0171] | NDCG@[10,20,50]: [0.3509, 0.3846, 0.4217]
Epoch 7 Avg Loss: 0.6042
Epoch 7 Evaluation - HR@[10,20,50]: [0.5343, 0.6677, 0.8544] | Precision@[10,20,50]: [0.0534, 0.0334, 0.0171] | NDCG@[10,20,50]: [0.3503, 0.3840, 0.4212]
Epoch 8 Avg Loss: 0.6011
Epoch 8 Evaluation - HR@[10,20,50]: [0.5358, 0.6697, 0.8560] | Precision@[10,20,50]: [0.0536, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3506, 0.3844, 0.4215]
Epoch 9 Avg Loss: 0.5978
Epoch 9 Evaluation - HR@[10,20,50]: [0.5374, 0.6716, 0.8585] | Precision@[10,20,50]: [0.0537, 0.0336, 0.0172] | NDCG@[10,20,50]: [0.3512, 0.3851, 0.4223]
Epoch 10 Avg Loss: 0.5949
Epoch 10 Evaluation - HR@[10,20,50]: [0.5378, 0.6717, 0.8590] | Precision@[10,20,50]: [0.0538, 0.0336, 0.0172] | NDCG@[10,20,50]: [0.3509, 0.3847, 0.4220]
```

## File: output\simplified_for_llm\logs\baseline\lightgcn_20260409_220638.txt

```text
Training started. Args: Namespace(model='LightGCN', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline', eval_only=False, resume=None, experiment_id=3)
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.6548
Epoch 1 Evaluation - HR@[10,20,50]: [0.5357, 0.6700, 0.8542] | Precision@[10,20,50]: [0.0536, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3509, 0.3848, 0.4215]
Epoch 2 Avg Loss: 0.6242
Epoch 2 Evaluation - HR@[10,20,50]: [0.5366, 0.6716, 0.8562] | Precision@[10,20,50]: [0.0537, 0.0336, 0.0171] | NDCG@[10,20,50]: [0.3518, 0.3859, 0.4227]
Epoch 3 Avg Loss: 0.6169
Epoch 3 Evaluation - HR@[10,20,50]: [0.5351, 0.6703, 0.8546] | Precision@[10,20,50]: [0.0535, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3505, 0.3847, 0.4213]
Epoch 4 Avg Loss: 0.6121
Epoch 4 Evaluation - HR@[10,20,50]: [0.5352, 0.6692, 0.8559] | Precision@[10,20,50]: [0.0535, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3510, 0.3849, 0.4220]
Epoch 5 Avg Loss: 0.6091
Epoch 5 Evaluation - HR@[10,20,50]: [0.5348, 0.6695, 0.8558] | Precision@[10,20,50]: [0.0535, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3506, 0.3846, 0.4217]
Epoch 6 Avg Loss: 0.6065
Epoch 6 Evaluation - HR@[10,20,50]: [0.5352, 0.6687, 0.8551] | Precision@[10,20,50]: [0.0535, 0.0334, 0.0171] | NDCG@[10,20,50]: [0.3511, 0.3848, 0.4219]
Epoch 7 Avg Loss: 0.6042
Epoch 7 Evaluation - HR@[10,20,50]: [0.5353, 0.6681, 0.8549] | Precision@[10,20,50]: [0.0535, 0.0334, 0.0171] | NDCG@[10,20,50]: [0.3504, 0.3839, 0.4211]
Epoch 8 Avg Loss: 0.6025
Epoch 8 Evaluation - HR@[10,20,50]: [0.5354, 0.6677, 0.8546] | Precision@[10,20,50]: [0.0535, 0.0334, 0.0171] | NDCG@[10,20,50]: [0.3505, 0.3839, 0.4211]
Epoch 9 Avg Loss: 0.5996
Epoch 9 Evaluation - HR@[10,20,50]: [0.5362, 0.6691, 0.8563] | Precision@[10,20,50]: [0.0536, 0.0335, 0.0171] | NDCG@[10,20,50]: [0.3500, 0.3836, 0.4209]
Epoch 10 Avg Loss: 0.5962
Epoch 10 Evaluation - HR@[10,20,50]: [0.5372, 0.6716, 0.8587] | Precision@[10,20,50]: [0.0537, 0.0336, 0.0172] | NDCG@[10,20,50]: [0.3508, 0.3847, 0.4220]
```

## File: output\simplified_for_llm\logs\baseline\nfm_20260408_232926.txt

```text
Training started. Args: Namespace(model='NFM', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline')
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.2616
Epoch 1 Evaluation - HR@[10,20,50]: [0.3156, 0.4463, 0.6662] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 2 Avg Loss: 0.0943
Epoch 2 Evaluation - HR@[10,20,50]: [0.3607, 0.4769, 0.7000] | Precision@[10,20,50]: [0.0361, 0.0238, 0.0140] | NDCG@[10,20,50]: [0.2321, 0.2614, 0.3054]
Epoch 3 Avg Loss: 0.0637
Epoch 3 Evaluation - HR@[10,20,50]: [0.3213, 0.4392, 0.6920] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 4 Avg Loss: 0.0532
Epoch 4 Evaluation - HR@[10,20,50]: [0.3009, 0.3979, 0.6040] | Precision@[10,20,50]: [0.0301, 0.0199, 0.0121] | NDCG@[10,20,50]: [0.2016, 0.2260, 0.2665]
Epoch 5 Avg Loss: 0.0447
Epoch 5 Evaluation - HR@[10,20,50]: [0.3224, 0.4373, 0.6800] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 6 Avg Loss: 0.0396
Epoch 6 Evaluation - HR@[10,20,50]: [0.3281, 0.4265, 0.6386] | Precision@[10,20,50]: [0.0328, 0.0213, 0.0128] | NDCG@[10,20,50]: [0.2124, 0.2371, 0.2788]
Epoch 7 Avg Loss: 0.0371
Epoch 7 Evaluation - HR@[10,20,50]: [0.2948, 0.4049, 0.6345] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 8 Avg Loss: 0.0350
Epoch 8 Evaluation - HR@[10,20,50]: [0.3216, 0.4113, 0.6172] | Precision@[10,20,50]: [0.0322, 0.0206, 0.0123] | NDCG@[10,20,50]: [0.2228, 0.2453, 0.2858]
Epoch 9 Avg Loss: 0.0341
Epoch 9 Evaluation - HR@[10,20,50]: [0.3070, 0.4236, 0.6799] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 10 Avg Loss: 0.0337
Epoch 10 Evaluation - HR@[10,20,50]: [0.3272, 0.4213, 0.6282] | Precision@[10,20,50]: [0.0327, 0.0211, 0.0126] | NDCG@[10,20,50]: [0.2149, 0.2385, 0.2791]
```

## File: output\simplified_for_llm\logs\baseline\nfm_20260409_220138.txt

```text
Training started. Args: Namespace(model='NFM', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline', eval_only=False, resume=None, experiment_id=2)
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.2600
Epoch 1 Evaluation - HR@[10,20,50]: [0.3104, 0.4145, 0.6463] | Precision@[10,20,50]: [0.0310, 0.0207, 0.0129] | NDCG@[10,20,50]: [0.1964, 0.2225, 0.2680]
Epoch 2 Avg Loss: 0.0977
Epoch 2 Evaluation - HR@[10,20,50]: [0.3131, 0.4136, 0.6393] | Precision@[10,20,50]: [0.0313, 0.0207, 0.0128] | NDCG@[10,20,50]: [0.2008, 0.2261, 0.2704]
Epoch 3 Avg Loss: 0.0766
Epoch 3 Evaluation - HR@[10,20,50]: [0.3507, 0.4521, 0.6788] | Precision@[10,20,50]: [0.0351, 0.0226, 0.0136] | NDCG@[10,20,50]: [0.2277, 0.2531, 0.2976]
Epoch 4 Avg Loss: 0.0671
Epoch 4 Evaluation - HR@[10,20,50]: [0.3416, 0.4475, 0.6787] | Precision@[10,20,50]: [0.0342, 0.0224, 0.0136] | NDCG@[10,20,50]: [0.2177, 0.2443, 0.2897]
Epoch 5 Avg Loss: 0.0617
Epoch 5 Evaluation - HR@[10,20,50]: [0.3083, 0.4079, 0.6343] | Precision@[10,20,50]: [0.0308, 0.0204, 0.0127] | NDCG@[10,20,50]: [0.2009, 0.2259, 0.2703]
Epoch 6 Avg Loss: 0.0581
Epoch 6 Evaluation - HR@[10,20,50]: [0.3247, 0.4254, 0.6504] | Precision@[10,20,50]: [0.0325, 0.0213, 0.0130] | NDCG@[10,20,50]: [0.2130, 0.2382, 0.2824]
Epoch 7 Avg Loss: 0.0559
Epoch 7 Evaluation - HR@[10,20,50]: [0.3206, 0.4210, 0.6483] | Precision@[10,20,50]: [0.0321, 0.0211, 0.0130] | NDCG@[10,20,50]: [0.2106, 0.2358, 0.2804]
Epoch 8 Avg Loss: 0.0528
Epoch 8 Evaluation - HR@[10,20,50]: [0.2509, 0.3594, 0.6160] | Precision@[10,20,50]: [0.0251, 0.0180, 0.0123] | NDCG@[10,20,50]: [0.1547, 0.1819, 0.2322]
Epoch 9 Avg Loss: 0.0510
Epoch 9 Evaluation - HR@[10,20,50]: [0.3305, 0.4377, 0.6725] | Precision@[10,20,50]: [0.0331, 0.0219, 0.0134] | NDCG@[10,20,50]: [0.2151, 0.2420, 0.2881]
Epoch 10 Avg Loss: 0.0501
Epoch 10 Evaluation - HR@[10,20,50]: [0.3155, 0.4164, 0.6475] | Precision@[10,20,50]: [0.0315, 0.0208, 0.0129] | NDCG@[10,20,50]: [0.2072, 0.2326, 0.2780]
```

## File: output\simplified_for_llm\logs\baseline\nfm_20260410_041310.txt

```text
Training started. Args: Namespace(model='NFM', epochs=10, batch_size=1024, data_dir='data/processed', model_dir='models/baseline', embed_dim=64, cpu=False, debug=False, use_bf16=True, lr=0.001, log_dir='output/logs/baseline', eval_only=False, resume=None, experiment_id=3)
Using Device: xpu
Train samples: 905893, Test samples: 226474
Epoch 1 Avg Loss: 0.2486
Epoch 1 Evaluation - HR@[10,20,50]: [0.2998, 0.4026, 0.6361] | Precision@[10,20,50]: [0.0300, 0.0201, 0.0127] | NDCG@[10,20,50]: [0.1915, 0.2173, 0.2631]
Epoch 2 Avg Loss: 0.0989
Epoch 2 Evaluation - HR@[10,20,50]: [0.3148, 0.4122, 0.6423] | Precision@[10,20,50]: [0.0315, 0.0206, 0.0128] | NDCG@[10,20,50]: [0.2057, 0.2301, 0.2753]
Epoch 3 Avg Loss: 0.0781
Epoch 3 Evaluation - HR@[10,20,50]: [0.3339, 0.4304, 0.6627] | Precision@[10,20,50]: [0.0334, 0.0215, 0.0133] | NDCG@[10,20,50]: [0.2213, 0.2455, 0.2911]
Epoch 4 Avg Loss: 0.0673
Epoch 4 Evaluation - HR@[10,20,50]: [0.3232, 0.4200, 0.6534] | Precision@[10,20,50]: [0.0323, 0.0210, 0.0131] | NDCG@[10,20,50]: [0.2166, 0.2409, 0.2866]
Epoch 5 Avg Loss: 0.0596
Epoch 5 Evaluation - HR@[10,20,50]: [0.3433, 0.4390, 0.6679] | Precision@[10,20,50]: [0.0343, 0.0220, 0.0134] | NDCG@[10,20,50]: [0.2333, 0.2573, 0.3022]
Epoch 6 Avg Loss: 0.0560
Epoch 6 Evaluation - HR@[10,20,50]: [0.3471, 0.4430, 0.6731] | Precision@[10,20,50]: [0.0347, 0.0222, 0.0135] | NDCG@[10,20,50]: [0.2350, 0.2590, 0.3041]
Epoch 7 Avg Loss: 0.0537
Epoch 7 Evaluation - HR@[10,20,50]: [0.3436, 0.4392, 0.6687] | Precision@[10,20,50]: [0.0344, 0.0220, 0.0134] | NDCG@[10,20,50]: [0.2347, 0.2587, 0.3037]
Epoch 8 Avg Loss: 0.0526
Epoch 8 Evaluation - HR@[10,20,50]: [0.3334, 0.4309, 0.6630] | Precision@[10,20,50]: [0.0333, 0.0215, 0.0133] | NDCG@[10,20,50]: [0.2244, 0.2488, 0.2943]
Epoch 9 Avg Loss: 0.0513
Epoch 9 Evaluation - HR@[10,20,50]: [0.3386, 0.4372, 0.6667] | Precision@[10,20,50]: [0.0339, 0.0219, 0.0133] | NDCG@[10,20,50]: [0.2289, 0.2536, 0.2986]
Epoch 10 Avg Loss: 0.0510
Epoch 10 Evaluation - HR@[10,20,50]: [0.3380, 0.4357, 0.6687] | Precision@[10,20,50]: [0.0338, 0.0218, 0.0134] | NDCG@[10,20,50]: [0.2286, 0.2532, 0.2988]
```

## File: output\simplified_for_llm\logs\depth_2\kgat_20260330_151647.txt

```text
Training started. Args: Namespace(epochs=10, batch_size=1024, without_kg=False, resume=None, data_dir='data/processed', model_dir='models/depth_2', embed_dim=64, layers=[64, 64], cpu=False, debug=False, use_bf16=True, lr=0.001, no_compile=True, log_dir='output/logs/depth_2')
Log file: output/logs/depth_2\kgat_20260330_151647.txt
Using Intel Arc GPU (XPU)
Train samples: 905893, Test samples: 226474
Expected iterations per epoch: 884
Initializing KGATAttention with embed_dim=64, layers=[64, 64]
Enabled BFloat16 precision
Model compilation disabled by user.
Epoch 1 done. Avg Loss: 0.3475
Epoch 1 Evaluation - HR@[10,20,50]: [0.6168, 0.7670, 0.9517] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 1 Current LR: 1.000000e-03
Epoch 2 done. Avg Loss: 0.2889
Epoch 2 Evaluation - HR@[10,20,50]: [0.6875, 0.8369, 0.9737] | Precision@[10,20,50]: [0.0688, 0.0418, 0.0195] | NDCG@[10,20,50]: [0.4452, 0.4830, 0.5108]
Epoch 2 Current LR: 1.000000e-03
Epoch 3 done. Avg Loss: 0.2751
Epoch 3 Evaluation - HR@[10,20,50]: [0.6141, 0.7510, 0.9312] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 3 Current LR: 1.000000e-03
Epoch 4 done. Avg Loss: 0.2662
Epoch 4 Evaluation - HR@[10,20,50]: [0.7080, 0.8564, 0.9752] | Precision@[10,20,50]: [0.0708, 0.0428, 0.0195] | NDCG@[10,20,50]: [0.4617, 0.4993, 0.5235]
Epoch 4 Current LR: 5.000000e-04
Epoch 5 done. Avg Loss: 0.2605
Epoch 5 Evaluation - HR@[10,20,50]: [0.6144, 0.7453, 0.9186] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 5 Current LR: 5.000000e-04
Epoch 6 done. Avg Loss: 0.2551
Epoch 6 Evaluation - HR@[10,20,50]: [0.7192, 0.8666, 0.9771] | Precision@[10,20,50]: [0.0719, 0.0433, 0.0195] | NDCG@[10,20,50]: [0.4714, 0.5087, 0.5312]
Epoch 6 Current LR: 5.000000e-04
Epoch 7 done. Avg Loss: 0.2525
Epoch 7 Evaluation - HR@[10,20,50]: [0.6138, 0.7417, 0.9096] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 7 Current LR: 2.500000e-04
Epoch 8 done. Avg Loss: 0.2500
Epoch 8 Evaluation - HR@[10,20,50]: [0.7250, 0.8725, 0.9772] | Precision@[10,20,50]: [0.0725, 0.0436, 0.0195] | NDCG@[10,20,50]: [0.4758, 0.5132, 0.5346]
Epoch 8 Current LR: 2.500000e-04
Epoch 9 done. Avg Loss: 0.2488
Epoch 9 Evaluation - HR@[10,20,50]: [0.6145, 0.7410, 0.9071] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 9 Current LR: 2.500000e-04
Epoch 10 done. Avg Loss: 0.2476
Epoch 10 Evaluation - HR@[10,20,50]: [0.7282, 0.8766, 0.9775] | Precision@[10,20,50]: [0.0728, 0.0438, 0.0196] | NDCG@[10,20,50]: [0.4788, 0.5165, 0.5371]
Epoch 10 Current LR: 1.250000e-04
```

## File: output\simplified_for_llm\logs\depth_3\kgat_20260331_113536.txt

```text
Training started. Args: Namespace(epochs=10, batch_size=1024, without_kg=False, resume=None, data_dir='data/processed', model_dir='models/depth_3', embed_dim=64, layers=[64, 64, 64], cpu=False, debug=False, use_bf16=True, lr=0.001, no_compile=True, log_dir='output/logs/depth_3')
Log file: output/logs/depth_3\kgat_20260331_113536.txt
Using Intel Arc GPU (XPU)
Train samples: 905893, Test samples: 226474
Expected iterations per epoch: 884
Initializing KGATAttention with embed_dim=64, layers=[64, 64, 64]
Enabled BFloat16 precision
Model compilation disabled by user.
Epoch 1 done. Avg Loss: 0.2857
Epoch 1 Evaluation - HR@[10,20,50]: [0.7007, 0.8514, 0.9788] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 1 Current LR: 1.000000e-03
Epoch 2 done. Avg Loss: 0.2189
Epoch 2 Evaluation - HR@[10,20,50]: [0.7790, 0.9120, 0.9899] | Precision@[10,20,50]: [0.0779, 0.0456, 0.0198] | NDCG@[10,20,50]: [0.5269, 0.5608, 0.5767]
Epoch 2 Current LR: 1.000000e-03
Epoch 3 done. Avg Loss: 0.2034
Epoch 3 Evaluation - HR@[10,20,50]: [0.6903, 0.8397, 0.9711] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 3 Current LR: 1.000000e-03
Epoch 4 done. Avg Loss: 0.1941
Epoch 4 Evaluation - HR@[10,20,50]: [0.7991, 0.9241, 0.9905] | Precision@[10,20,50]: [0.0799, 0.0462, 0.0198] | NDCG@[10,20,50]: [0.5445, 0.5764, 0.5900]
Epoch 4 Current LR: 5.000000e-04
Epoch 5 done. Avg Loss: 0.1875
Epoch 5 Evaluation - HR@[10,20,50]: [0.6806, 0.8211, 0.9622] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 5 Current LR: 5.000000e-04
Epoch 6 done. Avg Loss: 0.1838
Epoch 6 Evaluation - HR@[10,20,50]: [0.8130, 0.9324, 0.9906] | Precision@[10,20,50]: [0.0813, 0.0466, 0.0198] | NDCG@[10,20,50]: [0.5543, 0.5848, 0.5967]
Epoch 6 Current LR: 5.000000e-04
Epoch 7 done. Avg Loss: 0.1807
Epoch 7 Evaluation - HR@[10,20,50]: [0.6748, 0.8099, 0.9552] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 7 Current LR: 2.500000e-04
Epoch 8 done. Avg Loss: 0.1786
Epoch 8 Evaluation - HR@[10,20,50]: [0.8185, 0.9348, 0.9905] | Precision@[10,20,50]: [0.0818, 0.0467, 0.0198] | NDCG@[10,20,50]: [0.5573, 0.5871, 0.5984]
Epoch 8 Current LR: 2.500000e-04
Epoch 9 done. Avg Loss: 0.1777
Epoch 9 Evaluation - HR@[10,20,50]: [0.6735, 0.8075, 0.9541] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 9 Current LR: 2.500000e-04
Epoch 10 done. Avg Loss: 0.1768
Epoch 10 Evaluation - HR@[10,20,50]: [0.8225, 0.9368, 0.9901] | Precision@[10,20,50]: [0.0822, 0.0468, 0.0198] | NDCG@[10,20,50]: [0.5599, 0.5891, 0.6000]
Epoch 10 Current LR: 1.250000e-04
```

## File: output\simplified_for_llm\logs\depth_3\kgat_20260409_061132.txt

```text
Training started. Args: Namespace(epochs=3, batch_size=1024, without_kg=False, resume=None, data_dir='data/processed', model_dir='models/depth_3', embed_dim=64, layers=[64, 64, 64], cpu=False, debug=False, use_bf16=True, lr=0.001, no_compile=True, log_dir='output/logs/depth_3')
Log file: output/logs/depth_3\kgat_20260409_061132.txt
Using Intel Arc GPU (XPU)
Train samples: 905893, Test samples: 226474
Expected iterations per epoch: 884
Initializing KGATAttention with embed_dim=64, layers=[64, 64, 64]
Enabled BFloat16 precision
Model compilation disabled by user.
Epoch 1 done. Avg Loss: 0.2823
Epoch 1 Evaluation - HR@[10,20,50]: [0.7538, 0.8906, 0.9859] | Precision@[10,20,50]: [0.0754, 0.0445, 0.0197] | NDCG@[10,20,50]: [0.5059, 0.5406, 0.5601]
Epoch 1 Current LR: 1.000000e-03
Epoch 2 done. Avg Loss: 0.2190
Epoch 2 Evaluation - HR@[10,20,50]: [0.7788, 0.9097, 0.9890] | Precision@[10,20,50]: [0.0779, 0.0455, 0.0198] | NDCG@[10,20,50]: [0.5271, 0.5604, 0.5766]
Epoch 2 Current LR: 1.000000e-03
Epoch 3 done. Avg Loss: 0.2026
Epoch 3 Evaluation - HR@[10,20,50]: [0.7916, 0.9185, 0.9897] | Precision@[10,20,50]: [0.0792, 0.0459, 0.0198] | NDCG@[10,20,50]: [0.5364, 0.5687, 0.5833]
Epoch 3 Current LR: 1.000000e-03
```

## File: output\simplified_for_llm\logs\full_kgat\kgat_20260327_221731.txt

```text
Training started. Args: Namespace(epochs=10, batch_size=1024, without_kg=False, resume=None, data_dir='data/processed', model_dir='models/full_kgat', embed_dim=64, layers=[64], cpu=False, debug=False, use_bf16=True, lr=0.001, no_compile=True, log_dir='output/logs/full_kgat')
Log file: output/logs/full_kgat\kgat_20260327_221731.txt
Using Intel Arc GPU (XPU)
Train samples: 905893, Test samples: 226474
Expected iterations per epoch: 884
Initializing KGATAttention with embed_dim=64, layers=[64]
Enabled BFloat16 precision
Model compilation disabled by user.
Epoch 1 done. Avg Loss: 0.4813
Epoch 1 Evaluation - HR@[10,20,50]: [0.5346, 0.6727, 0.8547] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 1 Current LR: 1.000000e-03
Epoch 2 done. Avg Loss: 0.4167
Epoch 2 Evaluation - HR@[10,20,50]: [0.5980, 0.7449, 0.9051] | Precision@[10,20,50]: [0.0598, 0.0372, 0.0181] | NDCG@[10,20,50]: [0.3661, 0.4033, 0.4353]
Epoch 2 Current LR: 1.000000e-03
Epoch 3 done. Avg Loss: 0.4018
Epoch 3 Evaluation - HR@[10,20,50]: [0.5464, 0.6740, 0.8442] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 3 Current LR: 1.000000e-03
Epoch 4 done. Avg Loss: 0.3931
Epoch 4 Evaluation - HR@[10,20,50]: [0.6176, 0.7690, 0.9225] | Precision@[10,20,50]: [0.0618, 0.0385, 0.0184] | NDCG@[10,20,50]: [0.3848, 0.4232, 0.4540]
Epoch 4 Current LR: 1.000000e-03
Epoch 5 done. Avg Loss: 0.3860
Epoch 5 Evaluation - HR@[10,20,50]: [0.5481, 0.6738, 0.8398] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 5 Current LR: 5.000000e-04
Epoch 6 done. Avg Loss: 0.3814
Epoch 6 Evaluation - HR@[10,20,50]: [0.6284, 0.7801, 0.9316] | Precision@[10,20,50]: [0.0628, 0.0390, 0.0186] | NDCG@[10,20,50]: [0.3961, 0.4345, 0.4650]
Epoch 6 Current LR: 5.000000e-04
Epoch 7 done. Avg Loss: 0.3783
Epoch 7 Evaluation - HR@[10,20,50]: [0.5509, 0.6754, 0.8381] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 7 Current LR: 5.000000e-04
Epoch 8 done. Avg Loss: 0.3757
Epoch 8 Evaluation - HR@[10,20,50]: [0.6341, 0.7862, 0.9372] | Precision@[10,20,50]: [0.0634, 0.0393, 0.0187] | NDCG@[10,20,50]: [0.4015, 0.4400, 0.4704]
Epoch 8 Current LR: 5.000000e-04
Epoch 9 done. Avg Loss: 0.3733
Epoch 9 Evaluation - HR@[10,20,50]: [0.5522, 0.6756, 0.8363] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 9 Current LR: 5.000000e-04
Epoch 10 done. Avg Loss: 0.3717
Epoch 10 Evaluation - HR@[10,20,50]: [0.6383, 0.7910, 0.9404] | Precision@[10,20,50]: [0.0638, 0.0396, 0.0188] | NDCG@[10,20,50]: [0.4054, 0.4440, 0.4742]
Epoch 10 Current LR: 5.000000e-04
```

## File: output\simplified_for_llm\logs\wo_attn\kgat_20260328_071425.txt

```text
Training started. Args: Namespace(epochs=10, batch_size=1024, lr=0.001, embed_dim=64, layers=[64], use_bf16=True, resume=None, model_dir='models/wo_attn', cpu=False, debug=False, log_dir='output/logs/wo_attn')
Log file: output/logs/wo_attn\kgat_20260328_071425.txt
Using Intel Arc GPU (XPU)
Data Split - Train: 905893, Test: 226474
Constructing adjacency matrix (CPU)...
Adjacency matrix created. Edges: 7200165
Moving adjacency matrix to xpu...
Adjacency matrix cast to BFloat16.
Done. Coalesced: True
Initializing KGAT model...
Model cast to BFloat16.
Model moved to device.
Starting training from epoch 0 to 10...
Epoch 1 done. Avg Loss: 0.4911
Epoch 1 Evaluation - HR@[10,20,50]: [0.5457, 0.6558, 0.8410] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 1 Current LR: 1.000000e-03
Epoch 2 done. Avg Loss: 0.4342
Epoch 2 Evaluation - HR@[10,20,50]: [0.6029, 0.7211, 0.8943] | Precision@[10,20,50]: [0.0603, 0.0361, 0.0179] | NDCG@[10,20,50]: [0.4002, 0.4302, 0.4640]
Epoch 2 Current LR: 1.000000e-03
Epoch 3 done. Avg Loss: 0.4225
Epoch 3 Evaluation - HR@[10,20,50]: [0.5570, 0.6684, 0.8300] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 3 Current LR: 1.000000e-03
Epoch 4 done. Avg Loss: 0.4135
Epoch 4 Evaluation - HR@[10,20,50]: [0.6277, 0.7452, 0.9080] | Precision@[10,20,50]: [0.0628, 0.0373, 0.0182] | NDCG@[10,20,50]: [0.4241, 0.4539, 0.4857]
Epoch 4 Current LR: 1.000000e-03
Epoch 5 done. Avg Loss: 0.4075
Epoch 5 Evaluation - HR@[10,20,50]: [0.5582, 0.6725, 0.8224] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 5 Current LR: 1.000000e-03
Epoch 6 done. Avg Loss: 0.4032
Epoch 6 Evaluation - HR@[10,20,50]: [0.6418, 0.7613, 0.9113] | Precision@[10,20,50]: [0.0642, 0.0381, 0.0182] | NDCG@[10,20,50]: [0.4392, 0.4696, 0.4990]
Epoch 6 Current LR: 1.000000e-03
Epoch 7 done. Avg Loss: 0.3988
Epoch 7 Evaluation - HR@[10,20,50]: [0.5619, 0.6747, 0.8202] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 7 Current LR: 1.000000e-03
Epoch 8 done. Avg Loss: 0.3953
Epoch 8 Evaluation - HR@[10,20,50]: [0.6521, 0.7727, 0.9147] | Precision@[10,20,50]: [0.0652, 0.0386, 0.0183] | NDCG@[10,20,50]: [0.4496, 0.4802, 0.5081]
Epoch 8 Current LR: 1.000000e-03
Epoch 9 done. Avg Loss: 0.3922
Epoch 9 Evaluation - HR@[10,20,50]: [0.5635, 0.6778, 0.8181] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 9 Current LR: 1.000000e-03
Epoch 10 done. Avg Loss: 0.3901
Epoch 10 Evaluation - HR@[10,20,50]: [0.6592, 0.7800, 0.9137] | Precision@[10,20,50]: [0.0659, 0.0390, 0.0183] | NDCG@[10,20,50]: [0.4569, 0.4875, 0.5140]
Epoch 10 Current LR: 1.000000e-03
```

## File: output\simplified_for_llm\logs\wo_kg\kgat_20260330_105252.txt

```text
Training started. Args: Namespace(epochs=10, batch_size=1024, without_kg=True, resume=None, data_dir='data/processed', model_dir='models/wo_kg', embed_dim=64, layers=[64], cpu=False, debug=False, use_bf16=True, lr=0.001, no_compile=True, log_dir='output/logs/wo_kg')
Log file: output/logs/wo_kg\kgat_20260330_105252.txt
Using Intel Arc GPU (XPU)
Train samples: 905893, Test samples: 226474
Expected iterations per epoch: 884
Initializing KGATAttention with embed_dim=64, layers=[64]
Enabled BFloat16 precision
Model compilation disabled by user.
Epoch 1 done. Avg Loss: 0.5042
Epoch 1 Evaluation - HR@[10,20,50]: [0.5299, 0.6803, 0.8756] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 1 Current LR: 1.000000e-03
Epoch 2 done. Avg Loss: 0.4448
Epoch 2 Evaluation - HR@[10,20,50]: [0.6470, 0.7787, 0.9279] | Precision@[10,20,50]: [0.0647, 0.0389, 0.0186] | NDCG@[10,20,50]: [0.4275, 0.4608, 0.4908]
Epoch 2 Current LR: 1.000000e-03
Epoch 3 done. Avg Loss: 0.4244
Epoch 3 Evaluation - HR@[10,20,50]: [0.5707, 0.7118, 0.8913] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 3 Current LR: 1.000000e-03
Epoch 4 done. Avg Loss: 0.4099
Epoch 4 Evaluation - HR@[10,20,50]: [0.6842, 0.8179, 0.9493] | Precision@[10,20,50]: [0.0684, 0.0409, 0.0190] | NDCG@[10,20,50]: [0.4595, 0.4933, 0.5199]
Epoch 4 Current LR: 1.000000e-03
Epoch 5 done. Avg Loss: 0.3973
Epoch 5 Evaluation - HR@[10,20,50]: [0.5728, 0.7119, 0.8854] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 5 Current LR: 1.000000e-03
Epoch 6 done. Avg Loss: 0.3876
Epoch 6 Evaluation - HR@[10,20,50]: [0.6995, 0.8367, 0.9589] | Precision@[10,20,50]: [0.0700, 0.0418, 0.0192] | NDCG@[10,20,50]: [0.4722, 0.5070, 0.5317]
Epoch 6 Current LR: 1.000000e-03
Epoch 7 done. Avg Loss: 0.3796
Epoch 7 Evaluation - HR@[10,20,50]: [0.5789, 0.7103, 0.8751] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 7 Current LR: 5.000000e-04
Epoch 8 done. Avg Loss: 0.3742
Epoch 8 Evaluation - HR@[10,20,50]: [0.7130, 0.8522, 0.9634] | Precision@[10,20,50]: [0.0713, 0.0426, 0.0193] | NDCG@[10,20,50]: [0.4833, 0.5185, 0.5411]
Epoch 8 Current LR: 5.000000e-04
Epoch 9 done. Avg Loss: 0.3705
Epoch 9 Evaluation - HR@[10,20,50]: [0.5851, 0.7128, 0.8716] | Precision@[10,20,50]: [None, None, None] | NDCG@[10,20,50]: [None, None, None]
Epoch 9 Current LR: 5.000000e-04
Epoch 10 done. Avg Loss: 0.3679
Epoch 10 Evaluation - HR@[10,20,50]: [0.7225, 0.8621, 0.9658] | Precision@[10,20,50]: [0.0723, 0.0431, 0.0193] | NDCG@[10,20,50]: [0.4901, 0.5255, 0.5465]
Epoch 10 Current LR: 2.500000e-04
```

