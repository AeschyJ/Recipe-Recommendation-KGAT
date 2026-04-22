# Experiment Explanations (Simplified)

## File: output\fidelity\sampled_users.json

```text
[
  155043,
  83279,
  52425,
  161934,
  4742,
  13932,
  204898,
  98680,
  30044,
  189486,
  105337,
  92217,
  58393,
  79531,
  36707,
  211325,
  167312,
  88880,
  82612,
  141540,
  77300,
  164911,
  44469,
  207664,
  199484,
  147158,
  182304,
  71953,
  218242,
  220814,
  207024,
  126151,
  160665,
  25824,
  210947,
  196214,
  126599,
  187201,
  194407,
  36069,
  131064,
  34380,
  38331,
  108993,
  115081,
  92488,
  5283,
  148009,
  96106,
  206882,
  214548,
  58074,
  3350,
  88664,
  143053,
  92466,
  204608,
  109592,
  87832,
  192418,
  214584,
  200494,
  215308,
  160031,
  36716,
  26120,
  151796,
  85832,
  88303,
  20963,
  135827,
  1415,
  6189,
  215723,
  105949,
  174245,
  53175,
  76646,
  162719,
  13969,
  85261,
  13785,
  37066,
  68821,
  219285,
  141032,
  141386,
  121380,
  58491,
  172238,
  33837,
  49187,
  154867,
  128635,
  11120,
  43020,
  173771,
  22765,
  66617,
  219316
]
```

## File: output\fidelity\depth_3\explanations.json

```text
[
  {
    "user_id_remapped": 155043,
    "user_id_original": "2951693",
    "user_name": "User 2951693",
    "recommended_item_id_remapped": 170030,
    "recommended_item_id_original": "324540",
    "recommended_item_name": "chocolate and peanut butter protein bars",
    "score": 2.9293060302734375,
    "original_prob": 0.9492762684822083,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2951693 -> chocolate and peanut butter protein bars",
        "path_details": [
          {
            "id_remapped": 155043,
            "id_original": "2951693",
            "type": "USER",
            "name": "User 2951693"
          },
          {
            "id_remapped": 396600,
            "id_original": "324540",
            "type": "RECIPE",
            "name": "chocolate and peanut butter protein bars"
          }
        ],
        "contribution_score": 0.7782042622566223
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.10473591089248657,
      "prob_f_plus": 0.8445403575897217,
      "fidelity_minus": -0.004016876220703125,
      "prob_f_minus": 0.9532931447029114
    }
  },
  {
    "user_id_remapped": 83279,
    "user_id_original": "963993",
    "user_name": "User 963993",
    "recommended_item_id_remapped": 13816,
    "recommended_item_id_original": "26254",
    "recommended_item_name": "fantastic tropical thai chicken salad",
    "score": 2.6393957138061523,
    "original_prob": 0.9333543181419373,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 963993 -> awesome  and easy sweet and sour chicken -> chicken -> fantastic tropical thai chicken salad",
        "path_details": [
          {
            "id_remapped": 83279,
            "id_original": "963993",
            "type": "USER",
            "name": "User 963993"
          },
          {
            "id_remapped": 316370,
            "id_original": "156520",
            "type": "RECIPE",
            "name": "awesome  and easy sweet and sour chicken"
          },
          {
            "id_remapped": 473168,
            "id_original": "14961",
            "type": "TAG",
            "name": "chicken"
          },
          {
            "id_remapped": 240386,
            "id_original": "26254",
            "type": "RECIPE",
            "name": "fantastic tropical thai chicken salad"
          }
        ],
        "contribution_score": 0.00022213389332872144
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 963993 -> ain t none better chicken breast -> chicken -> fantastic tropical thai chicken salad",
        "path_details": [
          {
            "id_remapped": 83279,
            "id_original": "963993",
            "type": "USER",
            "name": "User 963993"
          },
          {
            "id_remapped": 448377,
            "id_original": "475703",
            "type": "RECIPE",
            "name": "ain t none better chicken breast"
          },
          {
            "id_remapped": 473168,
            "id_original": "14961",
            "type": "TAG",
            "name": "chicken"
          },
          {
            "id_remapped": 240386,
            "id_original": "26254",
            "type": "RECIPE",
            "name": "fantastic tropical thai chicken salad"
          }
        ],
        "contribution_score": 0.00018071580831727978
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 963993 -> pizza chicken -> chicken -> fantastic tropical thai chicken salad",
        "path_details": [
          {
            "id_remapped": 83279,
            "id_original": "963993",
            "type": "USER",
            "name": "User 963993"
          },
          {
            "id_remapped": 397596,
            "id_original": "326936",
            "type": "RECIPE",
            "name": "pizza chicken"
          },
          {
            "id_remapped": 473168,
            "id_original": "14961",
            "type": "TAG",
            "name": "chicken"
          },
          {
            "id_remapped": 240386,
            "id_original": "26254",
            "type": "RECIPE",
            "name": "fantastic tropical thai chicken salad"
          }
        ],
        "contribution_score": 0.0001744099160126751
      }
    ],
    "fidelity": {
      "fidelity_plus": -0.0010290741920471191,
      "prob_f_plus": 0.9343833923339844,
      "fidelity_minus": 0.5414725542068481,
      "prob_f_minus": 0.3918817639350891
    }
  },
  {
    "user_id_remapped": 52425,
    "user_id_original": "499434",
    "user_name": "User 499434",
    "recommended_item_id_remapped": 32217,
    "recommended_item_id_original": "56906",
    "recommended_item_name": "broccoli  sausage and pasta ears",
    "score": 2.9832868576049805,
    "original_prob": 0.9518133401870728,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 499434 -> egg drop soup  restaurant style -> stove-top -> broccoli  sausage and pasta ears",
        "path_details": [
          {
            "id_remapped": 52425,
            "id_original": "499434",
            "type": "USER",
            "name": "User 499434"
          },
          {
            "id_remapped": 261348,
            "id_original": "61070",
            "type": "RECIPE",
            "name": "egg drop soup  restaurant style"
          },
          {
            "id_remapped": 473090,
            "id_original": "14883",
            "type": "TAG",
            "name": "stove-top"
          },
          {
            "id_remapped": 258787,
            "id_original": "56906",
            "type": "RECIPE",
            "name": "broccoli  sausage and pasta ears"
          }
        ],
        "contribution_score": 1.7504780487028235e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.006064355373382568,
      "prob_f_plus": 0.9457489848136902,
      "fidelity_minus": 0.35165709257125854,
      "prob_f_minus": 0.6001562476158142
    }
  },
  {
    "user_id_remapped": 161934,
    "user_id_original": "1801866426",
    "user_name": "User 1801866426",
    "recommended_item_id_remapped": 91195,
    "recommended_item_id_original": "159349",
    "recommended_item_name": "chicken broccoli rice and cheese casserole",
    "score": 2.9621853828430176,
    "original_prob": 0.9508362412452698,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1801866426 -> chicken broccoli rice and cheese casserole",
        "path_details": [
          {
            "id_remapped": 161934,
            "id_original": "1801866426",
            "type": "USER",
            "name": "User 1801866426"
          },
          {
            "id_remapped": 317765,
            "id_original": "159349",
            "type": "RECIPE",
            "name": "chicken broccoli rice and cheese casserole"
          }
        ],
        "contribution_score": 0.7785170674324036
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.06266188621520996,
      "prob_f_plus": 0.8881743550300598,
      "fidelity_minus": -0.0017206072807312012,
      "prob_f_minus": 0.952556848526001
    }
  },
  {
    "user_id_remapped": 4742,
    "user_id_original": "42985",
    "user_name": "User 42985",
    "recommended_item_id_remapped": 126155,
    "recommended_item_id_original": "227542",
    "recommended_item_name": "chopped salad appetizer shells",
    "score": 2.9612412452697754,
    "original_prob": 0.950792133808136,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 13932,
    "user_id_original": "118065",
    "user_name": "User 118065",
    "recommended_item_id_remapped": 44755,
    "recommended_item_id_original": "78055",
    "recommended_item_name": "pepperoni puffs",
    "score": 2.9495463371276855,
    "original_prob": 0.9502421021461487,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 118065 -> sky high strawberry pie -> potluck -> pepperoni puffs",
        "path_details": [
          {
            "id_remapped": 13932,
            "id_original": "118065",
            "type": "USER",
            "name": "User 118065"
          },
          {
            "id_remapped": 387159,
            "id_original": "303173",
            "type": "RECIPE",
            "name": "sky high strawberry pie"
          },
          {
            "id_remapped": 473138,
            "id_original": "14931",
            "type": "TAG",
            "name": "potluck"
          },
          {
            "id_remapped": 271325,
            "id_original": "78055",
            "type": "RECIPE",
            "name": "pepperoni puffs"
          }
        ],
        "contribution_score": 9.080922650221819e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 118065 -> sky high strawberry pie -> to-go -> pepperoni puffs",
        "path_details": [
          {
            "id_remapped": 13932,
            "id_original": "118065",
            "type": "USER",
            "name": "User 118065"
          },
          {
            "id_remapped": 387159,
            "id_original": "303173",
            "type": "RECIPE",
            "name": "sky high strawberry pie"
          },
          {
            "id_remapped": 473139,
            "id_original": "14932",
            "type": "TAG",
            "name": "to-go"
          },
          {
            "id_remapped": 271325,
            "id_original": "78055",
            "type": "RECIPE",
            "name": "pepperoni puffs"
          }
        ],
        "contribution_score": 6.741192208649559e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 118065 -> guinness and filet mignon chili -> User 110433 -> pepperoni puffs",
        "path_details": [
          {
            "id_remapped": 13932,
            "id_original": "118065",
            "type": "USER",
            "name": "User 118065"
          },
          {
            "id_remapped": 240708,
            "id_original": "26772",
            "type": "RECIPE",
            "name": "guinness and filet mignon chili"
          },
          {
            "id_remapped": 12976,
            "id_original": "110433",
            "type": "USER",
            "name": "User 110433"
          },
          {
            "id_remapped": 271325,
            "id_original": "78055",
            "type": "RECIPE",
            "name": "pepperoni puffs"
          }
        ],
        "contribution_score": 3.729462347346998e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0010333061218261719,
      "prob_f_plus": 0.9492087960243225,
      "fidelity_minus": 0.021521270275115967,
      "prob_f_minus": 0.9287208318710327
    }
  },
  {
    "user_id_remapped": 204898,
    "user_id_original": "2001583649",
    "user_name": "User 2001583649",
    "recommended_item_id_remapped": 19714,
    "recommended_item_id_original": "35940",
    "recommended_item_name": "boranie bademjan  persian eggplant yogurt dip",
    "score": 2.8440442085266113,
    "original_prob": 0.9450100064277649,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 98680,
    "user_id_original": "1273842",
    "user_name": "User 1273842",
    "recommended_item_id_remapped": 117724,
    "recommended_item_id_original": "211115",
    "recommended_item_name": "just 1 dozen easy oatmeal cookies",
    "score": 2.9564170837402344,
    "original_prob": 0.9505658745765686,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1273842 -> easy authentic guacamole -> User 428885 -> just 1 dozen easy oatmeal cookies",
        "path_details": [
          {
            "id_remapped": 98680,
            "id_original": "1273842",
            "type": "USER",
            "name": "User 1273842"
          },
          {
            "id_remapped": 380160,
            "id_original": "287183",
            "type": "RECIPE",
            "name": "easy authentic guacamole"
          },
          {
            "id_remapped": 46637,
            "id_original": "428885",
            "type": "USER",
            "name": "User 428885"
          },
          {
            "id_remapped": 344294,
            "id_original": "211115",
            "type": "RECIPE",
            "name": "just 1 dozen easy oatmeal cookies"
          }
        ],
        "contribution_score": 2.4410263576928056e-07
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.03963267803192139,
      "prob_f_plus": 0.9109331965446472,
      "fidelity_minus": 0.0399971604347229,
      "prob_f_minus": 0.9105687141418457
    }
  },
  {
    "user_id_remapped": 30044,
    "user_id_original": "254362",
    "user_name": "User 254362",
    "recommended_item_id_remapped": 144613,
    "recommended_item_id_original": "267661",
    "recommended_item_name": "zurie s holey rustic olive and cheddar bread",
    "score": 2.8706376552581787,
    "original_prob": 0.9463757276535034,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 254362 -> cake doughnuts -> breads -> zurie s holey rustic olive and cheddar bread",
        "path_details": [
          {
            "id_remapped": 30044,
            "id_original": "254362",
            "type": "USER",
            "name": "User 254362"
          },
          {
            "id_remapped": 264490,
            "id_original": "66102",
            "type": "RECIPE",
            "name": "cake doughnuts"
          },
          {
            "id_remapped": 473129,
            "id_original": "14922",
            "type": "TAG",
            "name": "breads"
          },
          {
            "id_remapped": 371183,
            "id_original": "267661",
            "type": "RECIPE",
            "name": "zurie s holey rustic olive and cheddar bread"
          }
        ],
        "contribution_score": 0.0017231955331743264
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.019205868244171143,
      "prob_f_plus": 0.9271698594093323,
      "fidelity_minus": 0.6073597073554993,
      "prob_f_minus": 0.33901602029800415
    }
  },
  {
    "user_id_remapped": 189486,
    "user_id_original": "2001039240",
    "user_name": "User 2001039240",
    "recommended_item_id_remapped": 62745,
    "recommended_item_id_original": "107909",
    "recommended_item_name": "honey oatmeal bread  abm",
    "score": 2.9227423667907715,
    "original_prob": 0.948959231376648,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 105337,
    "user_id_original": "1404689",
    "user_name": "User 1404689",
    "recommended_item_id_remapped": 82091,
    "recommended_item_id_original": "141983",
    "recommended_item_name": "better than olive garden alfredo sauce",
    "score": 2.972111463546753,
    "original_prob": 0.951298177242279,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1404689 -> better than olive garden alfredo sauce",
        "path_details": [
          {
            "id_remapped": 105337,
            "id_original": "1404689",
            "type": "USER",
            "name": "User 1404689"
          },
          {
            "id_remapped": 308661,
            "id_original": "141983",
            "type": "RECIPE",
            "name": "better than olive garden alfredo sauce"
          }
        ],
        "contribution_score": 0.80945885181427
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.1956464648246765,
      "prob_f_plus": 0.7556517124176025,
      "fidelity_minus": -0.0006036162376403809,
      "prob_f_minus": 0.9519017934799194
    }
  },
  {
    "user_id_remapped": 92217,
    "user_id_original": "1148892",
    "user_name": "User 1148892",
    "recommended_item_id_remapped": 198638,
    "recommended_item_id_original": "397987",
    "recommended_item_name": "pleasant pheasant",
    "score": 2.9175031185150146,
    "original_prob": 0.9487048983573914,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1148892 -> pleasant pheasant",
        "path_details": [
          {
            "id_remapped": 92217,
            "id_original": "1148892",
            "type": "USER",
            "name": "User 1148892"
          },
          {
            "id_remapped": 425208,
            "id_original": "397987",
            "type": "RECIPE",
            "name": "pleasant pheasant"
          }
        ],
        "contribution_score": 0.8073540925979614
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0416683554649353,
      "prob_f_plus": 0.907036542892456,
      "fidelity_minus": -0.0043062567710876465,
      "prob_f_minus": 0.953011155128479
    }
  },
  {
    "user_id_remapped": 58393,
    "user_id_original": "577552",
    "user_name": "User 577552",
    "recommended_item_id_remapped": 91047,
    "recommended_item_id_original": "159074",
    "recommended_item_name": "irish goose with potato stuffing",
    "score": 2.928544044494629,
    "original_prob": 0.9492396116256714,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 79531,
    "user_id_original": "895900",
    "user_name": "User 895900",
    "recommended_item_id_remapped": 159788,
    "recommended_item_id_original": "301334",
    "recommended_item_name": "sausage kolaches   klobasnicky",
    "score": 2.870499610900879,
    "original_prob": 0.9463686347007751,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 36707,
    "user_id_original": "321095",
    "user_name": "User 321095",
    "recommended_item_id_remapped": 71061,
    "recommended_item_id_original": "122012",
    "recommended_item_name": "fresh strawberry cobbler cake",
    "score": 2.958312511444092,
    "original_prob": 0.9506548643112183,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 211325,
    "user_id_original": "2001805836",
    "user_name": "User 2001805836",
    "recommended_item_id_remapped": 189104,
    "recommended_item_id_original": "373127",
    "recommended_item_name": "jamaican beef patties  aka meat pies or pasties",
    "score": 2.879812240600586,
    "original_prob": 0.9468393921852112,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 167312,
    "user_id_original": "1803024121",
    "user_name": "User 1803024121",
    "recommended_item_id_remapped": 230421,
    "recommended_item_id_original": "517764",
    "recommended_item_name": "a1 chicken thighs en croute on a bed of dijon cooked spinach  a1",
    "score": 2.9650206565856934,
    "original_prob": 0.9509685635566711,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1803024121 -> a1 chicken thighs en croute on a bed of dijon cooked spinach  a1",
        "path_details": [
          {
            "id_remapped": 167312,
            "id_original": "1803024121",
            "type": "USER",
            "name": "User 1803024121"
          },
          {
            "id_remapped": 456991,
            "id_original": "517764",
            "type": "RECIPE",
            "name": "a1 chicken thighs en croute on a bed of dijon cooked spinach  a1"
          }
        ],
        "contribution_score": 0.7610481977462769
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.1982399821281433,
      "prob_f_plus": 0.7527285814285278,
      "fidelity_minus": -0.0011442303657531738,
      "prob_f_minus": 0.9521127939224243
    }
  },
  {
    "user_id_remapped": 88880,
    "user_id_original": "1081632",
    "user_name": "User 1081632",
    "recommended_item_id_remapped": 162630,
    "recommended_item_id_original": "307493",
    "recommended_item_name": "tzatziki    gyro sauce",
    "score": 2.9082679748535156,
    "original_prob": 0.9482536315917969,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 82612,
    "user_id_original": "952152",
    "user_name": "User 952152",
    "recommended_item_id_remapped": 48005,
    "recommended_item_id_original": "83828",
    "recommended_item_name": "chicken in garlic white wine cream sauce",
    "score": 2.954840660095215,
    "original_prob": 0.9504917860031128,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 952152 -> fajita seasoning mix -> User 134624 -> chicken in garlic white wine cream sauce",
        "path_details": [
          {
            "id_remapped": 82612,
            "id_original": "952152",
            "type": "USER",
            "name": "User 952152"
          },
          {
            "id_remapped": 241442,
            "id_original": "28011",
            "type": "RECIPE",
            "name": "fajita seasoning mix"
          },
          {
            "id_remapped": 15803,
            "id_original": "134624",
            "type": "USER",
            "name": "User 134624"
          },
          {
            "id_remapped": 274575,
            "id_original": "83828",
            "type": "RECIPE",
            "name": "chicken in garlic white wine cream sauce"
          }
        ],
        "contribution_score": 3.5688234097962268e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.01601731777191162,
      "prob_f_plus": 0.9344744682312012,
      "fidelity_minus": 0.023131728172302246,
      "prob_f_minus": 0.9273600578308105
    }
  },
  {
    "user_id_remapped": 141540,
    "user_id_original": "2410864",
    "user_name": "User 2410864",
    "recommended_item_id_remapped": 224644,
    "recommended_item_id_original": "486496",
    "recommended_item_name": "bacon lattice tomato muffins  rsc",
    "score": 2.9871621131896973,
    "original_prob": 0.9519908428192139,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2410864 -> bacon lattice tomato muffins  rsc",
        "path_details": [
          {
            "id_remapped": 141540,
            "id_original": "2410864",
            "type": "USER",
            "name": "User 2410864"
          },
          {
            "id_remapped": 451214,
            "id_original": "486496",
            "type": "RECIPE",
            "name": "bacon lattice tomato muffins  rsc"
          }
        ],
        "contribution_score": 0.7670428156852722
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.24146127700805664,
      "prob_f_plus": 0.7105295658111572,
      "fidelity_minus": 0.0004470944404602051,
      "prob_f_minus": 0.9515437483787537
    }
  },
  {
    "user_id_remapped": 77300,
    "user_id_original": "858320",
    "user_name": "User 858320",
    "recommended_item_id_remapped": 164983,
    "recommended_item_id_original": "312740",
    "recommended_item_name": "caramelized mango tart with mexican chocolate and pepitas",
    "score": 2.8405234813690186,
    "original_prob": 0.9448267221450806,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 164911,
    "user_id_original": "1802666341",
    "user_name": "User 1802666341",
    "recommended_item_id_remapped": 46439,
    "recommended_item_id_original": "81048",
    "recommended_item_name": "gyro meat",
    "score": 2.9962596893310547,
    "original_prob": 0.9524049162864685,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1802666341 -> delicious chicken pot pie -> User 1410992 -> gyro meat",
        "path_details": [
          {
            "id_remapped": 164911,
            "id_original": "1802666341",
            "type": "USER",
            "name": "User 1802666341"
          },
          {
            "id_remapped": 230821,
            "id_original": "10744",
            "type": "RECIPE",
            "name": "delicious chicken pot pie"
          },
          {
            "id_remapped": 105678,
            "id_original": "1410992",
            "type": "USER",
            "name": "User 1410992"
          },
          {
            "id_remapped": 273009,
            "id_original": "81048",
            "type": "RECIPE",
            "name": "gyro meat"
          }
        ],
        "contribution_score": 0.02038855835716071
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1802666341 -> delicious chicken pot pie -> User 37036 -> gyro meat",
        "path_details": [
          {
            "id_remapped": 164911,
            "id_original": "1802666341",
            "type": "USER",
            "name": "User 1802666341"
          },
          {
            "id_remapped": 230821,
            "id_original": "10744",
            "type": "RECIPE",
            "name": "delicious chicken pot pie"
          },
          {
            "id_remapped": 3749,
            "id_original": "37036",
            "type": "USER",
            "name": "User 37036"
          },
          {
            "id_remapped": 273009,
            "id_original": "81048",
            "type": "RECIPE",
            "name": "gyro meat"
          }
        ],
        "contribution_score": 2.493249555783394e-06
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1802666341 -> delicious chicken pot pie -> User 140132 -> gyro meat",
        "path_details": [
          {
            "id_remapped": 164911,
            "id_original": "1802666341",
            "type": "USER",
            "name": "User 1802666341"
          },
          {
            "id_remapped": 230821,
            "id_original": "10744",
            "type": "RECIPE",
            "name": "delicious chicken pot pie"
          },
          {
            "id_remapped": 16523,
            "id_original": "140132",
            "type": "USER",
            "name": "User 140132"
          },
          {
            "id_remapped": 273009,
            "id_original": "81048",
            "type": "RECIPE",
            "name": "gyro meat"
          }
        ],
        "contribution_score": 1.9960706221387662e-07
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.07286393642425537,
      "prob_f_plus": 0.8795409798622131,
      "fidelity_minus": 0.0017872452735900879,
      "prob_f_minus": 0.9506176710128784
    }
  },
  {
    "user_id_remapped": 44469,
    "user_id_original": "400233",
    "user_name": "User 400233",
    "recommended_item_id_remapped": 210150,
    "recommended_item_id_original": "433837",
    "recommended_item_name": "leftover pizza frittata  using leftover pizza",
    "score": 2.8738319873809814,
    "original_prob": 0.9465376138687134,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 207664,
    "user_id_original": "2001672857",
    "user_name": "User 2001672857",
    "recommended_item_id_remapped": 54353,
    "recommended_item_id_original": "94164",
    "recommended_item_name": "amish macaroni salad",
    "score": 2.987607002258301,
    "original_prob": 0.9520111083984375,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 199484,
    "user_id_original": "2001417044",
    "user_name": "User 2001417044",
    "recommended_item_id_remapped": 70712,
    "recommended_item_id_original": "121428",
    "recommended_item_name": "stoemp aux poireaux stoemp met prei   belgian mashed potatoes",
    "score": 2.880549192428589,
    "original_prob": 0.9468764662742615,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 147158,
    "user_id_original": "2652771",
    "user_name": "User 2652771",
    "recommended_item_id_remapped": 72861,
    "recommended_item_id_original": "125399",
    "recommended_item_name": "french toast sticks   oamc",
    "score": 2.9926657676696777,
    "original_prob": 0.9522416591644287,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2652771 -> the best ever waffles -> User 1282298 -> french toast sticks   oamc",
        "path_details": [
          {
            "id_remapped": 147158,
            "id_original": "2652771",
            "type": "USER",
            "name": "User 2652771"
          },
          {
            "id_remapped": 243770,
            "id_original": "31750",
            "type": "RECIPE",
            "name": "the best ever waffles"
          },
          {
            "id_remapped": 99110,
            "id_original": "1282298",
            "type": "USER",
            "name": "User 1282298"
          },
          {
            "id_remapped": 299431,
            "id_original": "125399",
            "type": "RECIPE",
            "name": "french toast sticks   oamc"
          }
        ],
        "contribution_score": 0.00811970390992225
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2652771 -> the best ever waffles -> User 447123 -> french toast sticks   oamc",
        "path_details": [
          {
            "id_remapped": 147158,
            "id_original": "2652771",
            "type": "USER",
            "name": "User 2652771"
          },
          {
            "id_remapped": 243770,
            "id_original": "31750",
            "type": "RECIPE",
            "name": "the best ever waffles"
          },
          {
            "id_remapped": 48136,
            "id_original": "447123",
            "type": "USER",
            "name": "User 447123"
          },
          {
            "id_remapped": 299431,
            "id_original": "125399",
            "type": "RECIPE",
            "name": "french toast sticks   oamc"
          }
        ],
        "contribution_score": 0.00018775539365885688
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2652771 -> the best ever waffles -> User 1244621 -> french toast sticks   oamc",
        "path_details": [
          {
            "id_remapped": 147158,
            "id_original": "2652771",
            "type": "USER",
            "name": "User 2652771"
          },
          {
            "id_remapped": 243770,
            "id_original": "31750",
            "type": "RECIPE",
            "name": "the best ever waffles"
          },
          {
            "id_remapped": 97187,
            "id_original": "1244621",
            "type": "USER",
            "name": "User 1244621"
          },
          {
            "id_remapped": 299431,
            "id_original": "125399",
            "type": "RECIPE",
            "name": "french toast sticks   oamc"
          }
        ],
        "contribution_score": 0.00012844297888580186
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.13873374462127686,
      "prob_f_plus": 0.8135079145431519,
      "fidelity_minus": 0.0012725591659545898,
      "prob_f_minus": 0.9509690999984741
    }
  },
  {
    "user_id_remapped": 182304,
    "user_id_original": "2000558696",
    "user_name": "User 2000558696",
    "recommended_item_id_remapped": 16208,
    "recommended_item_id_original": "30165",
    "recommended_item_name": "breakfast burritos  once a month cooking",
    "score": 2.9818835258483887,
    "original_prob": 0.9517489671707153,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2000558696 -> pasta with sausage  tomatoes  and cream -> User 11009 -> breakfast burritos  once a month cooking",
        "path_details": [
          {
            "id_remapped": 182304,
            "id_original": "2000558696",
            "type": "USER",
            "name": "User 2000558696"
          },
          {
            "id_remapped": 244437,
            "id_original": "32844",
            "type": "RECIPE",
            "name": "pasta with sausage  tomatoes  and cream"
          },
          {
            "id_remapped": 754,
            "id_original": "11009",
            "type": "USER",
            "name": "User 11009"
          },
          {
            "id_remapped": 242778,
            "id_original": "30165",
            "type": "RECIPE",
            "name": "breakfast burritos  once a month cooking"
          }
        ],
        "contribution_score": 8.586788734746579e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2000558696 -> pasta with sausage  tomatoes  and cream -> User 128950 -> breakfast burritos  once a month cooking",
        "path_details": [
          {
            "id_remapped": 182304,
            "id_original": "2000558696",
            "type": "USER",
            "name": "User 2000558696"
          },
          {
            "id_remapped": 244437,
            "id_original": "32844",
            "type": "RECIPE",
            "name": "pasta with sausage  tomatoes  and cream"
          },
          {
            "id_remapped": 15221,
            "id_original": "128950",
            "type": "USER",
            "name": "User 128950"
          },
          {
            "id_remapped": 242778,
            "id_original": "30165",
            "type": "RECIPE",
            "name": "breakfast burritos  once a month cooking"
          }
        ],
        "contribution_score": 2.157771116105102e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 2000558696 -> pasta with sausage  tomatoes  and cream -> pork -> breakfast burritos  once a month cooking",
        "path_details": [
          {
            "id_remapped": 182304,
            "id_original": "2000558696",
            "type": "USER",
            "name": "User 2000558696"
          },
          {
            "id_remapped": 244437,
            "id_original": "32844",
            "type": "RECIPE",
            "name": "pasta with sausage  tomatoes  and cream"
          },
          {
            "id_remapped": 473082,
            "id_original": "14875",
            "type": "TAG",
            "name": "pork"
          },
          {
            "id_remapped": 242778,
            "id_original": "30165",
            "type": "RECIPE",
            "name": "breakfast burritos  once a month cooking"
          }
        ],
        "contribution_score": 3.2243028182629873e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.2011389136314392,
      "prob_f_plus": 0.7506100535392761,
      "fidelity_minus": 0.009267747402191162,
      "prob_f_minus": 0.9424812197685242
    }
  },
  {
    "user_id_remapped": 71953,
    "user_id_original": "775635",
    "user_name": "User 775635",
    "recommended_item_id_remapped": 60403,
    "recommended_item_id_original": "103961",
    "recommended_item_name": "kelly s apple pork chops with stuffing",
    "score": 2.9890859127044678,
    "original_prob": 0.952078640460968,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 775635 -> reeses squares   5 ingredients   no bake  reese s -> User 168423 -> kelly s apple pork chops with stuffing",
        "path_details": [
          {
            "id_remapped": 71953,
            "id_original": "775635",
            "type": "USER",
            "name": "User 775635"
          },
          {
            "id_remapped": 242464,
            "id_original": "29679",
            "type": "RECIPE",
            "name": "reeses squares   5 ingredients   no bake  reese s"
          },
          {
            "id_remapped": 19890,
            "id_original": "168423",
            "type": "USER",
            "name": "User 168423"
          },
          {
            "id_remapped": 286973,
            "id_original": "103961",
            "type": "RECIPE",
            "name": "kelly s apple pork chops with stuffing"
          }
        ],
        "contribution_score": 4.808044982088032e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 775635 -> reeses squares   5 ingredients   no bake  reese s -> User 147109 -> kelly s apple pork chops with stuffing",
        "path_details": [
          {
            "id_remapped": 71953,
            "id_original": "775635",
            "type": "USER",
            "name": "User 775635"
          },
          {
            "id_remapped": 242464,
            "id_original": "29679",
            "type": "RECIPE",
            "name": "reeses squares   5 ingredients   no bake  reese s"
          },
          {
            "id_remapped": 17447,
            "id_original": "147109",
            "type": "USER",
            "name": "User 147109"
          },
          {
            "id_remapped": 286973,
            "id_original": "103961",
            "type": "RECIPE",
            "name": "kelly s apple pork chops with stuffing"
          }
        ],
        "contribution_score": 1.297685544565325e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 775635 -> reeses squares   5 ingredients   no bake  reese s -> User 189616 -> kelly s apple pork chops with stuffing",
        "path_details": [
          {
            "id_remapped": 71953,
            "id_original": "775635",
            "type": "USER",
            "name": "User 775635"
          },
          {
            "id_remapped": 242464,
            "id_original": "29679",
            "type": "RECIPE",
            "name": "reeses squares   5 ingredients   no bake  reese s"
          },
          {
            "id_remapped": 22503,
            "id_original": "189616",
            "type": "USER",
            "name": "User 189616"
          },
          {
            "id_remapped": 286973,
            "id_original": "103961",
            "type": "RECIPE",
            "name": "kelly s apple pork chops with stuffing"
          }
        ],
        "contribution_score": 5.163799634298793e-07
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.001353621482849121,
      "prob_f_plus": 0.9507250189781189,
      "fidelity_minus": 0.003036201000213623,
      "prob_f_minus": 0.9490424394607544
    }
  },
  {
    "user_id_remapped": 218242,
    "user_id_original": "2002060836",
    "user_name": "User 2002060836",
    "recommended_item_id_remapped": 183331,
    "recommended_item_id_original": "358622",
    "recommended_item_name": "homemade pancake syrup",
    "score": 2.898268222808838,
    "original_prob": 0.9477607607841492,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 220814,
    "user_id_original": "2002155102",
    "user_name": "User 2002155102",
    "recommended_item_id_remapped": 9396,
    "recommended_item_id_original": "19042",
    "recommended_item_name": "blt nibbles",
    "score": 3.0079126358032227,
    "original_prob": 0.9529302716255188,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2002155102 -> fruit salad with pudding -> User 229850 -> blt nibbles",
        "path_details": [
          {
            "id_remapped": 220814,
            "id_original": "2002155102",
            "type": "USER",
            "name": "User 2002155102"
          },
          {
            "id_remapped": 264754,
            "id_original": "66575",
            "type": "RECIPE",
            "name": "fruit salad with pudding"
          },
          {
            "id_remapped": 27127,
            "id_original": "229850",
            "type": "USER",
            "name": "User 229850"
          },
          {
            "id_remapped": 235966,
            "id_original": "19042",
            "type": "RECIPE",
            "name": "blt nibbles"
          }
        ],
        "contribution_score": 4.9716477493144934e-06
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2002155102 -> fruit salad with pudding -> User 1052873 -> blt nibbles",
        "path_details": [
          {
            "id_remapped": 220814,
            "id_original": "2002155102",
            "type": "USER",
            "name": "User 2002155102"
          },
          {
            "id_remapped": 264754,
            "id_original": "66575",
            "type": "RECIPE",
            "name": "fruit salad with pudding"
          },
          {
            "id_remapped": 87705,
            "id_original": "1052873",
            "type": "USER",
            "name": "User 1052873"
          },
          {
            "id_remapped": 235966,
            "id_original": "19042",
            "type": "RECIPE",
            "name": "blt nibbles"
          }
        ],
        "contribution_score": 3.5150513879307587e-06
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2002155102 -> fruit salad with pudding -> User 33159 -> blt nibbles",
        "path_details": [
          {
            "id_remapped": 220814,
            "id_original": "2002155102",
            "type": "USER",
            "name": "User 2002155102"
          },
          {
            "id_remapped": 264754,
            "id_original": "66575",
            "type": "RECIPE",
            "name": "fruit salad with pudding"
          },
          {
            "id_remapped": 3255,
            "id_original": "33159",
            "type": "USER",
            "name": "User 33159"
          },
          {
            "id_remapped": 235966,
            "id_original": "19042",
            "type": "RECIPE",
            "name": "blt nibbles"
          }
        ],
        "contribution_score": 1.216422532841488e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.030740678310394287,
      "prob_f_plus": 0.9221895933151245,
      "fidelity_minus": 0.0025092363357543945,
      "prob_f_minus": 0.9504210352897644
    }
  },
  {
    "user_id_remapped": 207024,
    "user_id_original": "2001651333",
    "user_name": "User 2001651333",
    "recommended_item_id_remapped": 39691,
    "recommended_item_id_original": "69138",
    "recommended_item_name": "fig preserves",
    "score": 2.9651880264282227,
    "original_prob": 0.9509764313697815,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2001651333 -> fig preserves",
        "path_details": [
          {
            "id_remapped": 207024,
            "id_original": "2001651333",
            "type": "USER",
            "name": "User 2001651333"
          },
          {
            "id_remapped": 266261,
            "id_original": "69138",
            "type": "RECIPE",
            "name": "fig preserves"
          }
        ],
        "contribution_score": 0.7976672053337097
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.06289488077163696,
      "prob_f_plus": 0.8880815505981445,
      "fidelity_minus": -0.0016499757766723633,
      "prob_f_minus": 0.9526264071464539
    }
  },
  {
    "user_id_remapped": 126151,
    "user_id_original": "1882806",
    "user_name": "User 1882806",
    "recommended_item_id_remapped": 99294,
    "recommended_item_id_original": "174723",
    "recommended_item_name": "close to mimi s honey bran muffins",
    "score": 2.7690505981445312,
    "original_prob": 0.9409803152084351,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 1882806 -> weight watchers baked oatmeal -> breakfast -> close to mimi s honey bran muffins",
        "path_details": [
          {
            "id_remapped": 126151,
            "id_original": "1882806",
            "type": "USER",
            "name": "User 1882806"
          },
          {
            "id_remapped": 384025,
            "id_original": "295560",
            "type": "RECIPE",
            "name": "weight watchers baked oatmeal"
          },
          {
            "id_remapped": 473081,
            "id_original": "14874",
            "type": "TAG",
            "name": "breakfast"
          },
          {
            "id_remapped": 325864,
            "id_original": "174723",
            "type": "RECIPE",
            "name": "close to mimi s honey bran muffins"
          }
        ],
        "contribution_score": 0.0008930121423114064
      }
    ],
    "fidelity": {
      "fidelity_plus": -0.0023009777069091797,
      "prob_f_plus": 0.9432812929153442,
      "fidelity_minus": 0.5226243436336517,
      "prob_f_minus": 0.4183559715747833
    }
  },
  {
    "user_id_remapped": 160665,
    "user_id_original": "1800352432",
    "user_name": "User 1800352432",
    "recommended_item_id_remapped": 144459,
    "recommended_item_id_original": "267289",
    "recommended_item_name": "kidd kraddick s famous brown bag turkey",
    "score": 2.972074508666992,
    "original_prob": 0.9512964487075806,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1800352432 -> kidd kraddick s famous brown bag turkey",
        "path_details": [
          {
            "id_remapped": 160665,
            "id_original": "1800352432",
            "type": "USER",
            "name": "User 1800352432"
          },
          {
            "id_remapped": 371029,
            "id_original": "267289",
            "type": "RECIPE",
            "name": "kidd kraddick s famous brown bag turkey"
          }
        ],
        "contribution_score": 0.7691131830215454
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.07472693920135498,
      "prob_f_plus": 0.8765695095062256,
      "fidelity_minus": -0.0011954903602600098,
      "prob_f_minus": 0.9524919390678406
    }
  },
  {
    "user_id_remapped": 25824,
    "user_id_original": "218928",
    "user_name": "User 218928",
    "recommended_item_id_remapped": 220880,
    "recommended_item_id_original": "471979",
    "recommended_item_name": "chicken and potato florentine soup  olive garden",
    "score": 2.8615827560424805,
    "original_prob": 0.9459143877029419,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 210947,
    "user_id_original": "2001791217",
    "user_name": "User 2001791217",
    "recommended_item_id_remapped": 111706,
    "recommended_item_id_original": "198948",
    "recommended_item_name": "basmati rice   indian style",
    "score": 2.9681410789489746,
    "original_prob": 0.9511139392852783,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2001791217 -> basmati rice   indian style",
        "path_details": [
          {
            "id_remapped": 210947,
            "id_original": "2001791217",
            "type": "USER",
            "name": "User 2001791217"
          },
          {
            "id_remapped": 338276,
            "id_original": "198948",
            "type": "RECIPE",
            "name": "basmati rice   indian style"
          }
        ],
        "contribution_score": 0.7910877466201782
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.1232789158821106,
      "prob_f_plus": 0.8278350234031677,
      "fidelity_minus": -0.001481175422668457,
      "prob_f_minus": 0.9525951147079468
    }
  },
  {
    "user_id_remapped": 196214,
    "user_id_original": "2001321112",
    "user_name": "User 2001321112",
    "recommended_item_id_remapped": 15015,
    "recommended_item_id_original": "28261",
    "recommended_item_name": "baked macaroni and cheese with stewed tomatoes",
    "score": 2.9867072105407715,
    "original_prob": 0.9519699811935425,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 2001321112 -> boursin cheese  homemade -> cheese -> baked macaroni and cheese with stewed tomatoes",
        "path_details": [
          {
            "id_remapped": 196214,
            "id_original": "2001321112",
            "type": "USER",
            "name": "User 2001321112"
          },
          {
            "id_remapped": 272801,
            "id_original": "80675",
            "type": "RECIPE",
            "name": "boursin cheese  homemade"
          },
          {
            "id_remapped": 473089,
            "id_original": "14882",
            "type": "TAG",
            "name": "cheese"
          },
          {
            "id_remapped": 241585,
            "id_original": "28261",
            "type": "RECIPE",
            "name": "baked macaroni and cheese with stewed tomatoes"
          }
        ],
        "contribution_score": 2.015077283169367e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 2001321112 -> boursin cheese  homemade -> novelty -> baked macaroni and cheese with stewed tomatoes",
        "path_details": [
          {
            "id_remapped": 196214,
            "id_original": "2001321112",
            "type": "USER",
            "name": "User 2001321112"
          },
          {
            "id_remapped": 272801,
            "id_original": "80675",
            "type": "RECIPE",
            "name": "boursin cheese  homemade"
          },
          {
            "id_remapped": 473119,
            "id_original": "14912",
            "type": "TAG",
            "name": "novelty"
          },
          {
            "id_remapped": 241585,
            "id_original": "28261",
            "type": "RECIPE",
            "name": "baked macaroni and cheese with stewed tomatoes"
          }
        ],
        "contribution_score": 1.1415777479728601e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.14285880327224731,
      "prob_f_plus": 0.8091111779212952,
      "fidelity_minus": 0.49036160111427307,
      "prob_f_minus": 0.4616083800792694
    }
  },
  {
    "user_id_remapped": 126599,
    "user_id_original": "1896042",
    "user_name": "User 1896042",
    "recommended_item_id_remapped": 103630,
    "recommended_item_id_original": "183150",
    "recommended_item_name": "loquat jam",
    "score": 2.865321636199951,
    "original_prob": 0.9461053609848022,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 187201,
    "user_id_original": "2000911443",
    "user_name": "User 2000911443",
    "recommended_item_id_remapped": 121355,
    "recommended_item_id_original": "218193",
    "recommended_item_name": "sweet butter bread  bread machine",
    "score": 2.9159350395202637,
    "original_prob": 0.9486286044120789,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 194407,
    "user_id_original": "2001262910",
    "user_name": "User 2001262910",
    "recommended_item_id_remapped": 178018,
    "recommended_item_id_original": "344870",
    "recommended_item_name": "basic vanilla custard",
    "score": 2.9633278846740723,
    "original_prob": 0.9508897066116333,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 36069,
    "user_id_original": "315210",
    "user_name": "User 315210",
    "recommended_item_id_remapped": 2527,
    "recommended_item_id_original": "7404",
    "recommended_item_name": "cinnamon baked apples",
    "score": 2.957059383392334,
    "original_prob": 0.950596034526825,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 315210 -> white pizza or pizza blanca -> User 663997 -> cinnamon baked apples",
        "path_details": [
          {
            "id_remapped": 36069,
            "id_original": "315210",
            "type": "USER",
            "name": "User 315210"
          },
          {
            "id_remapped": 298470,
            "id_original": "123588",
            "type": "RECIPE",
            "name": "white pizza or pizza blanca"
          },
          {
            "id_remapped": 64469,
            "id_original": "663997",
            "type": "USER",
            "name": "User 663997"
          },
          {
            "id_remapped": 229097,
            "id_original": "7404",
            "type": "RECIPE",
            "name": "cinnamon baked apples"
          }
        ],
        "contribution_score": 2.8338527589122126e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.004397332668304443,
      "prob_f_plus": 0.9461987018585205,
      "fidelity_minus": 0.006903111934661865,
      "prob_f_minus": 0.9436929225921631
    }
  },
  {
    "user_id_remapped": 131064,
    "user_id_original": "2026158",
    "user_name": "User 2026158",
    "recommended_item_id_remapped": 63183,
    "recommended_item_id_original": "108633",
    "recommended_item_name": "chicken breasts smothered in tomatoes and mozzarella",
    "score": 2.9219651222229004,
    "original_prob": 0.9489216804504395,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 34380,
    "user_id_original": "298350",
    "user_name": "User 298350",
    "recommended_item_id_remapped": 5721,
    "recommended_item_id_original": "13218",
    "recommended_item_name": "beef stew in a pumpkin",
    "score": 2.8472955226898193,
    "original_prob": 0.945178747177124,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 38331,
    "user_id_original": "336111",
    "user_name": "User 336111",
    "recommended_item_id_remapped": 181121,
    "recommended_item_id_original": "352893",
    "recommended_item_name": "self iced date cake",
    "score": 2.8783397674560547,
    "original_prob": 0.9467652440071106,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 108993,
    "user_id_original": "1486462",
    "user_name": "User 1486462",
    "recommended_item_id_remapped": 65696,
    "recommended_item_id_original": "112813",
    "recommended_item_name": "stew meat",
    "score": 2.9536194801330566,
    "original_prob": 0.9504342675209045,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1486462 -> stew meat",
        "path_details": [
          {
            "id_remapped": 108993,
            "id_original": "1486462",
            "type": "USER",
            "name": "User 1486462"
          },
          {
            "id_remapped": 292266,
            "id_original": "112813",
            "type": "RECIPE",
            "name": "stew meat"
          }
        ],
        "contribution_score": 0.7995900511741638
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.19694042205810547,
      "prob_f_plus": 0.7534938454627991,
      "fidelity_minus": -0.001977086067199707,
      "prob_f_minus": 0.9524113535881042
    }
  },
  {
    "user_id_remapped": 115081,
    "user_id_original": "1613425",
    "user_name": "User 1613425",
    "recommended_item_id_remapped": 12720,
    "recommended_item_id_original": "24464",
    "recommended_item_name": "spicy african peanut soup",
    "score": 2.9599647521972656,
    "original_prob": 0.9507323503494263,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 92488,
    "user_id_original": "1154318",
    "user_name": "User 1154318",
    "recommended_item_id_remapped": 158195,
    "recommended_item_id_original": "297254",
    "recommended_item_name": "mueller s baked macaroni and cheese",
    "score": 2.9629340171813965,
    "original_prob": 0.9508712291717529,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1154318 -> mueller s baked macaroni and cheese",
        "path_details": [
          {
            "id_remapped": 92488,
            "id_original": "1154318",
            "type": "USER",
            "name": "User 1154318"
          },
          {
            "id_remapped": 384765,
            "id_original": "297254",
            "type": "RECIPE",
            "name": "mueller s baked macaroni and cheese"
          }
        ],
        "contribution_score": 0.7835982441902161
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.025547027587890625,
      "prob_f_plus": 0.9253242015838623,
      "fidelity_minus": -0.0021982789039611816,
      "prob_f_minus": 0.9530695080757141
    }
  },
  {
    "user_id_remapped": 5283,
    "user_id_original": "48227",
    "user_name": "User 48227",
    "recommended_item_id_remapped": 83263,
    "recommended_item_id_original": "144131",
    "recommended_item_name": "italian antipasto squares",
    "score": 2.8372411727905273,
    "original_prob": 0.9446554183959961,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 48227 -> old fashioned walnut bread -> gifts -> italian antipasto squares",
        "path_details": [
          {
            "id_remapped": 5283,
            "id_original": "48227",
            "type": "USER",
            "name": "User 48227"
          },
          {
            "id_remapped": 237114,
            "id_original": "20867",
            "type": "RECIPE",
            "name": "old fashioned walnut bread"
          },
          {
            "id_remapped": 473177,
            "id_original": "14970",
            "type": "TAG",
            "name": "gifts"
          },
          {
            "id_remapped": 309833,
            "id_original": "144131",
            "type": "RECIPE",
            "name": "italian antipasto squares"
          }
        ],
        "contribution_score": 0.0002190983356937158
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.013559281826019287,
      "prob_f_plus": 0.9310961365699768,
      "fidelity_minus": 0.005189657211303711,
      "prob_f_minus": 0.9394657611846924
    }
  },
  {
    "user_id_remapped": 148009,
    "user_id_original": "2685063",
    "user_name": "User 2685063",
    "recommended_item_id_remapped": 3567,
    "recommended_item_id_original": "9356",
    "recommended_item_name": "motichoor ladoo",
    "score": 2.8817713260650635,
    "original_prob": 0.9469379782676697,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 96106,
    "user_id_original": "1223227",
    "user_name": "User 1223227",
    "recommended_item_id_remapped": 21781,
    "recommended_item_id_original": "39319",
    "recommended_item_name": "extra easy lazy day lasagna",
    "score": 2.889478921890259,
    "original_prob": 0.9473239183425903,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 1223227 -> smoked turkey casserole -> pasta -> extra easy lazy day lasagna",
        "path_details": [
          {
            "id_remapped": 96106,
            "id_original": "1223227",
            "type": "USER",
            "name": "User 1223227"
          },
          {
            "id_remapped": 374002,
            "id_original": "274141",
            "type": "RECIPE",
            "name": "smoked turkey casserole"
          },
          {
            "id_remapped": 473187,
            "id_original": "14980",
            "type": "TAG",
            "name": "pasta"
          },
          {
            "id_remapped": 248351,
            "id_original": "39319",
            "type": "RECIPE",
            "name": "extra easy lazy day lasagna"
          }
        ],
        "contribution_score": 0.00039241781298474424
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 1223227 -> smoked turkey casserole -> one-dish-meal -> extra easy lazy day lasagna",
        "path_details": [
          {
            "id_remapped": 96106,
            "id_original": "1223227",
            "type": "USER",
            "name": "User 1223227"
          },
          {
            "id_remapped": 374002,
            "id_original": "274141",
            "type": "RECIPE",
            "name": "smoked turkey casserole"
          },
          {
            "id_remapped": 473147,
            "id_original": "14940",
            "type": "TAG",
            "name": "one-dish-meal"
          },
          {
            "id_remapped": 248351,
            "id_original": "39319",
            "type": "RECIPE",
            "name": "extra easy lazy day lasagna"
          }
        ],
        "contribution_score": 0.0002762183782861857
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 1223227 -> mexican ham and bean soup -> weeknight -> extra easy lazy day lasagna",
        "path_details": [
          {
            "id_remapped": 96106,
            "id_original": "1223227",
            "type": "USER",
            "name": "User 1223227"
          },
          {
            "id_remapped": 255558,
            "id_original": "51626",
            "type": "RECIPE",
            "name": "mexican ham and bean soup"
          },
          {
            "id_remapped": 473099,
            "id_original": "14892",
            "type": "TAG",
            "name": "weeknight"
          },
          {
            "id_remapped": 248351,
            "id_original": "39319",
            "type": "RECIPE",
            "name": "extra easy lazy day lasagna"
          }
        ],
        "contribution_score": 0.00011452120088270295
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.009605050086975098,
      "prob_f_plus": 0.9377188682556152,
      "fidelity_minus": 0.27536916732788086,
      "prob_f_minus": 0.6719547510147095
    }
  },
  {
    "user_id_remapped": 206882,
    "user_id_original": "2001646908",
    "user_name": "User 2001646908",
    "recommended_item_id_remapped": 85008,
    "recommended_item_id_original": "147391",
    "recommended_item_name": "beans 101   beans and cornbread",
    "score": 2.9667978286743164,
    "original_prob": 0.9510514140129089,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2001646908 -> fried zucchini batter -> User 91584 -> beans 101   beans and cornbread",
        "path_details": [
          {
            "id_remapped": 206882,
            "id_original": "2001646908",
            "type": "USER",
            "name": "User 2001646908"
          },
          {
            "id_remapped": 302531,
            "id_original": "131300",
            "type": "RECIPE",
            "name": "fried zucchini batter"
          },
          {
            "id_remapped": 10753,
            "id_original": "91584",
            "type": "USER",
            "name": "User 91584"
          },
          {
            "id_remapped": 311578,
            "id_original": "147391",
            "type": "RECIPE",
            "name": "beans 101   beans and cornbread"
          }
        ],
        "contribution_score": 1.3518689596234875e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0730634331703186,
      "prob_f_plus": 0.8779879808425903,
      "fidelity_minus": 0.011509478092193604,
      "prob_f_minus": 0.9395419359207153
    }
  },
  {
    "user_id_remapped": 214548,
    "user_id_original": "2001924908",
    "user_name": "User 2001924908",
    "recommended_item_id_remapped": 77667,
    "recommended_item_id_original": "134349",
    "recommended_item_name": "almost fried plantains   virtually fat free",
    "score": 2.959962844848633,
    "original_prob": 0.9507322311401367,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2001924908 -> soft peanut butter cookies -> User 37449 -> almost fried plantains   virtually fat free",
        "path_details": [
          {
            "id_remapped": 214548,
            "id_original": "2001924908",
            "type": "USER",
            "name": "User 2001924908"
          },
          {
            "id_remapped": 341716,
            "id_original": "205890",
            "type": "RECIPE",
            "name": "soft peanut butter cookies"
          },
          {
            "id_remapped": 3826,
            "id_original": "37449",
            "type": "USER",
            "name": "User 37449"
          },
          {
            "id_remapped": 304237,
            "id_original": "134349",
            "type": "RECIPE",
            "name": "almost fried plantains   virtually fat free"
          }
        ],
        "contribution_score": 1.1235270092530277e-08
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.006325364112854004,
      "prob_f_plus": 0.9444068670272827,
      "fidelity_minus": 0.024999380111694336,
      "prob_f_minus": 0.9257328510284424
    }
  },
  {
    "user_id_remapped": 58074,
    "user_id_original": "573230",
    "user_name": "User 573230",
    "recommended_item_id_remapped": 63032,
    "recommended_item_id_original": "108383",
    "recommended_item_name": "joanne s creamy bruschetta",
    "score": 2.9896106719970703,
    "original_prob": 0.952102541923523,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 573230 -> creamy cajun chicken pasta -> User 603504 -> joanne s creamy bruschetta",
        "path_details": [
          {
            "id_remapped": 58074,
            "id_original": "573230",
            "type": "USER",
            "name": "User 573230"
          },
          {
            "id_remapped": 248209,
            "id_original": "39087",
            "type": "RECIPE",
            "name": "creamy cajun chicken pasta"
          },
          {
            "id_remapped": 60357,
            "id_original": "603504",
            "type": "USER",
            "name": "User 603504"
          },
          {
            "id_remapped": 289602,
            "id_original": "108383",
            "type": "RECIPE",
            "name": "joanne s creamy bruschetta"
          }
        ],
        "contribution_score": 0.000621531554091637
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 573230 -> creamy cajun chicken pasta -> User 868551 -> joanne s creamy bruschetta",
        "path_details": [
          {
            "id_remapped": 58074,
            "id_original": "573230",
            "type": "USER",
            "name": "User 573230"
          },
          {
            "id_remapped": 248209,
            "id_original": "39087",
            "type": "RECIPE",
            "name": "creamy cajun chicken pasta"
          },
          {
            "id_remapped": 77885,
            "id_original": "868551",
            "type": "USER",
            "name": "User 868551"
          },
          {
            "id_remapped": 289602,
            "id_original": "108383",
            "type": "RECIPE",
            "name": "joanne s creamy bruschetta"
          }
        ],
        "contribution_score": 0.00047118729503607313
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 573230 -> creamy cajun chicken pasta -> User 1506604 -> joanne s creamy bruschetta",
        "path_details": [
          {
            "id_remapped": 58074,
            "id_original": "573230",
            "type": "USER",
            "name": "User 573230"
          },
          {
            "id_remapped": 248209,
            "id_original": "39087",
            "type": "RECIPE",
            "name": "creamy cajun chicken pasta"
          },
          {
            "id_remapped": 109822,
            "id_original": "1506604",
            "type": "USER",
            "name": "User 1506604"
          },
          {
            "id_remapped": 289602,
            "id_original": "108383",
            "type": "RECIPE",
            "name": "joanne s creamy bruschetta"
          }
        ],
        "contribution_score": 7.854293243899087e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.20612752437591553,
      "prob_f_plus": 0.7459750175476074,
      "fidelity_minus": 0.0018615126609802246,
      "prob_f_minus": 0.9502410292625427
    }
  },
  {
    "user_id_remapped": 3350,
    "user_id_original": "34196",
    "user_name": "User 34196",
    "recommended_item_id_remapped": 81918,
    "recommended_item_id_original": "141679",
    "recommended_item_name": "lime bars",
    "score": 2.91652250289917,
    "original_prob": 0.9486571550369263,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 88664,
    "user_id_original": "1073874",
    "user_name": "User 1073874",
    "recommended_item_id_remapped": 52288,
    "recommended_item_id_original": "90765",
    "recommended_item_name": "famous challah",
    "score": 2.986088275909424,
    "original_prob": 0.9519416689872742,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1073874 -> panera s cream cheese potato soup -> User 385423 -> famous challah",
        "path_details": [
          {
            "id_remapped": 88664,
            "id_original": "1073874",
            "type": "USER",
            "name": "User 1073874"
          },
          {
            "id_remapped": 313451,
            "id_original": "150863",
            "type": "RECIPE",
            "name": "panera s cream cheese potato soup"
          },
          {
            "id_remapped": 43402,
            "id_original": "385423",
            "type": "USER",
            "name": "User 385423"
          },
          {
            "id_remapped": 278858,
            "id_original": "90765",
            "type": "RECIPE",
            "name": "famous challah"
          }
        ],
        "contribution_score": 1.317876371339034e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1073874 -> panera s cream cheese potato soup -> User 457661 -> famous challah",
        "path_details": [
          {
            "id_remapped": 88664,
            "id_original": "1073874",
            "type": "USER",
            "name": "User 1073874"
          },
          {
            "id_remapped": 313451,
            "id_original": "150863",
            "type": "RECIPE",
            "name": "panera s cream cheese potato soup"
          },
          {
            "id_remapped": 49081,
            "id_original": "457661",
            "type": "USER",
            "name": "User 457661"
          },
          {
            "id_remapped": 278858,
            "id_original": "90765",
            "type": "RECIPE",
            "name": "famous challah"
          }
        ],
        "contribution_score": 5.488941994897954e-06
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1073874 -> panera s cream cheese potato soup -> User 52262 -> famous challah",
        "path_details": [
          {
            "id_remapped": 88664,
            "id_original": "1073874",
            "type": "USER",
            "name": "User 1073874"
          },
          {
            "id_remapped": 313451,
            "id_original": "150863",
            "type": "RECIPE",
            "name": "panera s cream cheese potato soup"
          },
          {
            "id_remapped": 5890,
            "id_original": "52262",
            "type": "USER",
            "name": "User 52262"
          },
          {
            "id_remapped": 278858,
            "id_original": "90765",
            "type": "RECIPE",
            "name": "famous challah"
          }
        ],
        "contribution_score": 4.920589834555699e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0013415217399597168,
      "prob_f_plus": 0.9506001472473145,
      "fidelity_minus": 0.0012813210487365723,
      "prob_f_minus": 0.9506603479385376
    }
  },
  {
    "user_id_remapped": 143053,
    "user_id_original": "2458578",
    "user_name": "User 2458578",
    "recommended_item_id_remapped": 86587,
    "recommended_item_id_original": "150355",
    "recommended_item_name": "sour cream and 7 up biscuits",
    "score": 2.9715542793273926,
    "original_prob": 0.9512723684310913,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2458578 -> creamy cucumber salad -> User 128473 -> sour cream and 7 up biscuits",
        "path_details": [
          {
            "id_remapped": 143053,
            "id_original": "2458578",
            "type": "USER",
            "name": "User 2458578"
          },
          {
            "id_remapped": 294323,
            "id_original": "116236",
            "type": "RECIPE",
            "name": "creamy cucumber salad"
          },
          {
            "id_remapped": 15168,
            "id_original": "128473",
            "type": "USER",
            "name": "User 128473"
          },
          {
            "id_remapped": 313157,
            "id_original": "150355",
            "type": "RECIPE",
            "name": "sour cream and 7 up biscuits"
          }
        ],
        "contribution_score": 5.645561967812668e-09
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0009895563125610352,
      "prob_f_plus": 0.9502828121185303,
      "fidelity_minus": 0.025022804737091064,
      "prob_f_minus": 0.9262495636940002
    }
  },
  {
    "user_id_remapped": 92466,
    "user_id_original": "1153796",
    "user_name": "User 1153796",
    "recommended_item_id_remapped": 118347,
    "recommended_item_id_original": "212286",
    "recommended_item_name": "paula deen s mac and cheese",
    "score": 2.976733684539795,
    "original_prob": 0.9515119194984436,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 204608,
    "user_id_original": "2001573912",
    "user_name": "User 2001573912",
    "recommended_item_id_remapped": 54851,
    "recommended_item_id_original": "94964",
    "recommended_item_name": "marinated cucumbers  onions  and tomatoes",
    "score": 2.9811651706695557,
    "original_prob": 0.9517159461975098,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2001573912 -> marinated cucumbers  onions  and tomatoes",
        "path_details": [
          {
            "id_remapped": 204608,
            "id_original": "2001573912",
            "type": "USER",
            "name": "User 2001573912"
          },
          {
            "id_remapped": 281421,
            "id_original": "94964",
            "type": "RECIPE",
            "name": "marinated cucumbers  onions  and tomatoes"
          }
        ],
        "contribution_score": 0.7966668009757996
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.16612356901168823,
      "prob_f_plus": 0.7855923771858215,
      "fidelity_minus": -0.00015658140182495117,
      "prob_f_minus": 0.9518725275993347
    }
  },
  {
    "user_id_remapped": 109592,
    "user_id_original": "1501478",
    "user_name": "User 1501478",
    "recommended_item_id_remapped": 146142,
    "recommended_item_id_original": "271050",
    "recommended_item_name": "butternut squash   leeks",
    "score": 2.874636173248291,
    "original_prob": 0.9465782642364502,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 87832,
    "user_id_original": "1055474",
    "user_name": "User 1055474",
    "recommended_item_id_remapped": 88593,
    "recommended_item_id_original": "154142",
    "recommended_item_name": "classic san francisco sourdough bread",
    "score": 2.976165294647217,
    "original_prob": 0.9514856934547424,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 192418,
    "user_id_original": "2001181537",
    "user_name": "User 2001181537",
    "recommended_item_id_remapped": 12705,
    "recommended_item_id_original": "24440",
    "recommended_item_name": "honey rhubarb muffins",
    "score": 2.850667953491211,
    "original_prob": 0.9453532099723816,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 214584,
    "user_id_original": "2001926510",
    "user_name": "User 2001926510",
    "recommended_item_id_remapped": 28666,
    "recommended_item_id_original": "51104",
    "recommended_item_name": "peanut butter chocolate chip cookies",
    "score": 2.9767823219299316,
    "original_prob": 0.9515141844749451,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2001926510 -> peanut butter chocolate chip cookies",
        "path_details": [
          {
            "id_remapped": 214584,
            "id_original": "2001926510",
            "type": "USER",
            "name": "User 2001926510"
          },
          {
            "id_remapped": 255236,
            "id_original": "51104",
            "type": "RECIPE",
            "name": "peanut butter chocolate chip cookies"
          }
        ],
        "contribution_score": 0.7944315075874329
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.19839876890182495,
      "prob_f_plus": 0.7531154155731201,
      "fidelity_minus": -0.0005361437797546387,
      "prob_f_minus": 0.9520503282546997
    }
  },
  {
    "user_id_remapped": 200494,
    "user_id_original": "2001446407",
    "user_name": "User 2001446407",
    "recommended_item_id_remapped": 59585,
    "recommended_item_id_original": "102631",
    "recommended_item_name": "best low carb bread  bread machine",
    "score": 2.9895403385162354,
    "original_prob": 0.9520993232727051,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2001446407 -> best low carb bread  bread machine",
        "path_details": [
          {
            "id_remapped": 200494,
            "id_original": "2001446407",
            "type": "USER",
            "name": "User 2001446407"
          },
          {
            "id_remapped": 286155,
            "id_original": "102631",
            "type": "RECIPE",
            "name": "best low carb bread  bread machine"
          }
        ],
        "contribution_score": 0.7930690050125122
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.2217940092086792,
      "prob_f_plus": 0.7303053140640259,
      "fidelity_minus": 0.0004667043685913086,
      "prob_f_minus": 0.9516326189041138
    }
  },
  {
    "user_id_remapped": 215308,
    "user_id_original": "2001952228",
    "user_name": "User 2001952228",
    "recommended_item_id_remapped": 10161,
    "recommended_item_id_original": "20235",
    "recommended_item_name": "chinese wontons",
    "score": 2.977454423904419,
    "original_prob": 0.9515451788902283,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 2001952228 -> chicken pot pie with 2 crusts -> User 140132 -> chinese wontons",
        "path_details": [
          {
            "id_remapped": 215308,
            "id_original": "2001952228",
            "type": "USER",
            "name": "User 2001952228"
          },
          {
            "id_remapped": 291654,
            "id_original": "111777",
            "type": "RECIPE",
            "name": "chicken pot pie with 2 crusts"
          },
          {
            "id_remapped": 16523,
            "id_original": "140132",
            "type": "USER",
            "name": "User 140132"
          },
          {
            "id_remapped": 236731,
            "id_original": "20235",
            "type": "RECIPE",
            "name": "chinese wontons"
          }
        ],
        "contribution_score": 2.0790614582906286e-07
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.13130956888198853,
      "prob_f_plus": 0.8202356100082397,
      "fidelity_minus": 0.009794116020202637,
      "prob_f_minus": 0.9417510628700256
    }
  },
  {
    "user_id_remapped": 160031,
    "user_id_original": "1800228041",
    "user_name": "User 1800228041",
    "recommended_item_id_remapped": 1822,
    "recommended_item_id_original": "4957",
    "recommended_item_name": "crock pot chicken taco meat",
    "score": 2.9889235496520996,
    "original_prob": 0.9520711898803711,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1800228041 -> olive garden copycat zuppa toscana -> User 537578 -> crock pot chicken taco meat",
        "path_details": [
          {
            "id_remapped": 160031,
            "id_original": "1800228041",
            "type": "USER",
            "name": "User 1800228041"
          },
          {
            "id_remapped": 247749,
            "id_original": "38298",
            "type": "RECIPE",
            "name": "olive garden copycat zuppa toscana"
          },
          {
            "id_remapped": 55433,
            "id_original": "537578",
            "type": "USER",
            "name": "User 537578"
          },
          {
            "id_remapped": 228392,
            "id_original": "4957",
            "type": "RECIPE",
            "name": "crock pot chicken taco meat"
          }
        ],
        "contribution_score": 0.00043051882087930264
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1800228041 -> olive garden copycat zuppa toscana -> User 113517 -> crock pot chicken taco meat",
        "path_details": [
          {
            "id_remapped": 160031,
            "id_original": "1800228041",
            "type": "USER",
            "name": "User 1800228041"
          },
          {
            "id_remapped": 247749,
            "id_original": "38298",
            "type": "RECIPE",
            "name": "olive garden copycat zuppa toscana"
          },
          {
            "id_remapped": 13348,
            "id_original": "113517",
            "type": "USER",
            "name": "User 113517"
          },
          {
            "id_remapped": 228392,
            "id_original": "4957",
            "type": "RECIPE",
            "name": "crock pot chicken taco meat"
          }
        ],
        "contribution_score": 0.0003712328808720656
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1800228041 -> olive garden copycat zuppa toscana -> User 122577 -> crock pot chicken taco meat",
        "path_details": [
          {
            "id_remapped": 160031,
            "id_original": "1800228041",
            "type": "USER",
            "name": "User 1800228041"
          },
          {
            "id_remapped": 247749,
            "id_original": "38298",
            "type": "RECIPE",
            "name": "olive garden copycat zuppa toscana"
          },
          {
            "id_remapped": 14495,
            "id_original": "122577",
            "type": "USER",
            "name": "User 122577"
          },
          {
            "id_remapped": 228392,
            "id_original": "4957",
            "type": "RECIPE",
            "name": "crock pot chicken taco meat"
          }
        ],
        "contribution_score": 0.0001624138503469386
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.18480241298675537,
      "prob_f_plus": 0.7672687768936157,
      "fidelity_minus": 0.0015659332275390625,
      "prob_f_minus": 0.950505256652832
    }
  },
  {
    "user_id_remapped": 36716,
    "user_id_original": "321227",
    "user_name": "User 321227",
    "recommended_item_id_remapped": 119127,
    "recommended_item_id_original": "213910",
    "recommended_item_name": "guava cream cheese filling",
    "score": 2.9697160720825195,
    "original_prob": 0.9511870741844177,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 26120,
    "user_id_original": "221716",
    "user_name": "User 221716",
    "recommended_item_id_remapped": 132393,
    "recommended_item_id_original": "241139",
    "recommended_item_name": "blue ribbon dill pickles",
    "score": 2.9616503715515137,
    "original_prob": 0.9508112072944641,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 221716 -> amaretto biscotti with almonds -> to-go -> blue ribbon dill pickles",
        "path_details": [
          {
            "id_remapped": 26120,
            "id_original": "221716",
            "type": "USER",
            "name": "User 221716"
          },
          {
            "id_remapped": 282515,
            "id_original": "96750",
            "type": "RECIPE",
            "name": "amaretto biscotti with almonds"
          },
          {
            "id_remapped": 473139,
            "id_original": "14932",
            "type": "TAG",
            "name": "to-go"
          },
          {
            "id_remapped": 358963,
            "id_original": "241139",
            "type": "RECIPE",
            "name": "blue ribbon dill pickles"
          }
        ],
        "contribution_score": 1.4759289654191768e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 221716 -> amaretto biscotti with almonds -> gifts -> blue ribbon dill pickles",
        "path_details": [
          {
            "id_remapped": 26120,
            "id_original": "221716",
            "type": "USER",
            "name": "User 221716"
          },
          {
            "id_remapped": 282515,
            "id_original": "96750",
            "type": "RECIPE",
            "name": "amaretto biscotti with almonds"
          },
          {
            "id_remapped": 473177,
            "id_original": "14970",
            "type": "TAG",
            "name": "gifts"
          },
          {
            "id_remapped": 358963,
            "id_original": "241139",
            "type": "RECIPE",
            "name": "blue ribbon dill pickles"
          }
        ],
        "contribution_score": 1.4316022873270824e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 221716 -> chocolate  chocolate  chocolate  bundt cake with chocolate glaze -> for-large-groups -> blue ribbon dill pickles",
        "path_details": [
          {
            "id_remapped": 26120,
            "id_original": "221716",
            "type": "USER",
            "name": "User 221716"
          },
          {
            "id_remapped": 240958,
            "id_original": "27144",
            "type": "RECIPE",
            "name": "chocolate  chocolate  chocolate  bundt cake with chocolate glaze"
          },
          {
            "id_remapped": 473188,
            "id_original": "14981",
            "type": "TAG",
            "name": "for-large-groups"
          },
          {
            "id_remapped": 358963,
            "id_original": "241139",
            "type": "RECIPE",
            "name": "blue ribbon dill pickles"
          }
        ],
        "contribution_score": 1.1269468799511327e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.002114593982696533,
      "prob_f_plus": 0.9486966133117676,
      "fidelity_minus": 0.03343379497528076,
      "prob_f_minus": 0.9173774123191833
    }
  },
  {
    "user_id_remapped": 151796,
    "user_id_original": "2819666",
    "user_name": "User 2819666",
    "recommended_item_id_remapped": 119161,
    "recommended_item_id_original": "213970",
    "recommended_item_name": "spelt biscuits",
    "score": 2.9479146003723145,
    "original_prob": 0.950164794921875,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2819666 -> spelt biscuits",
        "path_details": [
          {
            "id_remapped": 151796,
            "id_original": "2819666",
            "type": "USER",
            "name": "User 2819666"
          },
          {
            "id_remapped": 345731,
            "id_original": "213970",
            "type": "RECIPE",
            "name": "spelt biscuits"
          }
        ],
        "contribution_score": 0.8187590837478638
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.010829806327819824,
      "prob_f_plus": 0.9393349885940552,
      "fidelity_minus": -0.002770662307739258,
      "prob_f_minus": 0.9529354572296143
    }
  },
  {
    "user_id_remapped": 85832,
    "user_id_original": "1012385",
    "user_name": "User 1012385",
    "recommended_item_id_remapped": 17393,
    "recommended_item_id_original": "32062",
    "recommended_item_name": "chile lovers spicy three meat chili",
    "score": 2.9448533058166504,
    "original_prob": 0.9500196576118469,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1012385 -> chile lovers spicy three meat chili",
        "path_details": [
          {
            "id_remapped": 85832,
            "id_original": "1012385",
            "type": "USER",
            "name": "User 1012385"
          },
          {
            "id_remapped": 243963,
            "id_original": "32062",
            "type": "RECIPE",
            "name": "chile lovers spicy three meat chili"
          }
        ],
        "contribution_score": 0.8179324269294739
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.11425495147705078,
      "prob_f_plus": 0.8357647061347961,
      "fidelity_minus": -0.002535581588745117,
      "prob_f_minus": 0.952555239200592
    }
  },
  {
    "user_id_remapped": 88303,
    "user_id_original": "1064955",
    "user_name": "User 1064955",
    "recommended_item_id_remapped": 86410,
    "recommended_item_id_original": "149981",
    "recommended_item_name": "pierogies and kielbasa skillet casserole",
    "score": 2.958914279937744,
    "original_prob": 0.9506831169128418,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1064955 -> brown gravy mix -> User 126440 -> pierogies and kielbasa skillet casserole",
        "path_details": [
          {
            "id_remapped": 88303,
            "id_original": "1064955",
            "type": "USER",
            "name": "User 1064955"
          },
          {
            "id_remapped": 312920,
            "id_original": "149886",
            "type": "RECIPE",
            "name": "brown gravy mix"
          },
          {
            "id_remapped": 15027,
            "id_original": "126440",
            "type": "USER",
            "name": "User 126440"
          },
          {
            "id_remapped": 312980,
            "id_original": "149981",
            "type": "RECIPE",
            "name": "pierogies and kielbasa skillet casserole"
          }
        ],
        "contribution_score": 2.0898800386908262e-07
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.026483476161956787,
      "prob_f_plus": 0.924199640750885,
      "fidelity_minus": 0.04235982894897461,
      "prob_f_minus": 0.9083232879638672
    }
  },
  {
    "user_id_remapped": 20963,
    "user_id_original": "177044",
    "user_name": "User 177044",
    "recommended_item_id_remapped": 91153,
    "recommended_item_id_original": "159253",
    "recommended_item_name": "bisquick chicken fingers",
    "score": 2.948183536529541,
    "original_prob": 0.9501774907112122,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> INGREDIENT -> RECIPE",
        "path_description": "User 177044 -> 2 alarm chili -> garlic salt -> bisquick chicken fingers",
        "path_details": [
          {
            "id_remapped": 20963,
            "id_original": "177044",
            "type": "USER",
            "name": "User 177044"
          },
          {
            "id_remapped": 248325,
            "id_original": "39280",
            "type": "RECIPE",
            "name": "2 alarm chili"
          },
          {
            "id_remapped": 458952,
            "id_original": "745",
            "type": "INGREDIENT",
            "name": "garlic salt"
          },
          {
            "id_remapped": 317723,
            "id_original": "159253",
            "type": "RECIPE",
            "name": "bisquick chicken fingers"
          }
        ],
        "contribution_score": 0.00018196940183913822
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.07058829069137573,
      "prob_f_plus": 0.8795892000198364,
      "fidelity_minus": 0.4244077801704407,
      "prob_f_minus": 0.5257697105407715
    }
  },
  {
    "user_id_remapped": 135827,
    "user_id_original": "2198371",
    "user_name": "User 2198371",
    "recommended_item_id_remapped": 82322,
    "recommended_item_id_original": "142387",
    "recommended_item_name": "the best bread machine challah",
    "score": 2.9591641426086426,
    "original_prob": 0.9506948590278625,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2198371 -> the best bread machine challah",
        "path_details": [
          {
            "id_remapped": 135827,
            "id_original": "2198371",
            "type": "USER",
            "name": "User 2198371"
          },
          {
            "id_remapped": 308892,
            "id_original": "142387",
            "type": "RECIPE",
            "name": "the best bread machine challah"
          }
        ],
        "contribution_score": 0.7917855381965637
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.11589652299880981,
      "prob_f_plus": 0.8347983360290527,
      "fidelity_minus": -0.0018628239631652832,
      "prob_f_minus": 0.9525576829910278
    }
  },
  {
    "user_id_remapped": 1415,
    "user_id_original": "19021",
    "user_name": "User 19021",
    "recommended_item_id_remapped": 85217,
    "recommended_item_id_original": "147767",
    "recommended_item_name": "fiesta lime chicken",
    "score": 2.900934934616089,
    "original_prob": 0.9478926062583923,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 6189,
    "user_id_original": "54048",
    "user_name": "User 54048",
    "recommended_item_id_remapped": 30680,
    "recommended_item_id_original": "54328",
    "recommended_item_name": "panko fried oysters for two",
    "score": 2.9804153442382812,
    "original_prob": 0.9516814947128296,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 54048 -> horseradish chive butter -> appetizers -> panko fried oysters for two",
        "path_details": [
          {
            "id_remapped": 6189,
            "id_original": "54048",
            "type": "USER",
            "name": "User 54048"
          },
          {
            "id_remapped": 243458,
            "id_original": "31262",
            "type": "RECIPE",
            "name": "horseradish chive butter"
          },
          {
            "id_remapped": 473109,
            "id_original": "14902",
            "type": "TAG",
            "name": "appetizers"
          },
          {
            "id_remapped": 257250,
            "id_original": "54328",
            "type": "RECIPE",
            "name": "panko fried oysters for two"
          }
        ],
        "contribution_score": 5.458201009482204e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 54048 -> stuffed tomato -> appetizers -> panko fried oysters for two",
        "path_details": [
          {
            "id_remapped": 6189,
            "id_original": "54048",
            "type": "USER",
            "name": "User 54048"
          },
          {
            "id_remapped": 247407,
            "id_original": "37759",
            "type": "RECIPE",
            "name": "stuffed tomato"
          },
          {
            "id_remapped": 473109,
            "id_original": "14902",
            "type": "TAG",
            "name": "appetizers"
          },
          {
            "id_remapped": 257250,
            "id_original": "54328",
            "type": "RECIPE",
            "name": "panko fried oysters for two"
          }
        ],
        "contribution_score": 3.9471069413609404e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 54048 -> baked spam -> southern-united-states -> panko fried oysters for two",
        "path_details": [
          {
            "id_remapped": 6189,
            "id_original": "54048",
            "type": "USER",
            "name": "User 54048"
          },
          {
            "id_remapped": 237447,
            "id_original": "21391",
            "type": "RECIPE",
            "name": "baked spam"
          },
          {
            "id_remapped": 473225,
            "id_original": "15018",
            "type": "TAG",
            "name": "southern-united-states"
          },
          {
            "id_remapped": 257250,
            "id_original": "54328",
            "type": "RECIPE",
            "name": "panko fried oysters for two"
          }
        ],
        "contribution_score": 2.774680086512172e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": -0.00021153688430786133,
      "prob_f_plus": 0.9518930315971375,
      "fidelity_minus": 0.2942920923233032,
      "prob_f_minus": 0.6573894023895264
    }
  },
  {
    "user_id_remapped": 215723,
    "user_id_original": "2001966665",
    "user_name": "User 2001966665",
    "recommended_item_id_remapped": 195271,
    "recommended_item_id_original": "388680",
    "recommended_item_name": "bread machine condensed milk sweet bread",
    "score": 2.895881175994873,
    "original_prob": 0.9476425051689148,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 105949,
    "user_id_original": "1416949",
    "user_name": "User 1416949",
    "recommended_item_id_remapped": 205706,
    "recommended_item_id_original": "420478",
    "recommended_item_name": "vegetable skewers  ww",
    "score": 2.7812631130218506,
    "original_prob": 0.9416548609733582,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 174245,
    "user_id_original": "2000164295",
    "user_name": "User 2000164295",
    "recommended_item_id_remapped": 101182,
    "recommended_item_id_original": "178370",
    "recommended_item_name": "simple vanilla cupcakes",
    "score": 2.990267038345337,
    "original_prob": 0.9521324634552002,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2000164295 -> simple vanilla cupcakes",
        "path_details": [
          {
            "id_remapped": 174245,
            "id_original": "2000164295",
            "type": "USER",
            "name": "User 2000164295"
          },
          {
            "id_remapped": 327752,
            "id_original": "178370",
            "type": "RECIPE",
            "name": "simple vanilla cupcakes"
          }
        ],
        "contribution_score": 0.791531503200531
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.20199787616729736,
      "prob_f_plus": 0.7501345872879028,
      "fidelity_minus": 0.000532984733581543,
      "prob_f_minus": 0.9515994787216187
    }
  },
  {
    "user_id_remapped": 53175,
    "user_id_original": "509357",
    "user_name": "User 509357",
    "recommended_item_id_remapped": 146338,
    "recommended_item_id_original": "271496",
    "recommended_item_name": "cantonese lobster",
    "score": 2.8994381427764893,
    "original_prob": 0.9478186368942261,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 76646,
    "user_id_original": "848766",
    "user_name": "User 848766",
    "recommended_item_id_remapped": 9438,
    "recommended_item_id_original": "19101",
    "recommended_item_name": "chocolate dessert crepes",
    "score": 2.860217571258545,
    "original_prob": 0.9458444118499756,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 848766 -> samosa chickpea -> weeknight -> chocolate dessert crepes",
        "path_details": [
          {
            "id_remapped": 76646,
            "id_original": "848766",
            "type": "USER",
            "name": "User 848766"
          },
          {
            "id_remapped": 237005,
            "id_original": "20683",
            "type": "RECIPE",
            "name": "samosa chickpea"
          },
          {
            "id_remapped": 473099,
            "id_original": "14892",
            "type": "TAG",
            "name": "weeknight"
          },
          {
            "id_remapped": 236008,
            "id_original": "19101",
            "type": "RECIPE",
            "name": "chocolate dessert crepes"
          }
        ],
        "contribution_score": 0.00032368840877006975
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.015597760677337646,
      "prob_f_plus": 0.9302466511726379,
      "fidelity_minus": 0.6306778192520142,
      "prob_f_minus": 0.3151665925979614
    }
  },
  {
    "user_id_remapped": 162719,
    "user_id_original": "1802427004",
    "user_name": "User 1802427004",
    "recommended_item_id_remapped": 137713,
    "recommended_item_id_original": "252661",
    "recommended_item_name": "blond toffee brownies",
    "score": 2.8911514282226562,
    "original_prob": 0.9474072456359863,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 13969,
    "user_id_original": "118278",
    "user_name": "User 118278",
    "recommended_item_id_remapped": 5183,
    "recommended_item_id_original": "12346",
    "recommended_item_name": "yams a la francaise",
    "score": 2.891150951385498,
    "original_prob": 0.9474072456359863,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 85261,
    "user_id_original": "1001444",
    "user_name": "User 1001444",
    "recommended_item_id_remapped": 89319,
    "recommended_item_id_original": "155543",
    "recommended_item_name": "creamy souper rice",
    "score": 2.8476781845092773,
    "original_prob": 0.9451985955238342,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 13785,
    "user_id_original": "116948",
    "user_name": "User 116948",
    "recommended_item_id_remapped": 3139,
    "recommended_item_id_original": "8596",
    "recommended_item_name": "olive garden fettuccine alfredo",
    "score": 3.005258083343506,
    "original_prob": 0.9528111219406128,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 116948 -> sirloin tips -> User 709548 -> olive garden fettuccine alfredo",
        "path_details": [
          {
            "id_remapped": 13785,
            "id_original": "116948",
            "type": "USER",
            "name": "User 116948"
          },
          {
            "id_remapped": 243192,
            "id_original": "30864",
            "type": "RECIPE",
            "name": "sirloin tips"
          },
          {
            "id_remapped": 67406,
            "id_original": "709548",
            "type": "USER",
            "name": "User 709548"
          },
          {
            "id_remapped": 229709,
            "id_original": "8596",
            "type": "RECIPE",
            "name": "olive garden fettuccine alfredo"
          }
        ],
        "contribution_score": 7.385082954427109e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 116948 -> honduran liver -> stove-top -> olive garden fettuccine alfredo",
        "path_details": [
          {
            "id_remapped": 13785,
            "id_original": "116948",
            "type": "USER",
            "name": "User 116948"
          },
          {
            "id_remapped": 270708,
            "id_original": "76979",
            "type": "RECIPE",
            "name": "honduran liver"
          },
          {
            "id_remapped": 473090,
            "id_original": "14883",
            "type": "TAG",
            "name": "stove-top"
          },
          {
            "id_remapped": 229709,
            "id_original": "8596",
            "type": "RECIPE",
            "name": "olive garden fettuccine alfredo"
          }
        ],
        "contribution_score": 1.5662307279232983e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 116948 -> crock pot creamy italian chicken -> pasta -> olive garden fettuccine alfredo",
        "path_details": [
          {
            "id_remapped": 13785,
            "id_original": "116948",
            "type": "USER",
            "name": "User 116948"
          },
          {
            "id_remapped": 236490,
            "id_original": "19859",
            "type": "RECIPE",
            "name": "crock pot creamy italian chicken"
          },
          {
            "id_remapped": 473187,
            "id_original": "14980",
            "type": "TAG",
            "name": "pasta"
          },
          {
            "id_remapped": 229709,
            "id_original": "8596",
            "type": "RECIPE",
            "name": "olive garden fettuccine alfredo"
          }
        ],
        "contribution_score": 1.883569284687479e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.005600929260253906,
      "prob_f_plus": 0.9472101926803589,
      "fidelity_minus": 0.04440760612487793,
      "prob_f_minus": 0.9084035158157349
    }
  },
  {
    "user_id_remapped": 37066,
    "user_id_original": "324470",
    "user_name": "User 324470",
    "recommended_item_id_remapped": 373,
    "recommended_item_id_original": "747",
    "recommended_item_name": "zesty fried chicken",
    "score": 2.8846328258514404,
    "original_prob": 0.9470815062522888,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 324470 -> purist s guacamole -> technique -> zesty fried chicken",
        "path_details": [
          {
            "id_remapped": 37066,
            "id_original": "324470",
            "type": "USER",
            "name": "User 324470"
          },
          {
            "id_remapped": 369645,
            "id_original": "264229",
            "type": "RECIPE",
            "name": "purist s guacamole"
          },
          {
            "id_remapped": 473104,
            "id_original": "14897",
            "type": "TAG",
            "name": "technique"
          },
          {
            "id_remapped": 226943,
            "id_original": "747",
            "type": "RECIPE",
            "name": "zesty fried chicken"
          }
        ],
        "contribution_score": 3.13627865045311e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 324470 -> karage tofu -> technique -> zesty fried chicken",
        "path_details": [
          {
            "id_remapped": 37066,
            "id_original": "324470",
            "type": "USER",
            "name": "User 324470"
          },
          {
            "id_remapped": 338576,
            "id_original": "199547",
            "type": "RECIPE",
            "name": "karage tofu"
          },
          {
            "id_remapped": 473104,
            "id_original": "14897",
            "type": "TAG",
            "name": "technique"
          },
          {
            "id_remapped": 226943,
            "id_original": "747",
            "type": "RECIPE",
            "name": "zesty fried chicken"
          }
        ],
        "contribution_score": 1.914998879271094e-05
      },
      {
        "path_structure": "USER -> RECIPE -> INGREDIENT -> RECIPE",
        "path_description": "User 324470 -> slightly spicy  black bean burgers -> cumin -> zesty fried chicken",
        "path_details": [
          {
            "id_remapped": 37066,
            "id_original": "324470",
            "type": "USER",
            "name": "User 324470"
          },
          {
            "id_remapped": 291442,
            "id_original": "111462",
            "type": "RECIPE",
            "name": "slightly spicy  black bean burgers"
          },
          {
            "id_remapped": 458251,
            "id_original": "44",
            "type": "INGREDIENT",
            "name": "cumin"
          },
          {
            "id_remapped": 226943,
            "id_original": "747",
            "type": "RECIPE",
            "name": "zesty fried chicken"
          }
        ],
        "contribution_score": 1.741626434193703e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0020222067832946777,
      "prob_f_plus": 0.9450592994689941,
      "fidelity_minus": 0.31934380531311035,
      "prob_f_minus": 0.6277377009391785
    }
  },
  {
    "user_id_remapped": 68821,
    "user_id_original": "728702",
    "user_name": "User 728702",
    "recommended_item_id_remapped": 221748,
    "recommended_item_id_original": "475446",
    "recommended_item_name": "shrimp nicoise quiche",
    "score": 2.960014820098877,
    "original_prob": 0.9507346749305725,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 728702 -> shrimp nicoise quiche",
        "path_details": [
          {
            "id_remapped": 68821,
            "id_original": "728702",
            "type": "USER",
            "name": "User 728702"
          },
          {
            "id_remapped": 448318,
            "id_original": "475446",
            "type": "RECIPE",
            "name": "shrimp nicoise quiche"
          }
        ],
        "contribution_score": 0.7803404927253723
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.18543362617492676,
      "prob_f_plus": 0.7653010487556458,
      "fidelity_minus": -0.001515209674835205,
      "prob_f_minus": 0.9522498846054077
    }
  },
  {
    "user_id_remapped": 219285,
    "user_id_original": "2002101352",
    "user_name": "User 2002101352",
    "recommended_item_id_remapped": 111228,
    "recommended_item_id_original": "197922",
    "recommended_item_name": "tabbouli   tabouli   tabbouleh salad  parsley salad",
    "score": 2.920978546142578,
    "original_prob": 0.9488738179206848,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2002101352 -> tabbouli   tabouli   tabbouleh salad  parsley salad",
        "path_details": [
          {
            "id_remapped": 219285,
            "id_original": "2002101352",
            "type": "USER",
            "name": "User 2002101352"
          },
          {
            "id_remapped": 337798,
            "id_original": "197922",
            "type": "RECIPE",
            "name": "tabbouli   tabouli   tabbouleh salad  parsley salad"
          }
        ],
        "contribution_score": 0.7878943085670471
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.1207846999168396,
      "prob_f_plus": 0.8280891180038452,
      "fidelity_minus": -0.0034022927284240723,
      "prob_f_minus": 0.9522761106491089
    }
  },
  {
    "user_id_remapped": 141032,
    "user_id_original": "2392741",
    "user_name": "User 2392741",
    "recommended_item_id_remapped": 221498,
    "recommended_item_id_original": "474439",
    "recommended_item_name": "easy sugar cookies",
    "score": 2.960449695587158,
    "original_prob": 0.9507550597190857,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 141386,
    "user_id_original": "2406751",
    "user_name": "User 2406751",
    "recommended_item_id_remapped": 193267,
    "recommended_item_id_original": "383692",
    "recommended_item_name": "red velvet cake from the bubble room",
    "score": 2.828512191772461,
    "original_prob": 0.9441972374916077,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 121380,
    "user_id_original": "1759099",
    "user_name": "User 1759099",
    "recommended_item_id_remapped": 81750,
    "recommended_item_id_original": "141395",
    "recommended_item_name": "fantastic double chocolate pecan biscotti",
    "score": 2.9376449584960938,
    "original_prob": 0.9496762156486511,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1759099 -> fantastic double chocolate pecan biscotti",
        "path_details": [
          {
            "id_remapped": 121380,
            "id_original": "1759099",
            "type": "USER",
            "name": "User 1759099"
          },
          {
            "id_remapped": 308320,
            "id_original": "141395",
            "type": "RECIPE",
            "name": "fantastic double chocolate pecan biscotti"
          }
        ],
        "contribution_score": 0.776867687702179
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.050113141536712646,
      "prob_f_plus": 0.8995630741119385,
      "fidelity_minus": -0.0030695199966430664,
      "prob_f_minus": 0.9527457356452942
    }
  },
  {
    "user_id_remapped": 58491,
    "user_id_original": "578543",
    "user_name": "User 578543",
    "recommended_item_id_remapped": 8217,
    "recommended_item_id_original": "17186",
    "recommended_item_name": "patti labelle s macaroni and cheese",
    "score": 2.9874954223632812,
    "original_prob": 0.9520060420036316,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 578543 -> tsr 1993 version of cinnabon cinnamon rolls by todd wilbur -> User 194282 -> patti labelle s macaroni and cheese",
        "path_details": [
          {
            "id_remapped": 58491,
            "id_original": "578543",
            "type": "USER",
            "name": "User 578543"
          },
          {
            "id_remapped": 270644,
            "id_original": "76864",
            "type": "RECIPE",
            "name": "tsr 1993 version of cinnabon cinnamon rolls by todd wilbur"
          },
          {
            "id_remapped": 23048,
            "id_original": "194282",
            "type": "USER",
            "name": "User 194282"
          },
          {
            "id_remapped": 234787,
            "id_original": "17186",
            "type": "RECIPE",
            "name": "patti labelle s macaroni and cheese"
          }
        ],
        "contribution_score": 0.0018807903996786188
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 578543 -> tsr 1993 version of cinnabon cinnamon rolls by todd wilbur -> User 136369 -> patti labelle s macaroni and cheese",
        "path_details": [
          {
            "id_remapped": 58491,
            "id_original": "578543",
            "type": "USER",
            "name": "User 578543"
          },
          {
            "id_remapped": 270644,
            "id_original": "76864",
            "type": "RECIPE",
            "name": "tsr 1993 version of cinnabon cinnamon rolls by todd wilbur"
          },
          {
            "id_remapped": 16022,
            "id_original": "136369",
            "type": "USER",
            "name": "User 136369"
          },
          {
            "id_remapped": 234787,
            "id_original": "17186",
            "type": "RECIPE",
            "name": "patti labelle s macaroni and cheese"
          }
        ],
        "contribution_score": 0.000103112225252524
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 578543 -> tsr 1993 version of cinnabon cinnamon rolls by todd wilbur -> User 140132 -> patti labelle s macaroni and cheese",
        "path_details": [
          {
            "id_remapped": 58491,
            "id_original": "578543",
            "type": "USER",
            "name": "User 578543"
          },
          {
            "id_remapped": 270644,
            "id_original": "76864",
            "type": "RECIPE",
            "name": "tsr 1993 version of cinnabon cinnamon rolls by todd wilbur"
          },
          {
            "id_remapped": 16523,
            "id_original": "140132",
            "type": "USER",
            "name": "User 140132"
          },
          {
            "id_remapped": 234787,
            "id_original": "17186",
            "type": "RECIPE",
            "name": "patti labelle s macaroni and cheese"
          }
        ],
        "contribution_score": 7.327980364438105e-08
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.003532230854034424,
      "prob_f_plus": 0.9484738111495972,
      "fidelity_minus": 0.001840353012084961,
      "prob_f_minus": 0.9501656889915466
    }
  },
  {
    "user_id_remapped": 172238,
    "user_id_original": "2000056072",
    "user_name": "User 2000056072",
    "recommended_item_id_remapped": 124714,
    "recommended_item_id_original": "224681",
    "recommended_item_name": "easy chili mac",
    "score": 2.8503332138061523,
    "original_prob": 0.945335865020752,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 33837,
    "user_id_original": "293349",
    "user_name": "User 293349",
    "recommended_item_id_remapped": 86480,
    "recommended_item_id_original": "150131",
    "recommended_item_name": "world famous paradise bakery sugar cookies",
    "score": 2.967193126678467,
    "original_prob": 0.9510698318481445,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 293349 -> gnocchi   tomato bake  with freezing instructions -> User 307214 -> world famous paradise bakery sugar cookies",
        "path_details": [
          {
            "id_remapped": 33837,
            "id_original": "293349",
            "type": "USER",
            "name": "User 293349"
          },
          {
            "id_remapped": 393873,
            "id_original": "318100",
            "type": "RECIPE",
            "name": "gnocchi   tomato bake  with freezing instructions"
          },
          {
            "id_remapped": 35285,
            "id_original": "307214",
            "type": "USER",
            "name": "User 307214"
          },
          {
            "id_remapped": 313050,
            "id_original": "150131",
            "type": "RECIPE",
            "name": "world famous paradise bakery sugar cookies"
          }
        ],
        "contribution_score": 1.235027275571845e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": -0.000618278980255127,
      "prob_f_plus": 0.9516881108283997,
      "fidelity_minus": 0.023460686206817627,
      "prob_f_minus": 0.9276091456413269
    }
  },
  {
    "user_id_remapped": 49187,
    "user_id_original": "458914",
    "user_name": "User 458914",
    "recommended_item_id_remapped": 68288,
    "recommended_item_id_original": "117026",
    "recommended_item_name": "tons of blueberry coffee cake",
    "score": 3.028545618057251,
    "original_prob": 0.9538471698760986,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 458914 -> kittencal s italian melt in your mouth meatballs -> User 935050 -> tons of blueberry coffee cake",
        "path_details": [
          {
            "id_remapped": 49187,
            "id_original": "458914",
            "type": "USER",
            "name": "User 458914"
          },
          {
            "id_remapped": 266283,
            "id_original": "69173",
            "type": "RECIPE",
            "name": "kittencal s italian melt in your mouth meatballs"
          },
          {
            "id_remapped": 81694,
            "id_original": "935050",
            "type": "USER",
            "name": "User 935050"
          },
          {
            "id_remapped": 294858,
            "id_original": "117026",
            "type": "RECIPE",
            "name": "tons of blueberry coffee cake"
          }
        ],
        "contribution_score": 0.0003271706417673119
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 458914 -> olive garden pasta e fagioli soup in a crock pot  copycat -> User 156352 -> tons of blueberry coffee cake",
        "path_details": [
          {
            "id_remapped": 49187,
            "id_original": "458914",
            "type": "USER",
            "name": "User 458914"
          },
          {
            "id_remapped": 243752,
            "id_original": "31717",
            "type": "RECIPE",
            "name": "olive garden pasta e fagioli soup in a crock pot  copycat"
          },
          {
            "id_remapped": 18387,
            "id_original": "156352",
            "type": "USER",
            "name": "User 156352"
          },
          {
            "id_remapped": 294858,
            "id_original": "117026",
            "type": "RECIPE",
            "name": "tons of blueberry coffee cake"
          }
        ],
        "contribution_score": 0.00017636336960119995
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 458914 -> kittencal s chocolate frosting icing -> User 1471487 -> tons of blueberry coffee cake",
        "path_details": [
          {
            "id_remapped": 49187,
            "id_original": "458914",
            "type": "USER",
            "name": "User 458914"
          },
          {
            "id_remapped": 277883,
            "id_original": "89207",
            "type": "RECIPE",
            "name": "kittencal s chocolate frosting icing"
          },
          {
            "id_remapped": 108352,
            "id_original": "1471487",
            "type": "USER",
            "name": "User 1471487"
          },
          {
            "id_remapped": 294858,
            "id_original": "117026",
            "type": "RECIPE",
            "name": "tons of blueberry coffee cake"
          }
        ],
        "contribution_score": 6.0039697961070266e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.000456392765045166,
      "prob_f_plus": 0.9533907771110535,
      "fidelity_minus": 0.004427075386047363,
      "prob_f_minus": 0.9494200944900513
    }
  },
  {
    "user_id_remapped": 154867,
    "user_id_original": "2945240",
    "user_name": "User 2945240",
    "recommended_item_id_remapped": 17708,
    "recommended_item_id_original": "32579",
    "recommended_item_name": "balsamic pan seared pork chops",
    "score": 2.9562785625457764,
    "original_prob": 0.9505594372749329,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 128635,
    "user_id_original": "1949760",
    "user_name": "User 1949760",
    "recommended_item_id_remapped": 84434,
    "recommended_item_id_original": "146339",
    "recommended_item_name": "oven roasted pork tenderloin with brown sugar garlic glaze",
    "score": 2.970480442047119,
    "original_prob": 0.9512225389480591,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 1949760 -> oven roasted pork tenderloin with brown sugar garlic glaze",
        "path_details": [
          {
            "id_remapped": 128635,
            "id_original": "1949760",
            "type": "USER",
            "name": "User 1949760"
          },
          {
            "id_remapped": 311004,
            "id_original": "146339",
            "type": "RECIPE",
            "name": "oven roasted pork tenderloin with brown sugar garlic glaze"
          }
        ],
        "contribution_score": 0.7958181500434875
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.025529086589813232,
      "prob_f_plus": 0.9256934523582458,
      "fidelity_minus": -0.0017343759536743164,
      "prob_f_minus": 0.9529569149017334
    }
  },
  {
    "user_id_remapped": 11120,
    "user_id_original": "94769",
    "user_name": "User 94769",
    "recommended_item_id_remapped": 23515,
    "recommended_item_id_original": "42169",
    "recommended_item_name": "jumbo shrimp parmesan",
    "score": 3.033750534057617,
    "original_prob": 0.9540757536888123,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 94769 -> carne guisada  mexican beef stew -> User 176153 -> jumbo shrimp parmesan",
        "path_details": [
          {
            "id_remapped": 11120,
            "id_original": "94769",
            "type": "USER",
            "name": "User 94769"
          },
          {
            "id_remapped": 274726,
            "id_original": "84098",
            "type": "RECIPE",
            "name": "carne guisada  mexican beef stew"
          },
          {
            "id_remapped": 20838,
            "id_original": "176153",
            "type": "USER",
            "name": "User 176153"
          },
          {
            "id_remapped": 250085,
            "id_original": "42169",
            "type": "RECIPE",
            "name": "jumbo shrimp parmesan"
          }
        ],
        "contribution_score": 0.0015735834497290074
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 94769 -> to die for crock pot roast -> User 299720 -> jumbo shrimp parmesan",
        "path_details": [
          {
            "id_remapped": 11120,
            "id_original": "94769",
            "type": "USER",
            "name": "User 94769"
          },
          {
            "id_remapped": 240994,
            "id_original": "27208",
            "type": "RECIPE",
            "name": "to die for crock pot roast"
          },
          {
            "id_remapped": 34525,
            "id_original": "299720",
            "type": "USER",
            "name": "User 299720"
          },
          {
            "id_remapped": 250085,
            "id_original": "42169",
            "type": "RECIPE",
            "name": "jumbo shrimp parmesan"
          }
        ],
        "contribution_score": 0.0010215856794040254
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 94769 -> creamy cajun chicken pasta -> User 65988 -> jumbo shrimp parmesan",
        "path_details": [
          {
            "id_remapped": 11120,
            "id_original": "94769",
            "type": "USER",
            "name": "User 94769"
          },
          {
            "id_remapped": 248209,
            "id_original": "39087",
            "type": "RECIPE",
            "name": "creamy cajun chicken pasta"
          },
          {
            "id_remapped": 8019,
            "id_original": "65988",
            "type": "USER",
            "name": "User 65988"
          },
          {
            "id_remapped": 250085,
            "id_original": "42169",
            "type": "RECIPE",
            "name": "jumbo shrimp parmesan"
          }
        ],
        "contribution_score": 0.0008751800533489236
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0005859136581420898,
      "prob_f_plus": 0.9534898400306702,
      "fidelity_minus": 0.0006226897239685059,
      "prob_f_minus": 0.9534530639648438
    }
  },
  {
    "user_id_remapped": 43020,
    "user_id_original": "380637",
    "user_name": "User 380637",
    "recommended_item_id_remapped": 87624,
    "recommended_item_id_original": "152283",
    "recommended_item_name": "gluten free buttermilk biscuits",
    "score": 2.974154472351074,
    "original_prob": 0.9513927698135376,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 380637 -> yoghurt homemade -> free-of-something -> gluten free buttermilk biscuits",
        "path_details": [
          {
            "id_remapped": 43020,
            "id_original": "380637",
            "type": "USER",
            "name": "User 380637"
          },
          {
            "id_remapped": 320601,
            "id_original": "164744",
            "type": "RECIPE",
            "name": "yoghurt homemade"
          },
          {
            "id_remapped": 473153,
            "id_original": "14946",
            "type": "TAG",
            "name": "free-of-something"
          },
          {
            "id_remapped": 314194,
            "id_original": "152283",
            "type": "RECIPE",
            "name": "gluten free buttermilk biscuits"
          }
        ],
        "contribution_score": 9.875153461657324e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 380637 -> yoghurt homemade -> gluten-free -> gluten free buttermilk biscuits",
        "path_details": [
          {
            "id_remapped": 43020,
            "id_original": "380637",
            "type": "USER",
            "name": "User 380637"
          },
          {
            "id_remapped": 320601,
            "id_original": "164744",
            "type": "RECIPE",
            "name": "yoghurt homemade"
          },
          {
            "id_remapped": 473169,
            "id_original": "14962",
            "type": "TAG",
            "name": "gluten-free"
          },
          {
            "id_remapped": 314194,
            "id_original": "152283",
            "type": "RECIPE",
            "name": "gluten free buttermilk biscuits"
          }
        ],
        "contribution_score": 8.203282844628327e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 380637 -> uncle bill s microwave potato chips -> User 215898 -> gluten free buttermilk biscuits",
        "path_details": [
          {
            "id_remapped": 43020,
            "id_original": "380637",
            "type": "USER",
            "name": "User 380637"
          },
          {
            "id_remapped": 252952,
            "id_original": "47195",
            "type": "RECIPE",
            "name": "uncle bill s microwave potato chips"
          },
          {
            "id_remapped": 25507,
            "id_original": "215898",
            "type": "USER",
            "name": "User 215898"
          },
          {
            "id_remapped": 314194,
            "id_original": "152283",
            "type": "RECIPE",
            "name": "gluten free buttermilk biscuits"
          }
        ],
        "contribution_score": 2.4141084096540736e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.001586318016052246,
      "prob_f_plus": 0.9498064517974854,
      "fidelity_minus": 0.06720346212387085,
      "prob_f_minus": 0.8841893076896667
    }
  },
  {
    "user_id_remapped": 173771,
    "user_id_original": "2000141599",
    "user_name": "User 2000141599",
    "recommended_item_id_remapped": 229061,
    "recommended_item_id_original": "507176",
    "recommended_item_name": "banana bread brownies",
    "score": 2.9737467765808105,
    "original_prob": 0.9513738751411438,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE",
        "path_description": "User 2000141599 -> banana bread brownies",
        "path_details": [
          {
            "id_remapped": 173771,
            "id_original": "2000141599",
            "type": "USER",
            "name": "User 2000141599"
          },
          {
            "id_remapped": 455631,
            "id_original": "507176",
            "type": "RECIPE",
            "name": "banana bread brownies"
          }
        ],
        "contribution_score": 0.7975589632987976
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.09168767929077148,
      "prob_f_plus": 0.8596861958503723,
      "fidelity_minus": -0.0012533068656921387,
      "prob_f_minus": 0.9526271820068359
    }
  },
  {
    "user_id_remapped": 22765,
    "user_id_original": "191915",
    "user_name": "User 191915",
    "recommended_item_id_remapped": 23592,
    "recommended_item_id_original": "42292",
    "recommended_item_name": "overnight french toast casserole",
    "score": 3.0356125831604004,
    "original_prob": 0.9541572332382202,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 191915 -> fruit salad with pudding -> User 192856 -> overnight french toast casserole",
        "path_details": [
          {
            "id_remapped": 22765,
            "id_original": "191915",
            "type": "USER",
            "name": "User 191915"
          },
          {
            "id_remapped": 264754,
            "id_original": "66575",
            "type": "RECIPE",
            "name": "fruit salad with pudding"
          },
          {
            "id_remapped": 22890,
            "id_original": "192856",
            "type": "USER",
            "name": "User 192856"
          },
          {
            "id_remapped": 250162,
            "id_original": "42292",
            "type": "RECIPE",
            "name": "overnight french toast casserole"
          }
        ],
        "contribution_score": 0.00014788125015548686
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 191915 -> vanilla  buttercream frosting  from sprinkles cupcakes -> User 142414 -> overnight french toast casserole",
        "path_details": [
          {
            "id_remapped": 22765,
            "id_original": "191915",
            "type": "USER",
            "name": "User 191915"
          },
          {
            "id_remapped": 350005,
            "id_original": "222188",
            "type": "RECIPE",
            "name": "vanilla  buttercream frosting  from sprinkles cupcakes"
          },
          {
            "id_remapped": 16829,
            "id_original": "142414",
            "type": "USER",
            "name": "User 142414"
          },
          {
            "id_remapped": 250162,
            "id_original": "42292",
            "type": "RECIPE",
            "name": "overnight french toast casserole"
          }
        ],
        "contribution_score": 2.057670933860791e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 191915 -> edna s apple crumble  aka  apple crisp -> User 229052 -> overnight french toast casserole",
        "path_details": [
          {
            "id_remapped": 22765,
            "id_original": "191915",
            "type": "USER",
            "name": "User 191915"
          },
          {
            "id_remapped": 274068,
            "id_original": "82925",
            "type": "RECIPE",
            "name": "edna s apple crumble  aka  apple crisp"
          },
          {
            "id_remapped": 27031,
            "id_original": "229052",
            "type": "USER",
            "name": "User 229052"
          },
          {
            "id_remapped": 250162,
            "id_original": "42292",
            "type": "RECIPE",
            "name": "overnight french toast casserole"
          }
        ],
        "contribution_score": 1.606984337513804e-05
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0019312500953674316,
      "prob_f_plus": 0.9522259831428528,
      "fidelity_minus": -0.0006189346313476562,
      "prob_f_minus": 0.9547761678695679
    }
  },
  {
    "user_id_remapped": 66617,
    "user_id_original": "698347",
    "user_name": "User 698347",
    "recommended_item_id_remapped": 32533,
    "recommended_item_id_original": "57424",
    "recommended_item_name": "comfort soup  spinach   meatballs",
    "score": 2.9654998779296875,
    "original_prob": 0.9509910345077515,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 698347 -> chicken marinade -> stove-top -> comfort soup  spinach   meatballs",
        "path_details": [
          {
            "id_remapped": 66617,
            "id_original": "698347",
            "type": "USER",
            "name": "User 698347"
          },
          {
            "id_remapped": 350035,
            "id_original": "222238",
            "type": "RECIPE",
            "name": "chicken marinade"
          },
          {
            "id_remapped": 473090,
            "id_original": "14883",
            "type": "TAG",
            "name": "stove-top"
          },
          {
            "id_remapped": 259103,
            "id_original": "57424",
            "type": "RECIPE",
            "name": "comfort soup  spinach   meatballs"
          }
        ],
        "contribution_score": 2.8161754506090957e-05
      },
      {
        "path_structure": "USER -> RECIPE -> TAG -> RECIPE",
        "path_description": "User 698347 -> baked chicken breasts -> one-dish-meal -> comfort soup  spinach   meatballs",
        "path_details": [
          {
            "id_remapped": 66617,
            "id_original": "698347",
            "type": "USER",
            "name": "User 698347"
          },
          {
            "id_remapped": 251870,
            "id_original": "45246",
            "type": "RECIPE",
            "name": "baked chicken breasts"
          },
          {
            "id_remapped": 473147,
            "id_original": "14940",
            "type": "TAG",
            "name": "one-dish-meal"
          },
          {
            "id_remapped": 259103,
            "id_original": "57424",
            "type": "RECIPE",
            "name": "comfort soup  spinach   meatballs"
          }
        ],
        "contribution_score": 1.000510219715712e-05
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 698347 -> artichoke and chicken bake -> User 63128 -> comfort soup  spinach   meatballs",
        "path_details": [
          {
            "id_remapped": 66617,
            "id_original": "698347",
            "type": "USER",
            "name": "User 698347"
          },
          {
            "id_remapped": 266233,
            "id_original": "69089",
            "type": "RECIPE",
            "name": "artichoke and chicken bake"
          },
          {
            "id_remapped": 7660,
            "id_original": "63128",
            "type": "USER",
            "name": "User 63128"
          },
          {
            "id_remapped": 259103,
            "id_original": "57424",
            "type": "RECIPE",
            "name": "comfort soup  spinach   meatballs"
          }
        ],
        "contribution_score": 1.9599048106417434e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.008303463459014893,
      "prob_f_plus": 0.9426875710487366,
      "fidelity_minus": 0.11603426933288574,
      "prob_f_minus": 0.8349567651748657
    }
  },
  {
    "user_id_remapped": 219316,
    "user_id_original": "2002102039",
    "user_name": "User 2002102039",
    "recommended_item_id_remapped": 87,
    "recommended_item_id_original": "198",
    "recommended_item_name": "chinese hot and sour pork soup",
    "score": 2.964003086090088,
    "original_prob": 0.9509211778640747,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  }
]
```

## File: output\fidelity\depth_3\metrics.json

```text
{
  "num_evaluated": 60,
  "total_requested": 100,
  "avg_fidelity_plus": 0.071107550462087,
  "avg_fidelity_minus": 0.08239506582419077
}
```

## File: output\fidelity\test\explanations.json

```text
[
  {
    "user_id_remapped": 14139,
    "user_id_original": "119643",
    "user_name": "User 119643",
    "recommended_item_id_remapped": 110055,
    "recommended_item_id_original": "195519",
    "recommended_item_name": "shredded pork barbacoa",
    "score": 0.9991135597229004,
    "original_prob": 0.7308842539787292,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 111880,
    "user_id_original": "1549375",
    "user_name": "User 1549375",
    "recommended_item_id_remapped": 38631,
    "recommended_item_id_original": "67256",
    "recommended_item_name": "best ever banana cake with cream cheese frosting",
    "score": 1.0845757722854614,
    "original_prob": 0.7473589777946472,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1549375 -> soft snickerdoodle cookies -> User 961101 -> best ever banana cake with cream cheese frosting",
        "path_details": [
          {
            "id_remapped": 111880,
            "id_original": "1549375",
            "type": "USER",
            "name": "User 1549375"
          },
          {
            "id_remapped": 282954,
            "id_original": "97496",
            "type": "RECIPE",
            "name": "soft snickerdoodle cookies"
          },
          {
            "id_remapped": 83105,
            "id_original": "961101",
            "type": "USER",
            "name": "User 961101"
          },
          {
            "id_remapped": 265201,
            "id_original": "67256",
            "type": "RECIPE",
            "name": "best ever banana cake with cream cheese frosting"
          }
        ],
        "contribution_score": 0.018048507730850347
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1549375 -> soft snickerdoodle cookies -> User 573418 -> best ever banana cake with cream cheese frosting",
        "path_details": [
          {
            "id_remapped": 111880,
            "id_original": "1549375",
            "type": "USER",
            "name": "User 1549375"
          },
          {
            "id_remapped": 282954,
            "id_original": "97496",
            "type": "RECIPE",
            "name": "soft snickerdoodle cookies"
          },
          {
            "id_remapped": 58084,
            "id_original": "573418",
            "type": "USER",
            "name": "User 573418"
          },
          {
            "id_remapped": 265201,
            "id_original": "67256",
            "type": "RECIPE",
            "name": "best ever banana cake with cream cheese frosting"
          }
        ],
        "contribution_score": 0.008496219323220224
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1549375 -> soft snickerdoodle cookies -> User 1220448 -> best ever banana cake with cream cheese frosting",
        "path_details": [
          {
            "id_remapped": 111880,
            "id_original": "1549375",
            "type": "USER",
            "name": "User 1549375"
          },
          {
            "id_remapped": 282954,
            "id_original": "97496",
            "type": "RECIPE",
            "name": "soft snickerdoodle cookies"
          },
          {
            "id_remapped": 95966,
            "id_original": "1220448",
            "type": "USER",
            "name": "User 1220448"
          },
          {
            "id_remapped": 265201,
            "id_original": "67256",
            "type": "RECIPE",
            "name": "best ever banana cake with cream cheese frosting"
          }
        ],
        "contribution_score": 0.005844184273397104
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.06760549545288086,
      "prob_f_plus": 0.6797534823417664,
      "fidelity_minus": 4.6193599700927734e-05,
      "prob_f_minus": 0.7473127841949463
    }
  },
  {
    "user_id_remapped": 163352,
    "user_id_original": "1802508407",
    "user_name": "User 1802508407",
    "recommended_item_id_remapped": 51312,
    "recommended_item_id_original": "89204",
    "recommended_item_name": "crock pot chicken with black beans   cream cheese",
    "score": 1.105439305305481,
    "original_prob": 0.7512778639793396,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1802508407 -> pronto pups aka corn dogs -> User 1930368 -> crock pot chicken with black beans   cream cheese",
        "path_details": [
          {
            "id_remapped": 163352,
            "id_original": "1802508407",
            "type": "USER",
            "name": "User 1802508407"
          },
          {
            "id_remapped": 247158,
            "id_original": "37374",
            "type": "RECIPE",
            "name": "pronto pups aka corn dogs"
          },
          {
            "id_remapped": 127902,
            "id_original": "1930368",
            "type": "USER",
            "name": "User 1930368"
          },
          {
            "id_remapped": 277882,
            "id_original": "89204",
            "type": "RECIPE",
            "name": "crock pot chicken with black beans   cream cheese"
          }
        ],
        "contribution_score": 0.08825717842197925
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1802508407 -> pronto pups aka corn dogs -> User 762440 -> crock pot chicken with black beans   cream cheese",
        "path_details": [
          {
            "id_remapped": 163352,
            "id_original": "1802508407",
            "type": "USER",
            "name": "User 1802508407"
          },
          {
            "id_remapped": 247158,
            "id_original": "37374",
            "type": "RECIPE",
            "name": "pronto pups aka corn dogs"
          },
          {
            "id_remapped": 71058,
            "id_original": "762440",
            "type": "USER",
            "name": "User 762440"
          },
          {
            "id_remapped": 277882,
            "id_original": "89204",
            "type": "RECIPE",
            "name": "crock pot chicken with black beans   cream cheese"
          }
        ],
        "contribution_score": 0.009204245423576745
      },
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1802508407 -> pronto pups aka corn dogs -> User 295018 -> crock pot chicken with black beans   cream cheese",
        "path_details": [
          {
            "id_remapped": 163352,
            "id_original": "1802508407",
            "type": "USER",
            "name": "User 1802508407"
          },
          {
            "id_remapped": 247158,
            "id_original": "37374",
            "type": "RECIPE",
            "name": "pronto pups aka corn dogs"
          },
          {
            "id_remapped": 34012,
            "id_original": "295018",
            "type": "USER",
            "name": "User 295018"
          },
          {
            "id_remapped": 277882,
            "id_original": "89204",
            "type": "RECIPE",
            "name": "crock pot chicken with black beans   cream cheese"
          }
        ],
        "contribution_score": 0.0006427640953192029
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.0350528359413147,
      "prob_f_plus": 0.7162250280380249,
      "fidelity_minus": -0.00012111663818359375,
      "prob_f_minus": 0.7513989806175232
    }
  },
  {
    "user_id_remapped": 97023,
    "user_id_original": "1241754",
    "user_name": "User 1241754",
    "recommended_item_id_remapped": 43638,
    "recommended_item_id_original": "76107",
    "recommended_item_name": "longhunter s bear roast",
    "score": 1.0150848627090454,
    "original_prob": 0.734014093875885,
    "explanations": [],
    "fidelity": {},
    "explanations_note": "無法找到解釋路徑。"
  },
  {
    "user_id_remapped": 160622,
    "user_id_original": "1800345045",
    "user_name": "User 1800345045",
    "recommended_item_id_remapped": 36456,
    "recommended_item_id_original": "63689",
    "recommended_item_name": "my family s favorite sloppy joes  pizza joes",
    "score": 1.1014845371246338,
    "original_prob": 0.750538170337677,
    "explanations": [
      {
        "path_structure": "USER -> RECIPE -> USER -> RECIPE",
        "path_description": "User 1800345045 -> chili s chicken crispers -> User 182624 -> my family s favorite sloppy joes  pizza joes",
        "path_details": [
          {
            "id_remapped": 160622,
            "id_original": "1800345045",
            "type": "USER",
            "name": "User 1800345045"
          },
          {
            "id_remapped": 377635,
            "id_original": "281782",
            "type": "RECIPE",
            "name": "chili s chicken crispers"
          },
          {
            "id_remapped": 21619,
            "id_original": "182624",
            "type": "USER",
            "name": "User 182624"
          },
          {
            "id_remapped": 263026,
            "id_original": "63689",
            "type": "RECIPE",
            "name": "my family s favorite sloppy joes  pizza joes"
          }
        ],
        "contribution_score": 8.007810048795055e-06
      }
    ],
    "fidelity": {
      "fidelity_plus": 0.01483684778213501,
      "prob_f_plus": 0.735701322555542,
      "fidelity_minus": 0.0002561211585998535,
      "prob_f_minus": 0.7502820491790771
    }
  }
]
```

## File: output\fidelity\test\metrics.json

```text
{
  "num_evaluated": 3,
  "total_requested": 5,
  "avg_fidelity_plus": 0.03916505972544352,
  "avg_fidelity_minus": 6.0399373372395836e-05
}
```

## File: output\fidelity\test\sampled_users.json

```text
[
  14139,
  111880,
  163352,
  97023,
  160622
]
```

