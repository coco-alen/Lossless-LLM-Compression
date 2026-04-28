{
  "model": "NousResearch/Meta-Llama-3-8B",
  "shape": [
    1024,
    4096
  ],
  "chunk_size": 1024,
  "meta": {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "model_type": "llama",
    "prompt_tokens": 1024,
    "block_rows_total": 32768,
    "block_width": 2048
  },
  "bf16_topk": {
    "top8": {
      "coverage": 0.8234846591949463,
      "escapes": 740359,
      "escape_rate": 0.1765153408050537,
      "compressed_bytes": 10949681,
      "ratio": 0.7661052408741406,
      "encode_gbs": {
        "mean": 64.84887071024536,
        "std": 3.320002805005348,
        "stderr": 2.34759649697764
      },
      "decode_gbs": {
        "mean": 157.28968232304663,
        "std": 2.840501255324093,
        "stderr": 2.008537699608567
      }
    },
    "top16": {
      "coverage": 0.9963395595550537,
      "escapes": 15353,
      "escape_rate": 0.003660440444946289,
      "compressed_bytes": 6398927,
      "ratio": 1.3109397872487059,
      "encode_gbs": {
        "mean": 36.165837068740096,
        "std": 9.715158399728402,
        "stderr": 6.8696543847494
      },
      "decode_gbs": {
        "mean": 121.58248246345336,
        "std": 6.521201274450942,
        "stderr": 4.611185642646617
      }
    }
  },
  "escape_metadata": {
    "top16_positions": {
      "coverage": 0.9963395595550537,
      "escapes": 15353,
      "escape_rate": 0.003660440444946289,
      "compressed_bytes": 6398927,
      "ratio": 1.3109397872487059,
      "encode_gbs": {
        "mean": 36.165837068740096,
        "std": 9.715158399728402,
        "stderr": 6.8696543847494
      },
      "decode_gbs": {
        "mean": 121.58248246345336,
        "std": 6.521201274450942,
        "stderr": 4.611185642646617
      }
    },
    "top15_sentinel": {
      "coverage": 0.9942362308502197,
      "escapes": 24175,
      "escape_rate": 0.0057637691497802734,
      "compressed_bytes": 6315631,
      "ratio": 1.3282295941609001,
      "encode_gbs": {
        "mean": 45.73488525236786,
        "std": 18.260914844475813,
        "stderr": 12.912416717198935
      },
      "decode_gbs": {
        "mean": 176.8948618691211,
        "std": 6.402486961710482,
        "stderr": 4.527241947083937
      }
    }
  },
  "precalibration": {
    "precalibrated": {
      "coverage": 0.9963395595550537,
      "escapes": 15353,
      "escape_rate": 0.003660440444946289,
      "compressed_bytes": 6398927,
      "ratio": 1.3109397872487059,
      "encode_gbs": {
        "mean": 49.63821631561376,
        "std": 12.467030180536355,
        "stderr": 7.1978432307278535
      },
      "decode_gbs": {
        "mean": 183.8852255456314,
        "std": 6.27803031200188,
        "stderr": 3.624622490614916
      }
    },
    "dynamic": {
      "coverage": 0.9963395595550537,
      "escapes": 15353,
      "escape_rate": 0.003660440444946289,
      "compressed_bytes": 6398927,
      "ratio": 1.3109397872487059,
      "encode_gbs": {
        "mean": 4.4480780417030745,
        "std": 1.1852503255535631,
        "stderr": 0.6843045945154412
      },
      "decode_gbs": {
        "mean": 140.89390722240705,
        "std": 33.166609789682326,
        "stderr": 19.14875109018037
      }
    }
  },
  "granularity": {
    "per_tensor": {
      "coverage": 0.9963395595550537,
      "escape_rate": 0.003660440444946289,
      "projected_ratio": 1.3109365093578735,
      "codebook_bytes": 16,
      "actual_ratio": 1.3109397872487059,
      "encode_gbs": 36.165837068740096,
      "decode_gbs": 121.58248246345336
    },
    "per_token": {
      "coverage": 0.9965686798095703,
      "escape_rate": 0.0034313201904296875,
      "projected_ratio": 1.3089643515634655,
      "codebook_bytes": 16384,
      "actual_ratio": null,
      "encode_gbs": null,
      "decode_gbs": null
    },
    "per_channel": {
      "coverage": 1.0,
      "escape_rate": 0.0,
      "projected_ratio": 1.3195876288659794,
      "codebook_bytes": 65536,
      "actual_ratio": null,
      "encode_gbs": null,
      "decode_gbs": null
    },
    "note": "Throughput is measured for the implemented per-tensor codec. Per-token and per-channel rows report full-shape coverage/ratio projection; they require separate per-group codebooks and are not implemented in the current high-throughput kernel.",
    "chunk_size": 1024
  },
  "calibration": null,
  "breakdown": {
    "model": "Qwen/Qwen3-32B",
    "transport_mode": "RoCE 4x200G",
    "rows": [
      {
        "seq_len": 2048,
        "raw_bytes": 16777216,
        "compressed_bytes": 12765542,
        "ratio": 1.3142580236702837,
        "native_transfer_ms": 27.289168218765447,
        "splitzip_encode_ms": 4.05221481807926,
        "splitzip_transfer_ms": 20.707676418871923,
        "splitzip_decode_ms": 1.4198768718489427,
        "splitzip_total_ms": 26.17976810880013,
        "encode_pct": 15.478421356670227,
        "transfer_pct": 79.09801314057934,
        "decode_pct": 5.4235655027504315
      },
      {
        "seq_len": 16384,
        "raw_bytes": 134217728,
        "compressed_bytes": 101374888,
        "ratio": 1.3239741187186318,
        "native_transfer_ms": 208.8935308803621,
        "splitzip_encode_ms": 12.632876679054363,
        "splitzip_transfer_ms": 158.25792453803402,
        "splitzip_decode_ms": 4.095187388921492,
        "splitzip_total_ms": 174.98598860600987,
        "encode_pct": 7.219364692962903,
        "transfer_pct": 90.44034085172387,
        "decode_pct": 2.340294455313231
      },
      {
        "seq_len": 65536,
        "raw_bytes": 536870912,
        "compressed_bytes": 405674860,
        "ratio": 1.3234019776330237,
        "native_transfer_ms": 863.3658150929224,
        "splitzip_encode_ms": 39.69903188611017,
        "splitzip_transfer_ms": 646.1309785780703,
        "splitzip_decode_ms": 9.78999925726682,
        "splitzip_total_ms": 695.6200097214473,
        "encode_pct": 5.706999702611655,
        "transfer_pct": 92.885622832616,
        "decode_pct": 1.4073774647723414
      }
    ]
  },
  "fp8_results": [
    {
      "format": "e4m3",
      "scheme": "top8",
      "coverage": 0.925175666809082,
      "escapes": 313836,
      "escape_rate": 0.07482433319091797,
      "raw_fp8_bytes": 4194304,
      "compressed_bytes": 4458702,
      "ratio_vs_fp8": 0.9407006792559808,
      "ratio_vs_bf16": 1.8814013585119616,
      "encode_gbs": {
        "mean": 22.822947475799502,
        "std": 5.72951509904964,
        "stderr": 4.051378979448714
      },
      "decode_gbs": {
        "mean": 78.32994482419285,
        "std": 0.7487240080497619,
        "stderr": 0.5294278233291578
      }
    },
    {
      "format": "e5m2",
      "scheme": "top8",
      "coverage": 0.823528528213501,
      "escapes": 740175,
      "escape_rate": 0.17647147178649902,
      "raw_fp8_bytes": 4194304,
      "compressed_bytes": 5092784,
      "ratio_vs_fp8": 0.8235778309074172,
      "ratio_vs_bf16": 1.6471556618148344,
      "encode_gbs": {
        "mean": 28.53759255574679,
        "std": 0.7235928039805677,
        "stderr": 0.5116573785124476
      },
      "decode_gbs": {
        "mean": 73.14227347455042,
        "std": 0.8701440603727892,
        "stderr": 0.6152847656987959
      }
    },
    {
      "format": "e5m2",
      "scheme": "top16",
      "coverage": 0.996016263961792,
      "escapes": 16709,
      "escape_rate": 0.003983736038208008,
      "raw_fp8_bytes": 4194304,
      "compressed_bytes": 3717974,
      "ratio_vs_fp8": 1.1281154736423655,
      "ratio_vs_bf16": 2.256230947284731,
      "encode_gbs": {
        "mean": 36.84944217881724,
        "std": 1.4468312603967932,
        "stderr": 1.0230641954592519
      },
      "decode_gbs": {
        "mean": 94.97118481109916,
        "std": 27.540752039334713,
        "stderr": 19.474252525990813
      }
    }
  ],
  "fp8_transfer": {
    "model": "Qwen/Qwen3-32B",
    "transport_mode": "RoCE 4x200G",
    "rows": {
      "e4m3_top8_exact": [
        {
          "seq_len": 1024,
          "native_ms": 3.54398204405632,
          "splitzip_ms": 4.165481241169049,
          "speedup": 0.8507977443349849,
          "ratio": 0.9407006792559808,
          "encode_gbs": 33.411000546205514,
          "decode_gbs": 102.76292926420088,
          "escape_rate": 0.07482433319091797
        },
        {
          "seq_len": 2048,
          "native_ms": 6.008373708153668,
          "splitzip_ms": 7.6669164474610865,
          "speedup": 0.7836753862295385,
          "ratio": 0.8904194709627319,
          "encode_gbs": 57.59314349324038,
          "decode_gbs": 196.5852049427431,
          "escape_rate": 0.09883582592010498
        },
        {
          "seq_len": 4096,
          "native_ms": 11.046920433252238,
          "splitzip_ms": 12.110207148147984,
          "speedup": 0.9121991307094731,
          "ratio": 0.9024005848612189,
          "encode_gbs": 113.24725856065345,
          "decode_gbs": 303.86390004372447,
          "escape_rate": 0.09287148714065552
        },
        {
          "seq_len": 8192,
          "native_ms": 24.12992060758756,
          "splitzip_ms": 24.06011945157742,
          "speedup": 1.0029011142754556,
          "ratio": 0.9279147760768653,
          "encode_gbs": 134.03345515456246,
          "decode_gbs": 345.7722881348079,
          "escape_rate": 0.08068343997001648
        },
        {
          "seq_len": 16384,
          "native_ms": 44.93720561250347,
          "splitzip_ms": 47.33488006408015,
          "speedup": 0.9493465611758009,
          "ratio": 0.943115962574625,
          "encode_gbs": 191.3817882516624,
          "decode_gbs": 430.3966672135487,
          "escape_rate": 0.07373537123203278
        },
        {
          "seq_len": 32768,
          "native_ms": 87.35002431987321,
          "splitzip_ms": 93.97835633832553,
          "speedup": 0.929469589842686,
          "ratio": 0.9611931153660592,
          "encode_gbs": 204.93611686445698,
          "decode_gbs": 459.4418758346463,
          "escape_rate": 0.0657588392496109
        },
        {
          "seq_len": 65536,
          "native_ms": 179.0906142483745,
          "splitzip_ms": 193.09134038861023,
          "speedup": 0.9274916932470495,
          "ratio": 0.9611931153660592,
          "encode_gbs": 225.74074076884628,
          "decode_gbs": 492.73807651419224,
          "escape_rate": 0.0657588392496109
        }
      ],
      "e5m2_top8_exact": [
        {
          "seq_len": 1024,
          "native_ms": 3.54398204405632,
          "splitzip_ms": 4.382310245300076,
          "speedup": 0.8087017681728849,
          "ratio": 0.8235778309074172,
          "encode_gbs": 41.51610882211755,
          "decode_gbs": 74.14275855459822,
          "escape_rate": 0.17647147178649902
        },
        {
          "seq_len": 2048,
          "native_ms": 6.008373708153668,
          "splitzip_ms": 6.6510102665332145,
          "speedup": 0.9033776024052785,
          "ratio": 0.8779488381731447,
          "encode_gbs": 53.012935954223714,
          "decode_gbs": 159.00883679013657,
          "escape_rate": 0.1478254795074463
        },
        {
          "seq_len": 4096,
          "native_ms": 11.046920433252238,
          "splitzip_ms": 11.839715630973204,
          "speedup": 0.9330393379003985,
          "ratio": 0.9678389922466578,
          "encode_gbs": 95.10612753991859,
          "decode_gbs": 183.82090973139702,
          "escape_rate": 0.10752499103546143
        },
        {
          "seq_len": 8192,
          "native_ms": 24.12992060758756,
          "splitzip_ms": 24.219222845647625,
          "speedup": 0.9963127537729348,
          "ratio": 1.0165942621969897,
          "encode_gbs": 159.8658136695387,
          "decode_gbs": 251.6798544089451,
          "escape_rate": 0.08864763379096985
        },
        {
          "seq_len": 16384,
          "native_ms": 44.93720561250347,
          "splitzip_ms": 47.221936769993235,
          "speedup": 0.9516171653734122,
          "ratio": 1.04518255354428,
          "encode_gbs": 182.88105091390972,
          "decode_gbs": 288.0348883115771,
          "escape_rate": 0.07839775085449219
        },
        {
          "seq_len": 32768,
          "native_ms": 87.35002431987321,
          "splitzip_ms": 87.13936748621367,
          "speedup": 1.0024174703091902,
          "ratio": 1.0687457964999503,
          "encode_gbs": 201.0909453064074,
          "decode_gbs": 333.7623320843418,
          "escape_rate": 0.07036176323890686
        },
        {
          "seq_len": 65536,
          "native_ms": 179.0906142483745,
          "splitzip_ms": 177.77392762840248,
          "speedup": 1.0074065226410718,
          "ratio": 1.0687458007550423,
          "encode_gbs": 232.16309416728393,
          "decode_gbs": 339.3929608798967,
          "escape_rate": 0.07036176323890686
        }
      ],
      "e5m2_top16_exact": [
        {
          "seq_len": 1024,
          "native_ms": 3.54398204405632,
          "splitzip_ms": 4.198654791627443,
          "speedup": 0.8440755956225292,
          "ratio": 1.1281154736423655,
          "encode_gbs": 33.02107969717374,
          "decode_gbs": 84.28423349234713,
          "escape_rate": 0.003983736038208008
        },
        {
          "seq_len": 2048,
          "native_ms": 6.008373708153668,
          "splitzip_ms": 6.139451727534031,
          "speedup": 0.9786498819116848,
          "ratio": 1.1315657384893991,
          "encode_gbs": 58.32343670311127,
          "decode_gbs": 160.81279651268372,
          "escape_rate": 0.00295412540435791
        },
        {
          "seq_len": 4096,
          "native_ms": 11.046920433252238,
          "splitzip_ms": 10.281837608105713,
          "speedup": 1.0744110979289703,
          "ratio": 1.134050521884994,
          "encode_gbs": 106.13403877349727,
          "decode_gbs": 252.28059342680842,
          "escape_rate": 0.002216517925262451
        },
        {
          "seq_len": 8192,
          "native_ms": 24.12992060758756,
          "splitzip_ms": 22.055828057432887,
          "speedup": 1.0940382988457191,
          "ratio": 1.1353413012825226,
          "encode_gbs": 178.99052827998116,
          "decode_gbs": 390.0679799908517,
          "escape_rate": 0.0018346011638641357
        },
        {
          "seq_len": 16384,
          "native_ms": 44.93720561250347,
          "splitzip_ms": 39.64209572225793,
          "speedup": 1.1335729051093653,
          "ratio": 1.1357813449012701,
          "encode_gbs": 198.00257820832977,
          "decode_gbs": 422.818979985491,
          "escape_rate": 0.0017046034336090088
        },
        {
          "seq_len": 32768,
          "native_ms": 87.35002431987321,
          "splitzip_ms": 82.89371647105807,
          "speedup": 1.0537592961026307,
          "ratio": 1.1361779471356623,
          "encode_gbs": 224.43139960777964,
          "decode_gbs": 319.1188854577844,
          "escape_rate": 0.0015875250101089478
        },
        {
          "seq_len": 65536,
          "native_ms": 179.0906142483745,
          "splitzip_ms": 165.63395060943776,
          "speedup": 1.081243389953714,
          "ratio": 1.1361779519446409,
          "encode_gbs": 250.85101701267135,
          "decode_gbs": 337.4773438206369,
          "escape_rate": 0.0015875250101089478
        }
      ]
    }
  }
}
