{
  "model": "Qwen/Qwen3-32B",
  "shape": [
    65536,
    4096
  ],
  "chunk_size": 1024,
  "meta": {
    "num_hidden_layers": 64,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 5120,
    "model_type": "qwen3",
    "prompt_tokens": 1024,
    "block_rows_total": 65536,
    "block_width": 2048
  },
  "bf16_topk": {
    "top8": {
      "coverage": 0.9210850521922112,
      "escapes": 21183570,
      "escape_rate": 0.07891494780778885,
      "compressed_bytes": 517383742,
      "ratio": 1.0376648286717907,
      "encode_gbs": {
        "mean": 440.0802409797445,
        "std": 5.062534628243021,
        "stderr": 1.6009140158721735
      },
      "decode_gbs": {
        "mean": 710.5373107925204,
        "std": 18.871341589852797,
        "stderr": 5.967642192699793
      }
    },
    "top16": {
      "coverage": 0.9984164088964462,
      "escapes": 425092,
      "escape_rate": 0.001583591103553772,
      "compressed_bytes": 405628828,
      "ratio": 1.3235521613370143,
      "encode_gbs": {
        "mean": 435.28766430882604,
        "std": 8.905308297685531,
        "stderr": 2.8161057486683054
      },
      "decode_gbs": {
        "mean": 1763.7872352625466,
        "std": 80.08955239504061,
        "stderr": 25.326540235172182
      }
    }
  },
  "escape_metadata": {
    "top16_positions": {
      "coverage": 0.9984164088964462,
      "escapes": 425092,
      "escape_rate": 0.001583591103553772,
      "compressed_bytes": 405628828,
      "ratio": 1.3235521613370143,
      "encode_gbs": {
        "mean": 435.28766430882604,
        "std": 8.905308297685531,
        "stderr": 2.8161057486683054
      },
      "decode_gbs": {
        "mean": 1763.7872352625466,
        "std": 80.08955239504061,
        "stderr": 25.326540235172182
      }
    },
    "top15_sentinel": {
      "coverage": 0.9972681179642677,
      "escapes": 733334,
      "escape_rate": 0.0027318820357322693,
      "compressed_bytes": 403386518,
      "ratio": 1.330909408330796,
      "encode_gbs": {
        "mean": 396.0037556287008,
        "std": 10.7824817893137,
        "stderr": 3.4097201283519083
      },
      "decode_gbs": {
        "mean": 620.7542157264127,
        "std": 22.92155990415321,
        "stderr": 7.248433682111495
      }
    }
  },
  "precalibration": {
    "precalibrated": {
      "coverage": 0.9984164088964462,
      "escapes": 425092,
      "escape_rate": 0.001583591103553772,
      "compressed_bytes": 405628828,
      "ratio": 1.3235521613370143,
      "encode_gbs": {
        "mean": 430.10259462328804,
        "std": 3.440437158538759,
        "stderr": 1.5386102717617773
      },
      "decode_gbs": {
        "mean": 1745.1332827395722,
        "std": 61.1499324634223,
        "stderr": 27.347081161546686
      }
    },
    "dynamic": {
      "coverage": 0.9984164088964462,
      "escapes": 425092,
      "escape_rate": 0.001583591103553772,
      "compressed_bytes": 405628828,
      "ratio": 1.3235521613370143,
      "encode_gbs": {
        "mean": 80.65108280630385,
        "std": 0.9494847136005033,
        "stderr": 0.4246224726415289
      },
      "decode_gbs": {
        "mean": 1779.076878567955,
        "std": 44.536074273138105,
        "stderr": 19.917137905143267
      }
    }
  },
  "granularity": {
    "per_tensor": {
      "coverage": 0.9984164088964462,
      "escape_rate": 0.001583591103553772,
      "projected_ratio": 1.3235521091295963,
      "codebook_bytes": 16,
      "actual_ratio": 1.3235521613370143,
      "encode_gbs": 435.28766430882604,
      "decode_gbs": 1763.7872352625466
    },
    "per_token": {
      "coverage": 0.9990037679672241,
      "escape_rate": 0.000996232032775879,
      "projected_ratio": 1.3237319750652095,
      "codebook_bytes": 1048576,
      "actual_ratio": null,
      "encode_gbs": null,
      "decode_gbs": null
    },
    "per_channel": {
      "coverage": 0.9990280941128731,
      "escape_rate": 0.0009719058871269226,
      "projected_ratio": 1.3270981996963511,
      "codebook_bytes": 65536,
      "actual_ratio": null,
      "encode_gbs": null,
      "decode_gbs": null
    },
    "note": "Throughput is measured for the implemented per-tensor codec. Per-token and per-channel rows report full-shape coverage/ratio projection; they require separate per-group codebooks and are not implemented in the current high-throughput kernel.",
    "chunk_size": 1024
  },
  "calibration": {
    "model": "Qwen/Qwen3-32B",
    "dataset_a": "WikiText-2 train",
    "prompt_count_per_set": 4,
    "a_to_a": 0.9983105659484863,
    "rows": [
      {
        "dataset": "WikiText-2",
        "domain": "Language",
        "a_to_b": 0.9982508420944214,
        "b_to_b": 0.9982508420944214
      },
      {
        "dataset": "HumanEval",
        "domain": "Code",
        "a_to_b": 0.9981491565704346,
        "b_to_b": 0.9981491565704346
      },
      {
        "dataset": "GSM8K",
        "domain": "Math",
        "a_to_b": 0.9979107975959778,
        "b_to_b": 0.9979107975959778
      },
      {
        "dataset": "MMLU",
        "domain": "Knowledge",
        "a_to_b": 0.9978247880935669,
        "b_to_b": 0.9978247880935669
      },
      {
        "dataset": "PTB",
        "domain": "Language",
        "a_to_b": 0.9976290464401245,
        "b_to_b": 0.9976290464401245
      }
    ]
  },
  "breakdown": {
    "model": "Qwen/Qwen3-32B",
    "transport_mode": "RoCE 4x200G",
    "rows": [
      {
        "seq_len": 2048,
        "raw_bytes": 16777216,
        "compressed_bytes": 12866139,
        "ratio": 1.3039821814454204,
        "native_transfer_ms": 56.47012565035381,
        "splitzip_encode_ms": 8.119228283213525,
        "splitzip_transfer_ms": 42.06324998260061,
        "splitzip_decode_ms": 2.9197045799765315,
        "splitzip_total_ms": 53.10218284579067,
        "encode_pct": 15.289820207187818,
        "transfer_pct": 79.2119037832263,
        "decode_pct": 5.4982760095858705
      },
      {
        "seq_len": 16384,
        "raw_bytes": 134217728,
        "compressed_bytes": 101955321,
        "ratio": 1.3164367164318966,
        "native_transfer_ms": 441.41778586298284,
        "splitzip_encode_ms": 24.133620258890772,
        "splitzip_transfer_ms": 323.4726907446779,
        "splitzip_decode_ms": 6.222642792402184,
        "splitzip_total_ms": 353.82895379597085,
        "encode_pct": 6.8207024891487515,
        "transfer_pct": 91.42063906143832,
        "decode_pct": 1.7586584494129218
      },
      {
        "seq_len": 65536,
        "raw_bytes": 536870912,
        "compressed_bytes": 405628828,
        "ratio": 1.3235521613370143,
        "native_transfer_ms": 1749.276286134722,
        "splitzip_encode_ms": 79.95811570617788,
        "splitzip_transfer_ms": 1297.602971690278,
        "splitzip_decode_ms": 19.393315465755297,
        "splitzip_total_ms": 1396.9544028622115,
        "encode_pct": 5.723745566952807,
        "transfer_pct": 92.88799756324381,
        "decode_pct": 1.3882568698033702
      }
    ]
  },
  "fp8_results": [
    {
      "format": "e4m3",
      "scheme": "top8",
      "coverage": 0.9217076450586319,
      "escapes": 21016444,
      "escape_rate": 0.0782923549413681,
      "raw_fp8_bytes": 268435456,
      "compressed_bytes": 287684278,
      "ratio_vs_fp8": 0.9330904624548165,
      "ratio_vs_bf16": 1.866180924909633,
      "encode_gbs": {
        "mean": 219.63626859917105,
        "std": 4.83605456358438,
        "stderr": 1.5292947309778224
      },
      "decode_gbs": {
        "mean": 366.85213357867895,
        "std": 7.927155394244954,
        "stderr": 2.506786641190408
      }
    },
    {
      "format": "e5m2",
      "scheme": "top8",
      "coverage": 0.9228173941373825,
      "escapes": 20718548,
      "escape_rate": 0.07718260586261749,
      "raw_fp8_bytes": 268435456,
      "compressed_bytes": 255974925,
      "ratio_vs_fp8": 1.0486787172610754,
      "ratio_vs_bf16": 2.097357434522151,
      "encode_gbs": {
        "mean": 221.5598316509351,
        "std": 46.04952335806594,
        "stderr": 14.562137897661389
      },
      "decode_gbs": {
        "mean": 340.80299937599676,
        "std": 2.8621462830927964,
        "stderr": 0.9050901251158312
      }
    },
    {
      "format": "e5m2",
      "scheme": "top16",
      "coverage": 0.9984399303793907,
      "escapes": 418778,
      "escape_rate": 0.0015600696206092834,
      "raw_fp8_bytes": 268435456,
      "compressed_bytes": 236242461,
      "ratio_vs_fp8": 1.1362709940614781,
      "ratio_vs_bf16": 2.2725419881229563,
      "encode_gbs": {
        "mean": 249.71338058813097,
        "std": 2.377780539658358,
        "stderr": 0.7519202281344738
      },
      "decode_gbs": {
        "mean": 564.858254817046,
        "std": 20.61851111814085,
        "stderr": 6.520145709483017
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
          "native_ms": 6.106968850207776,
          "splitzip_ms": 6.8555573192873185,
          "speedup": 0.8908055998637083,
          "ratio": 0.93869747149702,
          "encode_gbs": 40.031412136386336,
          "decode_gbs": 100.8472012859592,
          "escape_rate": 0.07573175430297852
        },
        {
          "seq_len": 2048,
          "native_ms": 12.140634348305467,
          "splitzip_ms": 12.874670984852502,
          "speedup": 0.9429859887362827,
          "ratio": 0.9303418747492151,
          "encode_gbs": 71.25446613407222,
          "decode_gbs": 225.08129000793446,
          "escape_rate": 0.0795588493347168
        },
        {
          "seq_len": 4096,
          "native_ms": 21.363469983295655,
          "splitzip_ms": 23.15134966968378,
          "speedup": 0.9227742783078727,
          "ratio": 0.9228232759248974,
          "encode_gbs": 122.7734357239877,
          "decode_gbs": 339.71217028733275,
          "escape_rate": 0.0830618143081665
        },
        {
          "seq_len": 8192,
          "native_ms": 43.15127359928215,
          "splitzip_ms": 49.485550575902366,
          "speedup": 0.8719974436395425,
          "ratio": 0.900446234949305,
          "encode_gbs": 157.5182646646834,
          "decode_gbs": 395.38129376374485,
          "escape_rate": 0.0938335657119751
        },
        {
          "seq_len": 16384,
          "native_ms": 86.2187714733486,
          "splitzip_ms": 91.20338303602914,
          "speedup": 0.9453461988278284,
          "ratio": 0.916145769163535,
          "encode_gbs": 190.00414430528286,
          "decode_gbs": 446.4297082174818,
          "escape_rate": 0.0862211138010025
        },
        {
          "seq_len": 32768,
          "native_ms": 175.18390946990343,
          "splitzip_ms": 187.50796488775794,
          "speedup": 0.9342744964181566,
          "ratio": 0.9606056002980548,
          "encode_gbs": 209.91769030847573,
          "decode_gbs": 482.0342955291343,
          "escape_rate": 0.06601335853338242
        },
        {
          "seq_len": 65536,
          "native_ms": 357.4450119788106,
          "splitzip_ms": 370.337407625284,
          "speedup": 0.9651874334565786,
          "ratio": 0.9330904624548165,
          "encode_gbs": 226.30115783700867,
          "decode_gbs": 366.237348351122,
          "escape_rate": 0.0782923549413681
        }
      ],
      "e5m2_top8_exact": [
        {
          "seq_len": 1024,
          "native_ms": 6.106968850207776,
          "splitzip_ms": 8.632210305741076,
          "speedup": 0.7074629363636076,
          "ratio": 0.8425205407248004,
          "encode_gbs": 33.73189981411775,
          "decode_gbs": 82.32247713906446,
          "escape_rate": 0.16607165336608887
        },
        {
          "seq_len": 2048,
          "native_ms": 12.140634348305467,
          "splitzip_ms": 12.987134172533967,
          "speedup": 0.9348201217464333,
          "ratio": 0.8530700465371507,
          "encode_gbs": 68.73469684343442,
          "decode_gbs": 173.01693239696706,
          "escape_rate": 0.16048002243041992
        },
        {
          "seq_len": 4096,
          "native_ms": 21.363469983295655,
          "splitzip_ms": 23.967156074567292,
          "speedup": 0.8913644120657881,
          "ratio": 0.8957640309108026,
          "encode_gbs": 124.78970919722141,
          "decode_gbs": 224.57918860565874,
          "escape_rate": 0.1391957402229309
        },
        {
          "seq_len": 8192,
          "native_ms": 43.15127359928215,
          "splitzip_ms": 44.73094522659297,
          "speedup": 0.9646850380802663,
          "ratio": 0.9444504553535753,
          "encode_gbs": 156.56029661540157,
          "decode_gbs": 267.7414921322961,
          "escape_rate": 0.11727246642112732
        },
        {
          "seq_len": 16384,
          "native_ms": 86.2187714733486,
          "splitzip_ms": 85.31210841929285,
          "speedup": 1.0106276010622042,
          "ratio": 1.0095528391991737,
          "encode_gbs": 191.76171349361215,
          "decode_gbs": 312.63424852014805,
          "escape_rate": 0.09126132726669312
        },
        {
          "seq_len": 32768,
          "native_ms": 175.18390946990343,
          "splitzip_ms": 159.63518750260565,
          "speedup": 1.097401595541359,
          "ratio": 1.0724349033499025,
          "encode_gbs": 209.64827775740773,
          "decode_gbs": 325.80205264598254,
          "escape_rate": 0.06913560628890991
        },
        {
          "seq_len": 65536,
          "native_ms": 357.4450119788106,
          "splitzip_ms": 336.74129613175006,
          "speedup": 1.0614825567427888,
          "ratio": 1.0486787172610754,
          "encode_gbs": 235.52574439525966,
          "decode_gbs": 341.1156223671478,
          "escape_rate": 0.07718260586261749
        }
      ],
      "e5m2_top16_exact": [
        {
          "seq_len": 1024,
          "native_ms": 6.106968850207776,
          "splitzip_ms": 7.903126303093517,
          "speedup": 0.7727282363964408,
          "ratio": 1.1229606314897214,
          "encode_gbs": 34.54925931428322,
          "decode_gbs": 77.8199461963873,
          "escape_rate": 0.005533933639526367
        },
        {
          "seq_len": 2048,
          "native_ms": 12.140634348305467,
          "splitzip_ms": 10.053636652028867,
          "speedup": 1.2075863459672012,
          "ratio": 1.1253899103523135,
          "encode_gbs": 72.57820791746326,
          "decode_gbs": 215.5897094586061,
          "escape_rate": 0.004801630973815918
        },
        {
          "seq_len": 4096,
          "native_ms": 21.363469983295655,
          "splitzip_ms": 18.997748629042746,
          "speedup": 1.1245264057571707,
          "ratio": 1.1297424904437419,
          "encode_gbs": 125.05617353453252,
          "decode_gbs": 322.3864693137545,
          "escape_rate": 0.003497481346130371
        },
        {
          "seq_len": 8192,
          "native_ms": 43.15127359928215,
          "splitzip_ms": 38.00820000012845,
          "speedup": 1.1353148425638764,
          "ratio": 1.1308461912603842,
          "encode_gbs": 167.36422328784582,
          "decode_gbs": 392.17910509567884,
          "escape_rate": 0.0031683743000030518
        },
        {
          "seq_len": 16384,
          "native_ms": 86.2187714733486,
          "splitzip_ms": 75.68026214504174,
          "speedup": 1.1392504337274854,
          "ratio": 1.1320224322029213,
          "encode_gbs": 210.5945950815015,
          "decode_gbs": 493.31443075084405,
          "escape_rate": 0.0028183460235595703
        },
        {
          "seq_len": 32768,
          "native_ms": 175.18390946990343,
          "splitzip_ms": 154.4295908263481,
          "speedup": 1.1343934056452498,
          "ratio": 1.1348490320847258,
          "encode_gbs": 234.18468810286657,
          "decode_gbs": 534.1181257700842,
          "escape_rate": 0.0019801557064056396
        },
        {
          "seq_len": 65536,
          "native_ms": 357.4450119788106,
          "splitzip_ms": 305.3666188410289,
          "speedup": 1.1705438313311292,
          "ratio": 1.1362709940614781,
          "encode_gbs": 249.16917487327723,
          "decode_gbs": 568.4048816046464,
          "escape_rate": 0.0015600696206092834
        }
      ]
    }
  }
}
