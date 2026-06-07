# LLM-Benchy: A Unified Benchmarking Framework for LLM Endpoints

To address the lack of standardized performance evaluation across large language model (LLM) serving infrastructures—analogous to the role of benchmark artifacts in additive manufacturing—we present **LLM-Benchy**, a lightweight benchmarking utility designed for universal compatibility with any OpenAI-compatible endpoint.

The tool leverages the **Speed Bench dataset** (provided by NVIDIA) to conduct category‑wise throughput and latency measurements. Key metrics include:

- **Token generation speed** (tokens per second) disaggregated by task category (e.g., reasoning, summarization, multilingual, coding).
- **Prompt processing throughput** (prompt tokens per second).
- **Concurrent request handling** across user‑defined concurrency levels.
- **Per‑category and aggregate latency** (time to first token, end‑to‑end latency , this tests specifically MTP next token prediction).

Results are automatically persisted to `benchmark-results.txt` with sensitive information (e.g., API keys) redacted.

Two deployment modes are supported:

1. **Standalone Python script** – direct execution with no additional dependencies beyond the standard `requests` library.
2. **Single‑file HTML dashboard** – interactive browser‑based frontend that requires the LLM server to enable **CORS** (`Access-Control-Allow-Origin` headers) to bypass cross‑origin restrictions.

The design prioritizes reproducibility, ease of integration into CI/CD pipelines, and cross‑engine compatibility (vLLM, SGLang, llama.cpp, etc.).


The html file can simply be put onto your desktop and you can run it.
The python file requires you to pip install requests, datasets and openai packages

## WebUI
<img width="1043" height="1153" alt="image" src="https://github.com/user-attachments/assets/6ff6c4a7-167d-4e54-91a2-f273633b35e4" />




## Sample output result benchmark-results.txt (either created by the python script OR after the benchmark is done in the WebUI you can click the copy button to get the data into your clipboard to post it on reddit/github or whereever you wanna see your benchmark scores

```
Benchmark run at 2026-06-07 04:20:29
Model: Qwen3.6 27B
Base URL: http://snabox:16384/v1
Inference engine: vllm
System description (CPU/RAM): Ryzen 3 3600X, 64GB ddr4
System description (GPU): 2 x RTX 2080 Ti with NVLINK, Power limit 146W/card (upgraded 22GB vram each)
Concurrency levels: [1, 2, 4, 8]
Samples per concurrency multiplier: 2
Temperature: 0.1
Max tokens: 1024


Concurrency = 1
category       samples  avg_prompt_t/s  agg_prompt_t/s  avg_pred_t/s    agg_pred_t/s  avg_latency
-------------- ------- --------------- --------------- ------------- --------------- ------------
coding               2          707.44          707.44         87.08           87.08      11.984s
humanities           2          273.89          273.89         74.89           74.89      13.797s
math                 2          275.76          275.76         74.57           74.57      13.829s
multilingual         2          424.42          424.42         83.77           83.77       3.878s
qa                   2          195.69          195.69         80.99           80.99       4.922s
rag                  2          865.43          865.43         88.72           88.72      10.284s
reasoning            2          275.18          275.18         74.71           74.71      13.809s
roleplay             2          448.22          448.22         76.65           76.65      12.611s
stem                 2          269.97          269.97         75.49           75.49      13.663s
summarization        2          297.95          297.95         80.42           80.42       8.724s
writing              2          875.18          875.18         76.72           76.72      14.200s
overall             22          446.28          446.28         79.45           79.45      11.064s

Concurrency = 2
category       samples  avg_prompt_t/s  agg_prompt_t/s  avg_pred_t/s    agg_pred_t/s  avg_latency
-------------- ------- --------------- --------------- ------------- --------------- ------------
coding               4          416.57          833.14         62.90          125.80      16.607s
humanities           4          179.36          358.73         52.94          105.88      19.529s
math                 4          185.08          370.16         54.46          108.92      18.964s
multilingual         4          262.28          524.56         67.29          134.58      13.119s
qa                   4          136.08          272.15         60.77          121.53      12.844s
rag                  4          687.08         1374.16         62.38          124.77      15.281s
reasoning            4          164.24          328.47         54.91          109.83      18.813s
roleplay             4          331.12          662.23         56.35          112.71      17.983s
stem                 4          165.17          330.33         54.94          109.87      18.807s
summarization        4          253.19          506.39         58.08          116.16      12.462s
writing              4          486.67          973.34         57.20          114.40      19.002s
overall             44          296.98          593.97         58.38          116.77      16.674s

Concurrency = 4
category       samples  avg_prompt_t/s  agg_prompt_t/s  avg_pred_t/s    agg_pred_t/s  avg_latency
-------------- ------- --------------- --------------- ------------- --------------- ------------
coding               8          407.94         1631.78         59.66          238.62      17.609s
humanities           8          128.33          513.34         54.34          217.35      19.070s
math                 8          142.29          569.16         55.21          220.86      18.873s
multilingual         8          255.25         1021.02         62.24          248.98      13.129s
qa                   8          102.07          408.27         55.91          223.65      13.881s
rag                  8          464.29         1857.14         59.27          237.06      14.237s
reasoning            8          123.75          495.00         53.38          213.53      19.400s
roleplay             8          285.01         1140.03         52.60          210.42      19.739s
stem                 8          145.67          582.70         52.80          211.20      19.605s
summarization        8          187.52          750.08         56.70          226.82      12.902s
writing              8          469.79         1879.15         52.94          211.78      20.968s
overall             88          246.54          986.15         55.92          223.66      17.219s

Concurrency = 8
category       samples  avg_prompt_t/s  agg_prompt_t/s  avg_pred_t/s    agg_pred_t/s  avg_latency
-------------- ------- --------------- --------------- ------------- --------------- ------------
coding              16          310.96         2487.72         52.97          423.78      20.097s
humanities          16           86.66          693.28         47.75          382.01      21.779s
math                16          101.60          812.82         48.57          388.54      21.535s
multilingual        16          207.72         1661.79         55.27          442.16      15.840s
qa                  16           79.46          635.64         50.47          403.72      17.106s
rag                 16          356.38         2851.05         49.91          399.27      21.292s
reasoning           16           91.60          732.77         47.30          378.38      21.963s
roleplay            16          212.65         1701.16         46.05          368.42      22.652s
stem                16           92.21          737.66         47.55          380.38      21.860s
summarization       16          117.37          938.94         50.31          402.47      15.084s
writing             16          341.90         2735.19         43.30          346.40      26.488s
overall            176          181.68         1453.46         49.04          392.32      20.518s

```
