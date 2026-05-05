## Documentation for GW-Eyes

### Configs for GW-Eyes

The configuration parameters used by GW-Eyes are stored in `GW_Eyes/src/config`. The `config.yaml` file contains the default paths for databases and outputs. For example, if you want to change the catalogs accessed by GW-Eyes, you can do so by modifying `em_csv_paths`.

The `collector.yaml` and `executor.yaml` files contain the configuration parameters for the two sub-agents. For instance, you can guide the model’s behavior by modifying the system prompt `instructions`, or extend the available tools by changing `mcp_command`.

### Access to more datasets

#### 1. GW catalogs

We provide a way to use GW-Eyes to download the skymap posteriors for GWTC-2.1, GWTC-3, and GWTC-4. This can be done by running:
```python
python -m GW_Eyes.src.run_agent --agent collector
```
Then, for example, you can enter:
```
Please download the skymap of GWTC4 for me, thanks.
```
Please note that posterior files are relatively large, so the download may take a considerable amount of time. In addition, make sure that your network settings allow access to data from GWOSC.

In addition, we also provide a script for downloading posteriors:
```python
python -m recipe.download_gw_data.download_GWTC_skymap --catalogs gwtc4
```
Here, `gwtc4` can be replaced with `gwtc3` or `gwtc2p1`.

If you still encounter network issues, you can manually download the `xxx_skymaps_xxx.tar.gz` file and place it under `GW_Eyes/data/gwpe/tmp`. You can then ask GW-Eyes to post-process the file and construct the database.

```python
python -m GW_Eyes.src.run_agent --agent collector
```
Then, enter:
```
I have already downloaded the skymap of GWTC4, please try to perform post process to construct database, thanks.
```

#### 2. EM catalogs

We also support the integration of additional EM catalogs, such as `AGN-Flares` (see the [documentation](../../recipe/AGN-Flares/READMD.md)). You only need to convert the data to the same format as `GW_Eyes/data/sne/SNE.csv` and modify `em_csv_paths` in `GW_Eyes/src/config/config.yaml` to make it accessible to GW-Eyes.

### More cases

More GW-Eyes use cases are shown below. To enable the executor, run the following command:
```python
python -m GW_Eyes.src.run_agent
```

Try (quick verification)
```
For GW230518, please show me how a coordinate (ra_deg = 120.0, dec_deg = -30.0) is related to the localization of this gravitational-wave event.
```
(Information retrieval)
```
Show the skymap of GW230518
```
```
Show the information of AT2017gfo
```
(Visualization)
```
Show the skymaps of different waveforms for GW231123 in a single figure for a more intuitive comparison.
```

Besides, different languages are all acceptable, for example, Chinese:
```
请展示GW230518的空间方位图
```

### How to extend

Counterpart association relies on basic temporal and spatial overlap between GW and EM signals, while more specialized criteria can be readily incorporated by extending the agent’s toolset.

This means we need to expand more tools based on the structure of the data (by extending the available tools through changes to `mcp_command` in `GW_Eyes/src/config/executor.yaml`), and we can also instruct the agent about the filtering logic by modifying the system prompt `instructions`. If the logic is too complex or requires guidance through few-shot examples, we can use the provided Retrieval-augmented Generation (RAG) (by changing the contents under `GW_Eyes/knowledge`) to guide the model’s behavior. In that case, `--rag` needs to be added when starting the executor to enable RAG.
