## Recipe : Searching for AGN-Flare counterparts of gravitational waves using GW-Eyes

----

In this recipe, we demonstrate how to associate gravitational-wave events with AGN Flares using the GW-Eyes agent. The AGN-Flares data are taken from [L. He et al. (2025)](https://arxiv.org/abs/2507.20232), originally based on six years of observations from the [Zwicky Transient Facility (ZTF) Data Release 23](https://www.ztf.caltech.edu/ztf-public-releases.html). We show how to use the GW-Eyes agent to perform counterpart association.

### Install catalogs

Make sure that `GW_Eyes/recipe/AGN-Flares` is the current working directory:
```bash
mkdir data
cd data
git clone git@github.com:Lyle0831/AGN-Flares.git
python data_process.py
```

To ensure that the CSV files can be recognized by the agent, run:
```bash
python add_config.py --catalog AGNFRC
```
`AGNFRC` can also be replaced with `AGNFCC`.
- AGNFCC contains 28,504 entries.
- AGNFRC contains 1,984 high-confidence flares selected using stricter criteria.

For more details, see [L. He et al. (2025)](https://arxiv.org/abs/2507.20232). Alternatively, you can manually edit the file `GW_Eyes/src/config/EM_CSV.jsonl` to achieve the same effect.
