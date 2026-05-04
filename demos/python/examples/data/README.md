# Example Data

This folder holds datasets downloaded at runtime by the demo scripts. The data files themselves are gitignored — only this README and `.gitkeep` are tracked.

## Intel Berkeley Lab Sensor Data

**File:** `intel_lab_data.txt.gz` (33 MB, auto-downloaded)
**Source:** [MIT CSAIL — Intel Lab Data](http://db.csail.mit.edu/labdata/labdata.html)
**License:** Public domain with attribution requested.

54 Mica2Dot sensors deployed in the Intel Berkeley Research Lab collected temperature, humidity, light, and voltage readings every ~31 seconds from February 28 to April 5, 2004. The full dataset contains ~2.3 million readings.

Used by `davo_intel_lab.py` to demonstrate DAVO's decay-aware storage with real IoT time-series data.

### Download

The demo script downloads the file automatically on first run. To download manually:

```bash
curl -o demos/python/examples/data/intel_lab_data.txt.gz http://db.csail.mit.edu/labdata/data.txt.gz
```
