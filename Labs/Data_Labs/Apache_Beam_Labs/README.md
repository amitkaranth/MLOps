# Apache Beam — Word Frequency Analysis

Apache Beam pipeline lab submission as a part of the MLOps course, Data_Labs Lab.

##Name: Amit Karanth Gurpur
##Lab Assignment 5, due on 28th March 2026

## How to Run

You'll need **Python 3.9+** installed.

1. Open a terminal and `cd` into this folder:
   ```
   cd Labs/Data_Labs/Apache_Beam_Labs
   ```

2. Create a virtual environment and install dependencies:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Register the environment as a Jupyter kernel (so the notebook can use it):
   ```
   pip install ipykernel
   python -m ipykernel install --user --name=apache-beam-lab
   ```

4. Open `Try_Apache_Beam_Python.ipynb`.

5. Select the kernel — look for `.venv (Python 3.9.x)` or `apache-beam-lab` in the kernel picker (top-right of the notebook).

6. **Run All** cells from top to bottom. The pipeline takes a few seconds. You should see:
   - Word count tuples printed in Cell 4
   - A bar chart in Cell 6
   - Summary stats in Cell 8
   - A CSV export confirmation in Cell 10

That's it. The output files land in `outputs/`.

## What This Lab Does

- Reads the full text of *Romeo and Juliet* from `data/romeo_and_juliet.txt`
- Tokenizes every line into words using a regex
- Lowercases all words so "Romeo" and "ROMEO" count as the same word
- Filters out ~40 common English stop words (the, and, of, etc.)
- Counts how often each remaining word appears using `beam.CombinePerKey(sum)`
- Writes the raw results to `outputs/part-00000-of-00001`
- Plots a horizontal bar chart of the top 20 most frequent words
- Prints summary statistics (unique words, total occurrences, top/bottom 5)
- Exports the full sorted word list to `outputs/word_counts.csv`

Everything runs locally on the DirectRunner — no cloud setup needed.

## Changes from the Original

The original lab ran a basic word count on *King Lear* with no processing or analysis. Here's what's different:

- **Swapped the dataset** — replaced `kinglear.txt` with `romeo_and_juliet.txt` (downloaded from Project Gutenberg)
- **Added case normalization** — inserted a `beam.Map(str.lower)` step so words aren't split by capitalization
- **Added stop-word filtering** — inserted a `beam.Filter` step that removes common English words before counting
- **Added a bar chart** — new cell that reads the output, sorts by frequency, and plots the top 20 words with matplotlib
- **Added summary statistics** — new cell that prints total unique words, total occurrences, average frequency, and the top/bottom 5 words
- **Added CSV export** — new cell that writes the full results to a sorted CSV using pandas
- **Added `requirements.txt`** — the original had no dependency file; this one lists `apache-beam`, `pandas`, and `matplotlib`
