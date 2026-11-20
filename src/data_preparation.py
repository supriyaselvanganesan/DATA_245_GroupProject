"""Python export of data_preparation.ipynb. Notebook outputs are omitted."""

from IPython.display import display

# %% [markdown] cell 0
# # ClinVar Missense Variant Dataset Preparation
#
# This notebook prepares the shared dataset for our Group 8 project, where we predict whether a missense variant is pathogenic or benign. A missense variant is a DNA change that replaces one amino acid in a protein with another.
#
# The generated datasets and feature configuration are saved in the `data/processed/` folder.
#
# The notebook downloads the ClinVar data, keeps the variants we need, adds conservation, population frequency and amino acid features, checks the result and saves the files below. Missing values, scaling and model training are left to each team member.
#
# ## Output files
#
# 1. **clinvar_expanded_model_features.csv**: the file for training. It has the variant ID, gene name, 28 features and the label (1 for pathogenic, 0 for benign).
# 2. **clinvar_expanded_project_dataset.csv**: the same variants with extra details such as genome position and scores from tools like CADD and REVEL. Use it for comparison only, not as model input.
# 3. **clinvar_full_annotated_all_rows.csv**: every original row, with the reason each removed row was dropped.
# 4. **feature_config.json**: the column lists for feature sets A, B and D.
#
# Some population frequency values are missing. They are left empty so each person can fill them in their own way.

# %% [markdown] cell 1
# ## 1. Setup
#
# Install Biopython for the BLOSUM62 table and create the working folder.

# %% code cell 2
# Install Biopython in the environment before running this notebook.

# %% code cell 3
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import time
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio.Align import substitution_matrices

pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 180)

# Find the repository root when run from the root or notebooks/ folder.
SEARCH_DIRS = (Path.cwd(), *Path.cwd().parents)
BASE_DIR = next((p for p in SEARCH_DIRS if (p / "data" / "processed").is_dir()), Path.cwd())
WORKDIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"
WORKDIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://zenodo.org/records/15275244/files/"
    "cleaned_ClinVar_with_preds.csv?download=1"
)
DATA_FILE = RAW_DIR / "cleaned_ClinVar_with_preds.csv"
EXPECTED_MD5 = "64d4ea4deea22e6835b6ec713fb7645b"

print("Working directory:", WORKDIR)
print("Platform:", platform.platform())
print("Python:", platform.python_version())
print("pandas:", pd.__version__)
print("numpy:", np.__version__)

# %% [markdown] cell 4
# ## 2. Download the data
#
# Download the cleaned ClinVar file and check that it matches the expected checksum, so everyone uses the same file.

# %% code cell 5
def download_file(url, destination):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print("Using existing file:", destination)
        return
    print("Downloading:", url)
    urllib.request.urlretrieve(url, destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1024**2:.1f} MB)")


def md5sum(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


download_file(DATA_URL, DATA_FILE)

actual_md5 = md5sum(DATA_FILE)
print("Expected MD5:", EXPECTED_MD5)
print("Actual MD5:  ", actual_md5)

if actual_md5 != EXPECTED_MD5:
    raise ValueError("Checksum mismatch. Delete the file and download it again.")

# %% [markdown] cell 6
# ## 3. Load the data
#
# Look at the row count, labels and genome version before filtering.

# %% code cell 7
df = pd.read_csv(DATA_FILE, low_memory=False)
# pandas reads the change N/A (asparagine to alanine) as missing, so this column is read as plain text
df["Amino_acids"] = pd.read_csv(
    DATA_FILE, usecols=["Amino_acids"], dtype=str, keep_default_na=False
)["Amino_acids"].replace("", np.nan)
source_cols = list(df.columns)

required_cols = [
    "VariationID",
    "GeneSymbol",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "Assembly",
    "BinaryClinicalSignificance",
    "Amino_acids",
]
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise KeyError(f"Required columns are missing: {missing_required}")

if df["VariationID"].duplicated().any():
    raise ValueError("VariationID is not unique in the source file.")

print("Rows:", len(df))
print("Columns:", df.shape[1])
print("Unique genes:", df["GeneSymbol"].nunique())

display(df.head())
display(df["BinaryClinicalSignificance"].value_counts(dropna=False).to_frame("count"))
display(df["Assembly"].value_counts(dropna=False).to_frame("count"))

# %% [markdown] cell 8
# ## 4. Select variants
#
# A variant is kept only if:
#
# 1. It is on the GRCh38 genome, which the conservation scores use.
# 2. Its label is B or P.
# 3. It has a valid position.
# 4. It is a single DNA letter change.
# 5. It changes one amino acid into a different one.
# 6. It belongs to one gene.
# 7. It is not a duplicate. If duplicates have different labels, all copies are removed.
#
# Removed rows are saved in the archive file with the reason.

# %% code cell 9
BASES = {"A", "C", "G", "T"}
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def parse_amino_acids(value):
    # The source file separates the two residues with a slash or a greater than sign
    parts = [p.strip() for p in str(value).strip().upper().replace(">", "/").split("/")]
    if len(parts) == 2 and parts[0] in STANDARD_AA and parts[1] in STANDARD_AA:
        return parts[0], parts[1]
    return np.nan, np.nan


assembly = df["Assembly"].fillna("").astype(str).str.strip().str.upper()
labels = df["BinaryClinicalSignificance"].fillna("").astype(str).str.strip().str.upper()
position = pd.to_numeric(df["PositionVCF"], errors="coerce")
chrom = df["Chromosome"].fillna("").astype(str).str.strip()
ref = df["ReferenceAlleleVCF"].fillna("").astype(str).str.strip().str.upper()
alt = df["AlternateAlleleVCF"].fillna("").astype(str).str.strip().str.upper()
gene = df["GeneSymbol"].fillna("").astype(str).str.strip()

aa_pairs = pd.DataFrame(
    df["Amino_acids"].map(parse_amino_acids).tolist(),
    index=df.index,
    columns=["AA_ref", "AA_alt"],
)
df["AA_ref"] = aa_pairs["AA_ref"]
df["AA_alt"] = aa_pairs["AA_alt"]

is_snv = ref.isin(BASES) & alt.isin(BASES) & ref.ne(alt)
is_missense = df["AA_ref"].notna() & df["AA_alt"].notna() & df["AA_ref"].ne(df["AA_alt"])
single_gene = ~gene.isin(["", "-", "nan", "None"]) & ~gene.str.contains(r"[;,|]")

rules = [
    ("not_grch38", ~assembly.str.startswith("GRCH38")),
    ("label_not_b_or_p", ~labels.isin(["B", "P"])),
    ("invalid_position", position.isna() | position.lt(1)),
    ("not_snv", ~is_snv),
    ("not_missense", ~is_missense),
    ("gene_missing_or_multiple", ~single_gene),
]

df["exclusion_reason"] = pd.Series(np.nan, index=df.index, dtype="object")
for reason, failed in rules:
    df.loc[df["exclusion_reason"].isna() & failed, "exclusion_reason"] = reason

# Repeated genomic changes among the rows that are still kept
kept = df["exclusion_reason"].isna()
keys = pd.DataFrame({"chrom": chrom, "pos": position, "ref": ref, "alt": alt, "label": labels}).loc[kept]
key_cols = ["chrom", "pos", "ref", "alt"]
n_labels = keys.groupby(key_cols)["label"].transform("nunique")
conflicting = n_labels.gt(1)
repeated = keys.duplicated(key_cols) & ~conflicting

df.loc[conflicting[conflicting].index, "exclusion_reason"] = "conflicting_duplicate"
df.loc[repeated[repeated].index, "exclusion_reason"] = "duplicate_position"

reason_order = [r for r, _ in rules] + ["conflicting_duplicate", "duplicate_position"]
removed = df["exclusion_reason"].value_counts().reindex(reason_order, fill_value=0)

display(removed.to_frame("rows_removed"))
print("Rows kept:", int(df["exclusion_reason"].isna().sum()), "of", len(df))

# %% [markdown] cell 10
# ## 5. Prepare positions for UCSC
#
# UCSC counts positions from 0 and ClinVar counts from 1, so a variant at position p is looked up as p minus 1 to p. Each position is tagged with its row number so the scores can be matched back.

# %% code cell 11
def ucsc_chrom(value):
    value = str(value).strip()
    if value.startswith("chr"):
        return value
    if value in {"M", "MT", "25"}:
        return "chrM"
    return "chr" + value


cohort = df["exclusion_reason"].isna()

bed = pd.DataFrame({
    "chrom": chrom[cohort].map(ucsc_chrom),
    "start": position[cohort].astype("int64") - 1,
    "end": position[cohort].astype("int64"),
    "name": "row_" + df.index.to_series()[cohort].astype(str),
})

chrom_order = {f"chr{i}": i for i in range(1, 23)}
chrom_order.update({"chrX": 23, "chrY": 24, "chrM": 25})

bed["_order"] = bed["chrom"].map(chrom_order).fillna(99)
bed = (
    bed.sort_values(["_order", "chrom", "start"])
    .drop(columns="_order")
    .reset_index(drop=True)
)

BED_FILE = WORKDIR / "clinvar_missense_sites_hg38_sorted.bed"
bed.to_csv(BED_FILE, sep="\t", header=False, index=False)

print("BED rows:", len(bed))
display(bed.head())

# %% [markdown] cell 12
# ## 6. Get the UCSC tool
#
# Download the UCSC program that reads conservation scores.

# %% code cell 13
TOOL_BUILDS = {"Linux": "linux.x86_64", "Darwin": "macOSX.x86_64"}

system = platform.system()
if system not in TOOL_BUILDS:
    raise OSError("Run this notebook on Linux or macOS, for example in Google Colab.")

UCSC_TOOL = WORKDIR / "bigWigAverageOverBed"
UCSC_TOOL_URL = (
    f"https://hgdownload.soe.ucsc.edu/admin/exe/{TOOL_BUILDS[system]}/"
    "bigWigAverageOverBed"
)

if not UCSC_TOOL.exists():
    print("Downloading", UCSC_TOOL_URL)
    urllib.request.urlretrieve(UCSC_TOOL_URL, UCSC_TOOL)
    UCSC_TOOL.chmod(0o755)

# Running the tool without arguments prints its usage line
result = subprocess.run([str(UCSC_TOOL)], capture_output=True, text=True)
usage = (result.stdout + result.stderr).strip().splitlines()
print(usage[0] if usage else "Tool installed.")

# %% [markdown] cell 14
# ## 7. Get conservation scores
#
# Conservation shows whether a DNA position has stayed the same across species. Changes at highly conserved positions are more likely to be harmful.
#
# We get phyloP and phastCons scores for primates, mammals and vertebrates, six scores in total. This step is slow. Finished scores are saved, so they are reused if the run stops.

# %% code cell 15
TRACKS = {
    "phyloP17way_primate": "phyloP17way/hg38.phyloP17way.bw",
    "phastCons17way_primate": "phastCons17way/hg38.phastCons17way.bw",
    "phyloP470way_mammalian": "phyloP470way/hg38.phyloP470way.bw",
    "phastCons470way_mammalian": "phastCons470way/hg38.phastCons470way.bw",
    "phyloP100way_vertebrate": "phyloP100way/hg38.phyloP100way.bw",
    "phastCons100way_vertebrate": "phastCons100way/hg38.phastCons100way.bw",
}
TRACK_BASE_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
conservation_cols = list(TRACKS)


def line_count(path):
    with open(path) as f:
        return sum(1 for _ in f)


def extract_track(track_name, url, retries=3):
    out_path = WORKDIR / f"{track_name}_average_over_bed.tsv"

    if out_path.exists() and line_count(out_path) == len(bed):
        print(f"{track_name}: using saved output")
        return out_path

    cmd = [str(UCSC_TOOL), url, str(BED_FILE), str(out_path)]

    for attempt in range(1, retries + 1):
        print(f"{track_name}: attempt {attempt} of {retries}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and out_path.exists() and line_count(out_path) == len(bed):
            return out_path

        print("Extraction failed or returned incomplete output.")
        print((result.stdout + result.stderr)[-1000:])
        out_path.unlink(missing_ok=True)
        if attempt < retries:
            time.sleep(8 * attempt)

    raise RuntimeError(f"Failed to extract {track_name}")


def load_track_scores(path):
    cols = ["name", "size", "covered", "sum", "mean0", "mean"]
    result = pd.read_csv(path, sep="\t", header=None, names=cols)

    if set(result["name"]) != set(bed["name"]):
        raise ValueError(f"{path.name} does not match the BED intervals.")

    result["row_index"] = result["name"].str.replace("row_", "", regex=False).astype("int64")
    # Intervals with no covered base have no score
    result["score"] = pd.to_numeric(result["mean"], errors="coerce").mask(result["covered"].eq(0))
    return result.set_index("row_index")["score"]


for track_name, track_path in TRACKS.items():
    scores = load_track_scores(extract_track(track_name, TRACK_BASE_URL + track_path))
    df[track_name] = np.nan
    df.loc[scores.index, track_name] = scores.values
    print(f"{track_name}: {df.loc[cohort, track_name].notna().mean():.2%} of cohort scored\n")

# %% [markdown] cell 16
# ## 8. Keep variants with all six scores
#
# Variants missing any conservation score are removed.

# %% code cell 17
coverage = pd.DataFrame({
    "missing": df.loc[cohort, conservation_cols].isna().sum(),
    "coverage_percent": 100 * df.loc[cohort, conservation_cols].notna().mean(),
})
display(coverage)

missing_conservation = cohort & df[conservation_cols].isna().any(axis=1)
df.loc[missing_conservation, "exclusion_reason"] = "missing_conservation"
print("Removed for missing conservation:", int(missing_conservation.sum()))

expanded = df.loc[df["exclusion_reason"].isna()].copy()
expanded["BinaryClinicalSignificance"] = labels.loc[expanded.index]
expanded["label"] = expanded["BinaryClinicalSignificance"].map({"B": 0, "P": 1}).astype("int8")

print("Final cohort:", expanded.shape)
print("Unique genes:", expanded["GeneSymbol"].nunique())
display(expanded["BinaryClinicalSignificance"].value_counts().to_frame("count"))

# %% [markdown] cell 18
# ## 9. Population frequency features
#
# This shows how common a variant is in gnomAD. Harmful variants are usually rare.
#
# For each of the three gnomAD frequency columns we keep:
#
# 1. The frequency.
# 2. A flag that is 1 when the frequency is missing.
# 3. A rarity score, the negative log10 of the frequency. For example, 0.0001 becomes 4, so higher means rarer.
#
# Missing values are left empty.

# %% code cell 19
AF_COLS = ["gnomADe_AF", "gnomADg_AF", "gnomAD_AF"]
missing_af = [c for c in AF_COLS if c not in expanded.columns]
if missing_af:
    raise KeyError(f"Allele frequency columns are missing: {missing_af}")

EPS = 1e-8
population_feature_cols = []

for col in AF_COLS:
    values = pd.to_numeric(expanded[col], errors="coerce")
    out_of_range = values.lt(0) | values.gt(1)
    values = values.mask(out_of_range)

    expanded[col] = values
    expanded[f"{col}_missing"] = values.isna().astype("int8")
    expanded[f"{col}_neglog10"] = -np.log10(values + EPS)

    population_feature_cols += [col, f"{col}_missing", f"{col}_neglog10"]
    print(f"{col}: {int(out_of_range.sum())} values outside 0 to 1 set to missing")

display(pd.DataFrame({
    "missing": expanded[AF_COLS].isna().sum(),
    "missing_percent": 100 * expanded[AF_COLS].isna().mean(),
    "median_nonmissing": expanded[AF_COLS].median(),
}))

# %% [markdown] cell 20
# ## 10. Amino acid features
#
# These describe how different the new amino acid is from the original. For hydropathy, molecular weight, size, polarity and charge, we keep the change and its absolute size. We also record whether aromatic status changes.

# %% code cell 21
AA = {
    "A": {"hydropathy": 1.8,  "mw": 89.09,  "volume": 88.6,  "polarity": 8.1,  "charge": 0,  "aromatic": 0},
    "R": {"hydropathy": -4.5, "mw": 174.20, "volume": 173.4, "polarity": 10.5, "charge": 1,  "aromatic": 0},
    "N": {"hydropathy": -3.5, "mw": 132.12, "volume": 114.1, "polarity": 11.6, "charge": 0,  "aromatic": 0},
    "D": {"hydropathy": -3.5, "mw": 133.10, "volume": 111.1, "polarity": 13.0, "charge": -1, "aromatic": 0},
    "C": {"hydropathy": 2.5,  "mw": 121.16, "volume": 108.5, "polarity": 5.5,  "charge": 0,  "aromatic": 0},
    "Q": {"hydropathy": -3.5, "mw": 146.15, "volume": 143.8, "polarity": 10.5, "charge": 0,  "aromatic": 0},
    "E": {"hydropathy": -3.5, "mw": 147.13, "volume": 138.4, "polarity": 12.3, "charge": -1, "aromatic": 0},
    "G": {"hydropathy": -0.4, "mw": 75.07,  "volume": 60.1,  "polarity": 9.0,  "charge": 0,  "aromatic": 0},
    "H": {"hydropathy": -3.2, "mw": 155.16, "volume": 153.2, "polarity": 10.4, "charge": 0,  "aromatic": 1},
    "I": {"hydropathy": 4.5,  "mw": 131.17, "volume": 166.7, "polarity": 5.2,  "charge": 0,  "aromatic": 0},
    "L": {"hydropathy": 3.8,  "mw": 131.17, "volume": 166.7, "polarity": 4.9,  "charge": 0,  "aromatic": 0},
    "K": {"hydropathy": -3.9, "mw": 146.19, "volume": 168.6, "polarity": 11.3, "charge": 1,  "aromatic": 0},
    "M": {"hydropathy": 1.9,  "mw": 149.21, "volume": 162.9, "polarity": 5.7,  "charge": 0,  "aromatic": 0},
    "F": {"hydropathy": 2.8,  "mw": 165.19, "volume": 189.9, "polarity": 5.2,  "charge": 0,  "aromatic": 1},
    "P": {"hydropathy": -1.6, "mw": 115.13, "volume": 112.7, "polarity": 8.0,  "charge": 0,  "aromatic": 0},
    "S": {"hydropathy": -0.8, "mw": 105.09, "volume": 89.0,  "polarity": 9.2,  "charge": 0,  "aromatic": 0},
    "T": {"hydropathy": -0.7, "mw": 119.12, "volume": 116.1, "polarity": 8.6,  "charge": 0,  "aromatic": 0},
    "W": {"hydropathy": -0.9, "mw": 204.23, "volume": 227.8, "polarity": 5.4,  "charge": 0,  "aromatic": 1},
    "Y": {"hydropathy": -1.3, "mw": 181.19, "volume": 193.6, "polarity": 6.2,  "charge": 0,  "aromatic": 1},
    "V": {"hydropathy": 4.2,  "mw": 117.15, "volume": 140.0, "polarity": 5.9,  "charge": 0,  "aromatic": 0},
}
AA_TABLE = pd.DataFrame(AA).T.astype(float)

aa_feature_cols = []

for prop in ["hydropathy", "mw", "volume", "polarity", "charge"]:
    change = expanded["AA_alt"].map(AA_TABLE[prop]) - expanded["AA_ref"].map(AA_TABLE[prop])
    expanded[f"AA_{prop}_change"] = change
    expanded[f"AA_{prop}_abs_change"] = change.abs()
    aa_feature_cols += [f"AA_{prop}_change", f"AA_{prop}_abs_change"]

aromatic_change = expanded["AA_alt"].map(AA_TABLE["aromatic"]) - expanded["AA_ref"].map(AA_TABLE["aromatic"])
expanded["AA_aromatic_change"] = aromatic_change
expanded["AA_aromatic_status_changed"] = aromatic_change.abs()
aa_feature_cols += ["AA_aromatic_change", "AA_aromatic_status_changed"]

display(expanded[["AA_ref", "AA_alt"] + aa_feature_cols].head())

# %% [markdown] cell 22
# ## 11. BLOSUM62 score
#
# BLOSUM62 scores how common an amino acid swap is in nature. A negative score means a rare swap, which is more likely to be harmful.

# %% code cell 23
blosum62 = substitution_matrices.load("BLOSUM62")

expanded["BLOSUM62_score"] = [
    float(blosum62[r, a]) for r, a in zip(expanded["AA_ref"], expanded["AA_alt"])
]
aa_feature_cols.append("BLOSUM62_score")

print("BLOSUM62 coverage:", f"{expanded['BLOSUM62_score'].notna().mean():.2%}")

# %% [markdown] cell 24
# ## 12. Protein position
#
# Saved for analysis only. It is not a feature because the same position means different things in different proteins.

# %% code cell 25
if "Protein_position" in expanded.columns:
    # A position range keeps its first position
    expanded["Protein_position_numeric"] = pd.to_numeric(
        expanded["Protein_position"].astype(str).str.split("-").str[0],
        errors="coerce",
    )
    print("Protein position coverage:", f"{expanded['Protein_position_numeric'].notna().mean():.2%}")

# %% [markdown] cell 26
# ## 13. Published tool scores
#
# Scores from tools like CADD, REVEL and AlphaMissense are kept for comparison, not for training.

# %% code cell 27
benchmark_candidates = [
    "CADD_PHRED_score",
    "CADD_RAW_score",
    "REVEL_score",
    "SIFT_score",
    "PolyPhen_score",
    "BayesDel_score",
    "ClinPred_score",
    "VEST4_score",
    "VEST4_rankscore",
    "EVE_score",
    "PrimateAI_score",
    "AM_score",
]
benchmark_cols = [c for c in benchmark_candidates if c in expanded.columns]

display(pd.DataFrame({
    "available": expanded[benchmark_cols].notna().sum(),
    "coverage_percent": 100 * expanded[benchmark_cols].notna().mean(),
}).sort_values("coverage_percent", ascending=False))

# %% [markdown] cell 28
# ## 14. Feature sets
#
# 1. Set A: 6 conservation features.
# 2. Set B: set A plus 9 population features, 15 in total.
# 3. Set D: set B plus 13 amino acid features, 28 in total.

# %% code cell 29
FEATURE_GROUPS = {
    "conservation": conservation_cols,
    "population": population_feature_cols,
    "amino_acid": aa_feature_cols,
}

FEATURE_SETS = {
    "A": conservation_cols,
    "B": conservation_cols + population_feature_cols,
    "D": conservation_cols + population_feature_cols + aa_feature_cols,
}

expected_sizes = {"A": 6, "B": 15, "D": 28}
for name, cols in FEATURE_SETS.items():
    if len(cols) != expected_sizes[name] or len(set(cols)) != len(cols):
        raise ValueError(f"Feature set {name} has {len(cols)} columns, expected {expected_sizes[name]}.")

primary_feature_cols = FEATURE_SETS["D"]

for group, cols in FEATURE_GROUPS.items():
    print(f"{group} ({len(cols)} features)")
    for col in cols:
        print("   ", col)
    print()

# %% [markdown] cell 30
# ## 15. Final checks
#
# If any check fails, the notebook stops before saving.

# %% code cell 31
variant_key = ["Chromosome", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
af_values = expanded[AF_COLS]

checks = {
    "VariationID is unique": not expanded["VariationID"].duplicated().any(),
    "No repeated genomic change": not expanded.duplicated(variant_key).any(),
    "Labels are only B and P": expanded["BinaryClinicalSignificance"].isin(["B", "P"]).all(),
    "Numeric label matches B and P": (expanded["label"] == expanded["BinaryClinicalSignificance"].eq("P").astype("int8")).all(),
    "Every row is a missense change": expanded["AA_ref"].ne(expanded["AA_alt"]).all(),
    "Every row has one gene": expanded["GeneSymbol"].notna().all() and not expanded["GeneSymbol"].astype(str).str.contains(r"[;,|]").any(),
    "No missing conservation scores": expanded[conservation_cols].notna().all().all(),
    "No missing amino acid features": expanded[aa_feature_cols].notna().all().all(),
    "Allele frequencies within 0 to 1": (af_values.isna() | (af_values.ge(0) & af_values.le(1))).all().all(),
}

check_table = pd.Series(checks, name="passed").to_frame()
display(check_table)

failed = check_table.index[~check_table["passed"]].tolist()
if failed:
    raise ValueError(f"Quality checks failed: {failed}")

display(pd.Series({
    "rows": len(expanded),
    "unique_genes": expanded["GeneSymbol"].nunique(),
    "benign": int(expanded["label"].eq(0).sum()),
    "pathogenic": int(expanded["label"].eq(1).sum()),
    "pathogenic_percent": round(100 * expanded["label"].mean(), 2),
}, name="value").to_frame())

feature_missing = pd.DataFrame({
    "missing": expanded[primary_feature_cols].isna().sum(),
    "missing_percent": 100 * expanded[primary_feature_cols].isna().mean(),
})
display(feature_missing[feature_missing["missing"] > 0].sort_values("missing_percent", ascending=False))

# %% [markdown] cell 32
# ## 16. Feature averages by class
#
# Pathogenic variants should be more conserved, rarer and have lower BLOSUM62 scores than benign ones.

# %% code cell 33
check_cols = (
    conservation_cols
    + [c for c in population_feature_cols if c.endswith("_neglog10")]
    + [c for c in aa_feature_cols if c.endswith("_abs_change")]
    + ["BLOSUM62_score"]
)

display(expanded.groupby("BinaryClinicalSignificance")[check_cols].mean().T)

# %% [markdown] cell 34
# ## 17. Feature correlation
#
# Shows which features move together. Conservation scores are expected to correlate with each other, and so are the gnomAD frequencies.

# %% code cell 35
corr = expanded[primary_feature_cols].corr(method="spearman")

plt.figure(figsize=(14, 12))
im = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
plt.colorbar(im, label="Spearman correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
plt.title("Correlation of the 28 Biological Features")
plt.tight_layout()
plt.show()

# %% [markdown] cell 36
# ## 18. Save the files
#
# Save the outputs and print a checksum for each, so the team can confirm they have the same files.

# %% code cell 37
identifier_cols = [
    "VariationID",
    "GeneSymbol",
    "Chromosome",
    "PositionVCF",
    "ReferenceAlleleVCF",
    "AlternateAlleleVCF",
    "Assembly",
]
raw_biology_cols = [c for c in ["Amino_acids", "Variant", "Protein_position"] if c in expanded.columns]
extra_cols = [c for c in ["AA_ref", "AA_alt", "Protein_position_numeric"] if c in expanded.columns]
target_cols = ["BinaryClinicalSignificance", "label"]

model_cols = ["VariationID", "GeneSymbol"] + primary_feature_cols + target_cols
full_cols = list(dict.fromkeys(
    identifier_cols + raw_biology_cols + primary_feature_cols + extra_cols + benchmark_cols + target_cols
))
archive_cols = list(dict.fromkeys(source_cols + ["AA_ref", "AA_alt"] + conservation_cols + ["exclusion_reason"]))

MODEL_OUTPUT = WORKDIR / "clinvar_expanded_model_features.csv"
FULL_OUTPUT = WORKDIR / "clinvar_expanded_project_dataset.csv"
ARCHIVE_OUTPUT = WORKDIR / "clinvar_full_annotated_all_rows.csv"
CONFIG_OUTPUT = WORKDIR / "feature_config.json"

expanded[model_cols].to_csv(MODEL_OUTPUT, index=False)
expanded[full_cols].to_csv(FULL_OUTPUT, index=False)
df[archive_cols].to_csv(ARCHIVE_OUTPUT, index=False)

feature_config = {
    "target": "label",
    "group_column": "GeneSymbol",
    "feature_groups": FEATURE_GROUPS,
    "feature_sets": FEATURE_SETS,
}
with open(CONFIG_OUTPUT, "w") as f:
    json.dump(feature_config, f, indent=2)

outputs = {
    MODEL_OUTPUT: expanded[model_cols].shape,
    FULL_OUTPUT: expanded[full_cols].shape,
    ARCHIVE_OUTPUT: df[archive_cols].shape,
    CONFIG_OUTPUT: None,
}

display(pd.DataFrame([
    {
        "file": path.name,
        "rows": shape[0] if shape else "",
        "columns": shape[1] if shape else "",
        "size_mb": round(path.stat().st_size / 1024**2, 2),
        "md5": md5sum(path),
    }
    for path, shape in outputs.items()
]))

# %% [markdown] cell 38
# ## 19. Summary

# %% code cell 39
summary = pd.Series({
    "Rows in source file": len(df),
    "Rows removed by cohort checks": int((df["exclusion_reason"].notna() & df["exclusion_reason"].ne("missing_conservation")).sum()),
    "Rows removed for missing conservation": int(df["exclusion_reason"].eq("missing_conservation").sum()),
    "Rows in final dataset": len(expanded),
    "Unique genes": expanded["GeneSymbol"].nunique(),
    "Benign": int(expanded["label"].eq(0).sum()),
    "Pathogenic": int(expanded["label"].eq(1).sum()),
    "Conservation features": len(conservation_cols),
    "Population features": len(population_feature_cols),
    "Amino acid features": len(aa_feature_cols),
    "Total features": len(primary_feature_cols),
    "Benchmark predictors kept": len(benchmark_cols),
}, name="value")

display(summary.to_frame())

# %% [markdown] cell 40
# ## 20. Download the files
#
# Downloads the model file, project file and feature list in Colab. Remove the # on the last line to also download the archive file.

# %% code cell 41
try:
    from google.colab import files
except ImportError:
    files = None

if files is None:
    print("Not running in Colab. The files are saved in", WORKDIR)
else:
    files.download(str(MODEL_OUTPUT))
    files.download(str(FULL_OUTPUT))
    files.download(str(CONFIG_OUTPUT))
    # files.download(str(ARCHIVE_OUTPUT))

# %% [markdown] cell 42
# ## 21. Using the files
#
# How to load the data and select feature set A, B or D. Make any further changes in your own notebook and keep the shared files unchanged.

# %% code cell 43
model_df = pd.read_csv(MODEL_OUTPUT)

with open(CONFIG_OUTPUT) as f:
    config = json.load(f)

X_A = model_df[config["feature_sets"]["A"]]
X_B = model_df[config["feature_sets"]["B"]]
X_D = model_df[config["feature_sets"]["D"]]
y = model_df[config["target"]]
genes = model_df[config["group_column"]]

print("Set A:", X_A.shape)
print("Set B:", X_B.shape)
print("Set D:", X_D.shape)
print("Pathogenic share:", f"{y.mean():.2%}")
print("Genes:", genes.nunique())
