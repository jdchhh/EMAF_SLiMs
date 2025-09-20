import os
import pandas as pd
import subprocess
import numpy as np
import torch
import joblib
from Bio import SeqIO
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

HOME_DIR = os.path.expanduser("~")
DB_PATH = os.path.join(HOME_DIR, "PSSMdataset/swissprot/swissprot")
AAINDEX_PATH = os.path.join(HOME_DIR, "AAindex/AAindex1.csv")
PROTBERT_PATH = "/root/autodl-tmp/project_hyf/protBERT"

NUM_ITERATIONS = 3
E_VALUE = 0.001
MAX_SEQ_LENGTH = 5000
SLIDING_WINDOW = [5,7,9,11,21,41]
HIDDEN_SIZE = 1024
SCALE_FEATURE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_pssm(sequence, output_file, db_path=DB_PATH):
    temp_fasta = "temp_seq.fasta"
    temp_blast_out = "temp_blast_output.xml"
    with open(temp_fasta, "w") as f:
        f.write(f">temp_seq\n{sequence}\n")

    cmd = [
        "psiblast",
        "-query", temp_fasta,
        "-db", db_path,
        "-out_ascii_pssm", output_file,
        "-num_iterations", str(NUM_ITERATIONS),
        "-evalue", str(E_VALUE),
        "-out", temp_blast_out,
        "-num_threads", "4"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.remove(temp_fasta)
        os.remove(temp_blast_out)
        raise RuntimeError(f"PSI-BLAST failed: {result.stderr}")

    pssm_data = []
    with open(output_file, "r") as f:
        lines = f.readlines()
        start_idx = 3
        while start_idx < len(lines) and lines[start_idx].strip() != "//":
            parts = lines[start_idx].strip().split()
            if len(parts) >= 22:
                pssm_row = list(map(int, parts[2:22]))
                pssm_data.append(pssm_row)
            start_idx += 1

    os.remove(temp_fasta)
    os.remove(temp_blast_out)

    pssm_matrix = np.array(pssm_data, dtype=np.float32)
    if len(pssm_matrix) != len(sequence):
        raise ValueError(f"PSSM length ({len(pssm_matrix)}) mismatch with sequence length ({len(sequence)})")
    return pssm_matrix


def init_protbert(model_path=PROTBERT_PATH):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return tokenizer, model


def get_semantic_embedding(sequence, tokenizer, model):
    sequence_with_spaces = " ".join(list(sequence))
    inputs = tokenizer(
        sequence_with_spaces,
        return_tensors="pt",
        truncation=True,
        padding=False
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 1:-1, :].squeeze(0).cpu().numpy()

    if len(embedding) != len(sequence):
        raise ValueError(f"Embedding length ({len(embedding)}) mismatch with sequence length ({len(sequence)})")
    return embedding


def load_aaindex(aaindex_path=AAINDEX_PATH):
    aaindex = pd.read_csv(aaindex_path, index_col=0)
    standard_aas = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    aaindex = aaindex.loc[standard_aas].fillna(0)
    scaler = StandardScaler()
    aaindex_scaled = scaler.fit_transform(aaindex)
    return dict(zip(standard_aas, aaindex_scaled))


def get_physicochemical_feature(sequence, aaindex_dict, window_size=SLIDING_WINDOW):
    if window_size % 2 == 0:
        raise ValueError(f"Sliding window size must be odd (current: {window_size})")
    half_window = window_size // 2
    seq_length = len(sequence)

    raw_phy_features = []
    for aa in sequence:
        if aa not in aaindex_dict:
            raw_phy_features.append(np.zeros(544, dtype=np.float32))
        else:
            raw_phy_features.append(aaindex_dict[aa])
    raw_phy_features = np.array(raw_phy_features)

    phy_features = []
    for i in range(seq_length):
        start = max(0, i - half_window)
        end = min(seq_length, i + half_window + 1)
        window_avg = np.mean(raw_phy_features[start:end], axis=0)
        pooled = window_avg.reshape(-1, 5).mean(axis=1)[:108]
        phy_features.append(pooled)

    phy_features = np.array(phy_features, dtype=np.float32)
    if phy_features.shape != (seq_length, 108):
        raise ValueError(f"Physicochemical feature shape ({phy_features.shape}) not as expected ([{seq_length}, 108])")
    return phy_features


def load_sequences(input_file, format="csv", column="sequence"):
    if format.lower() == "csv":
        df = pd.read_csv(input_file)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in CSV file")
        sequences = df[column].tolist()
        seq_ids = [f"seq_{i}" for i in range(len(sequences))]

    elif format.lower() == "fasta":
        records = list(SeqIO.parse(input_file, "fasta"))
        sequences = [str(record.seq) for record in records]
        seq_ids = [record.id for record in records]

    else:
        raise ValueError("Unsupported format: only CSV or FASTA allowed")

    valid_data = [(seq_id, seq) for seq_id, seq in zip(seq_ids, sequences) if len(seq) <= MAX_SEQ_LENGTH]
    valid_ids, valid_seqs = zip(*valid_data) if valid_data else ([], [])

    print(f"=== Sequence Loading Completed ===")
    print(f"Total sequences: {len(sequences)}, Valid sequences (≤{MAX_SEQ_LENGTH}): {len(valid_seqs)}")
    print(f"Valid sequence IDs: {valid_ids}")
    return valid_ids, valid_seqs


def integrate_features(sequence, seq_id, output_dir, tokenizer, model, aaindex_dict):
    seq_output_dir = os.path.join(output_dir, seq_id)
    os.makedirs(seq_output_dir, exist_ok=True)

    print(f"Processing sequence {seq_id} (length: {len(sequence)})...")
    pssm_path = os.path.join(seq_output_dir, f"{seq_id}_pssm.pssm")
    pssm = generate_pssm(sequence, pssm_path)
    semantic_emb = get_semantic_embedding(sequence, tokenizer, model)
    phy_feat = get_physicochemical_feature(sequence, aaindex_dict)

    integrated = np.hstack([semantic_emb, phy_feat, pssm])
    if integrated.shape != (len(sequence), 1152):
        raise ValueError(f"Integrated feature shape ({integrated.shape}) not as expected ([{len(sequence)}, 1152])")

    if SCALE_FEATURE:
        scaler = StandardScaler()
        integrated = scaler.fit_transform(integrated)
        joblib.dump(scaler, os.path.join(seq_output_dir, f"{seq_id}_scaler.joblib"))

    np.save(os.path.join(seq_output_dir, f"{seq_id}_integrated_feat.npy"), integrated)
    print(f"Sequence {seq_id} processed. Results saved to: {seq_output_dir}")
    return integrated


def main():
    parser = argparse.ArgumentParser(
        description="Complete Protein Sequence Preprocessing (Semantic Embedding + Physicochemical Features + PSSM)")
    parser.add_argument("-i", "--input", required=True, help="Input file path (CSV/FASTA)")
    parser.add_argument("-o", "--output", required=True, help="Total output directory")
    parser.add_argument("-f", "--format", default="csv", choices=["csv", "fasta"], help="Input file format")
    parser.add_argument("-c", "--column", default="sequence", help="Sequence column name in CSV (only for CSV format)")
    args = parser.parse_args()

    print("=== Initializing Preprocessing Components ===")
    tokenizer, protbert_model = init_protbert()
    aaindex_dict = load_aaindex()
    os.makedirs(args.output, exist_ok=True)

    seq_ids, sequences = load_sequences(args.input, args.format, args.column)
    if not sequences:
        print("No valid sequences. Exiting program.")
        return

    print("\n=== Starting Batch Preprocessing ===")
    for seq_id, seq in tqdm(zip(seq_ids, sequences), total=len(sequences), desc="Processing Progress"):
        try:
            integrate_features(seq, seq_id, args.output, tokenizer, protbert_model, aaindex_dict)
        except Exception as e:
            print(f"Failed to process sequence {seq_id}: {str(e)}")
            continue

    report_path = os.path.join(args.output, "preprocessing_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Protein Sequence Preprocessing Report\n")
        f.write(f"====================\n")
        f.write(f"Input File: {args.input}\n")
        f.write(f"Input Format: {args.format}\n")
        f.write(f"Total Sequences: {len(sequences) + (len(seq_ids) - len(sequences))}\n")
        f.write(f"Valid Sequences: {len(sequences)}\n")
        f.write(f"Max Sequence Length: {MAX_SEQ_LENGTH}\n")
        f.write(f"Feature Dimension: Semantic Embedding (1024) + Physicochemical (108) + PSSM (20) = 1152D\n")
        f.write(f"Feature Scaling: {'Enabled' if SCALE_FEATURE else 'Disabled'}\n")
        f.write(f"Output Directory: {args.output}\n")
        f.write(f"Completion Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n=== Preprocessing Completed ===")
    print(f"Report saved to: {report_path}")
    print(f"Integrated features saved to: {args.output}/[Sequence ID]/[Sequence ID]_integrated_feat.npy")


if __name__ == "__main__":
    main()