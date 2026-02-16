import numpy as np
from pathlib import Path
import sys

def main():
    """
    Reconstructs the combined BCI IV 2a dataset from individual subject files.
    This is necessary because the combined file was too large for GitHub.
    """
    # Project root is the parent of the 'scripts' directory
    project_root = Path(__file__).resolve().parent.parent
    bci_dir = project_root / "processed" / "bci_iv_2a"

    print(f"Checking for BCI data in: {bci_dir}")

    if not bci_dir.exists():
        print(f"Error: Directory not found: {bci_dir}")
        sys.exit(1)

    # Find individual X and y files
    # Only look for ones that match the pattern A??T_X.npy to avoid picking up the combined one if it exists
    x_files = sorted(list(bci_dir.glob("A??T_X.npy")))
    y_files = sorted(list(bci_dir.glob("A??T_y.npy")))

    if not x_files:
        print("No individual subject files (A??T_X.npy) found.")
        print("Please ensure you have cloned the repository correctly.")
        sys.exit(1)

    print(f"Found {len(x_files)} subject/session files.")

    # Load and concatenate
    all_X = []
    all_y = []

    for x_path, y_path in zip(x_files, y_files):
        print(f"Loading {x_path.name} and {y_path.name}...")
        try:
            X = np.load(x_path)
            y = np.load(y_path)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error loading files: {e}")
            sys.exit(1)

    print("Concatenating data...")
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Save combined files
    out_x = bci_dir / "bci2a_all_subjects_X.npy"
    out_y = bci_dir / "bci2a_all_subjects_y.npy"

    print(f"Saving combined data to {out_x.name}...")
    np.save(out_x, X_all)
    np.save(out_y, y_all)

    print("Success! BCI IV 2a data reconstructed.")
    print(f"Combined shapes: X={X_all.shape}, y={y_all.shape}")

if __name__ == "__main__":
    main()
